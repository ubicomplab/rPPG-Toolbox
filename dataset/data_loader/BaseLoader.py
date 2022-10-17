"""The Base Class for data-loading.

Provides a pytorch-style data-loader for end-to-end training pipelines.
Extend the class to support specific datasets.
Dataset already supported:UBFC,PURE and COHFACE
"""
import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager
from tqdm import tqdm
import pandas as pd
import csv
from math import ceil

import cv2
import numpy as np
from torch.utils.data import Dataset


class BaseLoader(Dataset):
    """The base class for data loading based on pytorch Dataset.

    The dataloader supports both providing data for pytorch training and common data-preprocessing methods,
    including reading files, resizing each frame, chunking, and video-signal synchronization.
    """

    @staticmethod
    def add_data_loader_args(parser):
        """Adds arguments to parser for training process"""
        parser.add_argument(
            "--cached_path", default=None, type=str)
        parser.add_argument(
            "--preprocess", default=None, action='store_true')
        return parser

    def __init__(self, name, data_path, config_data):
        """Inits dataloader with lists of files.

        Args:
            name(str): name of the dataloader.
            data_path(string): path to the folder containing all data.
            config_data(CfgNode): data settings(ref:config.py).
        """
        self.name = name
        self.data_path = data_path
        self.cached_path = config_data.CACHED_PATH
        self.file_list_path = config_data.FILE_LIST_PATH
        assert (config_data.BEGIN < config_data.END)
        assert (config_data.BEGIN > 0 or config_data.BEGIN == 0)
        assert (config_data.END < 1 or config_data.END == 1)
        if config_data.DATASET == "SCAMPS":
            self.cached_path = config_data.CACHED_PATH + "_" + self.name
            self.file_list_path = config_data.FILE_LIST_PATH[:-4] + "_" + self.name \
                                  + config_data.FILE_LIST_PATH[-4:] # append split name before .csv ext
        print('Cached Data Path', self.cached_path)
        print('File List Path', self.file_list_path)
        self.inputs = list()
        self.labels = list()
        self.len = 0
        self.data_format = config_data.DATA_FORMAT
        data_dirs = self.get_data(self.data_path)
        self.do_preprocess = config_data.DO_PREPROCESS
        if config_data.DO_PREPROCESS:
            self.preprocess_dataset(data_dirs, config_data.PREPROCESS, config_data.BEGIN, config_data.END)
        else:
            print('FILE LIST PATH', self.file_list_path)
            if not os.path.exists(self.file_list_path):
                print('File list does not exist... generating now...')
                self.build_file_list_retroactive(data_dirs, config_data.BEGIN, config_data.END)
                print('File list generated.')
            self.load()
        print(self.name + " dataset len:", self.len)

    def get_data(self, data_path):
        """Returns data directories under the path."""
        return None

    def get_data_subset(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping subjects between splits"""

        return None

    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):
        """Parses and preprocesses all data.

        Args:
            config_preprocess(CfgNode): preprocessing settings(ref:config.py).
        """
        data_dirs = self.get_data_subset(data_dirs, begin, end)
        print("Number of files to preprocess:", len(data_dirs))

        file_list_dict = self.multi_process_manager(data_dirs, config_preprocess)
        self.build_file_list(file_list_dict) # build file list
        self.load() # load all data and corresponding labels (sorted for consistency)

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.inputs)

    def __getitem__(self, index):
        """Returns a clip of video(3,T,W,H) and it's corresponding signals(T)."""
        data = np.load(self.inputs[index])
        label = np.load(self.labels[index])
        if self.data_format == 'NDCHW':
            data = np.transpose(data, (0, 3, 1, 2))
        elif self.data_format == 'NCDHW':
            data = np.transpose(data, (3, 0, 1, 2))
        elif self.data_format == 'NDHWC':
            pass
        else:
            raise ValueError('Unsupported Data Format!')
        data = np.float32(data)
        label = np.float32(label)
        item_path = self.inputs[index]
        item_path_filename = item_path.split('/')[-1]
        split_idx = item_path_filename.index('_')
        filename = item_path_filename[:split_idx]
        chunk_id = item_path_filename[split_idx + 6:].split('.')[0]
        return data, label, filename, chunk_id

    def preprocess(self, frames, bvps, config_preprocess, large_box=False):
        """Preprocesses a pair of data.

        Args:
            frames(np.array): Frames in a video.
            bvps(np.array): Bvp signal labels for a video.
            config_preprocess(CfgNode): preprocessing settings(ref:config.py).
            large_box(bool): Whether to use a large bounding box in face cropping, e.g. in moving situations.
        """
        frames = self.resize(
            frames,
            config_preprocess.DYNAMIC_DETECTION,
            config_preprocess.DYNAMIC_DETECTION_FREQUENCY,
            config_preprocess.W,
            config_preprocess.H,
            config_preprocess.LARGE_FACE_BOX,
            config_preprocess.CROP_FACE,
            config_preprocess.LARGE_BOX_COEF)
        # data_type
        data = list()
        for data_type in config_preprocess.DATA_TYPE:
            f_c = frames.copy()
            if data_type == "Raw":
                data.append(f_c[:-1, :, :, :])
            elif data_type == "Normalized":
                data.append(BaseLoader.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                data.append(BaseLoader.standardized_data(f_c)[:-1, :, :, :])
            else:
                raise ValueError("Unsupported data type!")
        data = np.concatenate(data, axis=3)
        if config_preprocess.LABEL_TYPE == "Raw":
            bvps = bvps[:-1]
        elif config_preprocess.LABEL_TYPE == "Normalized":
            bvps = BaseLoader.diff_normalize_label(bvps)
        elif config_preprocess.LABEL_TYPE == "Standardized":
            bvps = BaseLoader.standardized_label(bvps)[:-1]
        else:
            raise ValueError("Unsupported label type!")

        if config_preprocess.DO_CHUNK:
            frames_clips, bvps_clips = self.chunk(
                data, bvps, config_preprocess.CHUNK_LENGTH)
        else:
            frames_clips = np.array([data])
            bvps_clips = np.array([bvps])

        return frames_clips, bvps_clips

    def facial_detection(self, frame, larger_box=False, larger_box_size=1.0):
        """Conducts face detection on a single frame.
        Sets larger_box=True for larger bounding box, e.g. moving trials."""
        detector = cv2.CascadeClassifier(
            './dataset/haarcascade_frontalface_default.xml')
        face_zone = detector.detectMultiScale(frame)
        if len(face_zone) < 1:
            print("ERROR:No Face Detected")
            result = [0, 0, frame.shape[0], frame.shape[1]]
        elif len(face_zone) >= 2:
            result = np.argmax(face_zone, axis=0)
            result = face_zone[result[2]]
            print("WARN:More than one faces are detected(Only cropping the biggest one.)")
        else:
            result = face_zone[0]
        if larger_box:
            print("Larger Bounding Box")
            result[0] = max(0, result[0] - (larger_box_size - 1.0) / 2 * result[2])
            result[1] = max(0, result[1] - (larger_box_size - 1.0) / 2 * result[3])
            result[2] = larger_box_size * result[2]
            result[3] = larger_box_size * result[3]
        return result

    def resize(self, frames, dynamic_det, det_length,
               w, h, larger_box, crop_face, larger_box_size):
        """

        :param dynamic_det: If False, it will use the only first frame to do facial detection and
                            the detected result will be used for all frames to do cropping and resizing.
                            If True, it will implement facial detection every "det_length" frames,
                            [i*det_length, (i+1)*det_length] of frames will use the i-th detected region to do cropping.
        :param det_length: the interval of dynamic detection
        :param larger_box: whether enlarge the detected region.
        :param crop_face:  whether crop the frames.
        :param larger_box_size: the coefficient of the larger region(height and weight),
                            the middle point of the detected region will stay still during the process of enlarging.
        """
        if dynamic_det:
            det_num = ceil(frames.shape[0] / det_length)
        else:
            det_num = 1
        face_region = list()

        # obtain detection region. it will do facial detection every "det_length" frames, totally "det_num" times.
        for idx in range(det_num):
            if crop_face:
                face_region.append(self.facial_detection(frames[det_length * idx], larger_box, larger_box_size))
            else:
                # if crop_face:False, the face_region will be the whole frame, namely cropping nothing.
                face_region.append([0, 0, frames.shape[1], frames.shape[2]])
        face_region_all = np.asarray(face_region, dtype='int')
        resize_frames = np.zeros((frames.shape[0], h, w, 3))

        # if dynamic_det: True, the frame under processing will use the (i // det_length)-th facial region.
        # if dynamic_det: False, the frame will only use the first region obtrained from the first frame.
        for i in range(0, frames.shape[0]):
            frame = frames[i]
            if dynamic_det:
                reference_index = i // det_length
            else:
                reference_index = 0
            if crop_face:
                face_region = face_region_all[reference_index]
                frame = frame[max(face_region[1], 0):min(face_region[1] + face_region[3], frame.shape[0]),
                        max(face_region[0], 0):min(face_region[0] + face_region[2], frame.shape[1])]
            resize_frames[i] = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        return resize_frames

    def chunk(self, frames, bvps, chunk_length):
        """Chunks the data into clips."""
        clip_num = frames.shape[0] // chunk_length
        frames_clips = [
            frames[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        bvps_clips = [
            bvps[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        return np.array(frames_clips), np.array(bvps_clips)

    def save(self, frames_clips, bvps_clips, filename):
        """Saves the preprocessing data."""
        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)
        count = 0
        for i in range(len(bvps_clips)):
            assert (len(self.inputs) == len(self.labels))
            input_path_name = self.cached_path + os.sep + \
                              "{0}_input{1}.npy".format(filename, str(count))
            label_path_name = self.cached_path + os.sep + \
                              "{0}_label{1}.npy".format(filename, str(count))
            self.inputs.append(input_path_name)
            self.labels.append(label_path_name)
            np.save(input_path_name, frames_clips[i])
            np.save(label_path_name, bvps_clips[i])
            count += 1
        return count

    def save_multi_process(self, frames_clips, bvps_clips, filename):
        """Saves the preprocessing data."""
        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)
        count = 0
        input_path_name_list = []
        label_path_name_list = []
        for i in range(len(bvps_clips)):
            assert (len(self.inputs) == len(self.labels))
            input_path_name = self.cached_path + os.sep + \
                              "{0}_input{1}.npy".format(filename, str(count))
            label_path_name = self.cached_path + os.sep + \
                              "{0}_label{1}.npy".format(filename, str(count))
            input_path_name_list.append(input_path_name)
            label_path_name_list.append(label_path_name)
            np.save(input_path_name, frames_clips[i])
            np.save(label_path_name, bvps_clips[i])
            count += 1
        return count, input_path_name_list, label_path_name_list

    def multi_process_manager(self, data_dirs, config_preprocess):

        file_num = len(data_dirs)
        choose_range = choose_range = range(0, file_num)
        pbar = tqdm(list(choose_range))

        # shared data resource
        manager = Manager()
        file_list_dict = manager.dict()
        p_list = []
        running_num = 0

        for i in choose_range:
            process_flag = True
            while process_flag:  # ensure that every i creates a process
                if running_num < 8:  # in case of too many processes
                    p = Process(target=self.preprocess_dataset_subprocess, \
                                args=(data_dirs, config_preprocess, i, file_list_dict))
                    p.start()
                    p_list.append(p)
                    running_num += 1
                    process_flag = False
                for p_ in p_list:
                    if not p_.is_alive():
                        p_list.remove(p_)
                        p_.join()
                        running_num -= 1
                        pbar.update(1)
        # join all processes
        for p_ in p_list:
            p_.join()
            pbar.update(1)
        pbar.close()

        return file_list_dict

    def build_file_list(self, file_list_dict):
        """build file list"""

        file_list = []
        for process_num, file_paths in file_list_dict.items():
            file_list = file_list + file_paths

        if not file_list:
            raise ValueError(self.name, 'No files in file list')

        file_list_df = pd.DataFrame(file_list, columns = ['input_files'])   
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)
        file_list_df.to_csv(self.file_list_path)

    def build_file_list_retroactive(self, data_dirs, begin, end):

        # get data split
        data_dirs = self.get_data_subset(data_dirs, begin, end)

        # generate a list of unique raw-data file names 
        filename_list = []
        for i in range(len(data_dirs)):
            filename_list.append(data_dirs[i]['index'])
        filename_list = list(set(filename_list)) # ensure all indexes are unique

        # generate a list of all preprocessed / chunked data files 
        file_list = []
        for fname in filename_list:
            processed_file_data = list(glob.glob(self.cached_path + os.sep + "{0}_input*.npy".format(fname)))
            file_list += processed_file_data

        if not file_list:
            raise ValueError(self.name, 'File list empty. Check preprocessed data folder exists and is not empty.')

        file_list_df = pd.DataFrame(file_list, columns = ['input_files'])   
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)
        file_list_df.to_csv(self.file_list_path)

    def load(self):
        """Loads the preprocessing data listed in the file list"""

        file_list_path = self.file_list_path # get list of files in 

        # TO DO: Insert functionality to generate file list if it does not already exist

        file_list_df = pd.read_csv(file_list_path) 
        inputs = file_list_df['input_files'].tolist()
        if inputs == []:
            raise ValueError(self.name + ' dataset loading data error!')
        inputs = sorted(inputs)  # sort input file name list
        labels = [input_file.replace("input", "label") for input_file in inputs]
        self.inputs = inputs
        self.labels = labels
        self.len = len(inputs)
        print("Loaded data len:", self.len)

    @staticmethod
    def diff_normalize_data(data):
        """Difference frames and normalization data"""
        n, h, w, c = data.shape
        normalized_len = n - 1
        normalized_data = np.zeros((normalized_len, h, w, c), dtype=np.float32)
        for j in range(normalized_len - 1):
            normalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                    data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
        normalized_data = normalized_data / np.std(normalized_data)
        normalized_data[np.isnan(normalized_data)] = 0
        return normalized_data

    @staticmethod
    def diff_normalize_label(label):
        """Difference frames and normalization labels"""
        diff_label = np.diff(label, axis=0)
        normalized_label = diff_label / np.std(diff_label)
        normalized_label[np.isnan(normalized_label)] = 0
        return normalized_label

    @staticmethod
    def standardized_data(data):
        """Difference frames and normalization data"""
        data = data - np.mean(data)
        data = data / np.std(data)
        data[np.isnan(data)] = 0
        return data

    @staticmethod
    def standardized_label(label):
        label = label - np.mean(label)
        label = label / np.std(label)
        label[np.isnan(label)] = 0
        return label
