"""The Base Class for data-loading.

Provides a pytorch-style data-loader for end-to-end training pipelines.
Extend the class to support specific datasets.
Dataset already supported:UBFC,PURE and COHFACE
"""
import numpy as np
import os
import cv2
import glob
import re
from torch.utils.data import Dataset
from math import ceil

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
        self.cached_path = os.path.join(config_data.CACHED_PATH, name)
        self.inputs = list()
        self.labels = list()
        self.len = 0
        self.data_format = config_data.DATA_FORMAT
        data_dirs = self.get_data(self.data_path)
        if config_data.DO_PREPROCESS:
            self.preprocess_dataset(data_dirs, config_data.PREPROCESS)
        else:
            self.load()

    def get_data(self, data_path):
        """Returns data directories under the path."""
        return None

    def preprocess_dataset(self, data_dirs, config_preprocess):
        """Parses and preprocesses all data.

        Args:
            config_preprocess(CfgNode): preprocessing settings(ref:config.py).

        """
        pass

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
        else:
            raise ValueError('Unsupported Data Format!')
        data = np.float32(data)
        label = np.float32(label)
        item_path = self.inputs[index]
        item_path_filename = item_path.split('/')[-1]
        split_idx = item_path_filename.index('_')
        filename = item_path_filename[:split_idx]
        chunk_id = item_path_filename[split_idx+6:].split('.')[0]
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
            config_preprocess.DETECTION_LENGTH,
            config_preprocess.W,
            config_preprocess.H,
            config_preprocess.LARGE_FACE_BOX,
            config_preprocess.FACE_DETECT,
            config_preprocess.CROP_FACE,
            config_preprocess.LARGER_BOX_SIZE)
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
                data, bvps, config_preprocess.CLIP_LENGTH)
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
        if (len(face_zone) < 1):
            print("ERROR:No Face Detected")
            result = [0, 0, frame.shape[0], frame.shape[1]]
        elif (len(face_zone) >= 2):
            result = np.argmax(face_zone, axis=0)
            result = face_zone[result[2]]
            print("WARN:More than one faces are detected(Only cropping the biggest one.)")
        else:
            result = face_zone[0]
        if larger_box:
            print("Larger Bounding Box")
            result[0] = max(0, result[0] - (larger_box_size-1.0) / 2 * result[2])
            result[1] = max(0, result[1] - (larger_box_size-1.0) / 2 * result[3])
            result[2] = larger_box_size * result[2]
            result[3] = larger_box_size * result[3]
        return result

    def resize(self, frames, danymic_det, det_length,
               w, h, larger_box, face_detection, crop_face, larger_box_size):
        """Resizes each frame, crops the face area if flag is true."""
        if face_detection:
            if danymic_det:
                det_num = ceil(frames.shape[0] / det_length)
            else:
                det_num = 1
            face_region = list()
            for idx in range(det_num):
                if crop_face:
                    face_region.append(self.facial_detection(frames[det_length*idx], larger_box,larger_box_size))
                else:
                    face_region.append(frames[0])
            face_region_all = np.asarray(face_region, dtype='int')
        else:
            assert (danymic_det == False)           # danymic_det can be True only when face_detection is True.
            face_region_all = [0, 0, frames[0].shape[0], frames[0].shape[1]]

        resize_frames = np.zeros((frames.shape[0], h, w, 3))
        for i in range(0, frames.shape[0]):
            frame = frames[i]
            if danymic_det:
                reference_index = i // det_length
            else:
                reference_index = 0
            if crop_face:
                face_region = face_region_all[reference_index]
                # print('cropping face!!!')
                frame = frame[max(face_region[1],
                                  0):min(face_region[1] + face_region[3],
                                         frame.shape[0]),
                              max(face_region[0],
                                  0):min(face_region[0] + face_region[2],
                                         frame.shape[1])]
            resize_frames[i] = cv2.resize(
                frame, (w, h), interpolation=cv2.INTER_AREA)
        # resize_frames = np.float32(resize_frames) / 255
        # resize_frames[resize_frames > 1] = 1
        # resize_frames[resize_frames < (1 / 255)] = 1 / 255
        return resize_frames

    def chunk(self, frames, bvps, clip_length):
        """Chunks the data into clips."""
        # assert (frames.shape[0] == bvps.shape[0])
        clip_num = frames.shape[0] // clip_length
        frames_clips = [
            frames[i * clip_length:(i + 1) * clip_length] for i in range(clip_num)]
        bvps_clips = [
            bvps[i * clip_length:(i + 1) * clip_length] for i in range(clip_num)]
        return np.array(frames_clips), np.array(bvps_clips)

    def save(self, frames_clips, bvps_clips, filename):
        """Saves the preprocessing data."""
        if (not os.path.exists(self.cached_path)):
            os.makedirs(self.cached_path)
            print(self.cached_path)
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

    def load(self):
        """Loads the preprocessing data."""
        inputs = glob.glob(os.path.join(self.cached_path, "*input*.npy"))
        labels = [input.replace("input", "label") for input in inputs]
        assert (len(inputs) == len(labels))
        self.inputs = inputs
        self.labels = labels
        self.len = len(inputs)

    @staticmethod
    def diff_normalize_data(data):
        """Difference frames and normalization data"""
        n, h, w, c = data.shape
        normalized_len = n - 1
        normalized_data = np.zeros((normalized_len, h, w, c), dtype=np.float32)
        for j in range(normalized_len - 1):
            normalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                data[j + 1, :, :, :] + data[j, :, :, :])
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
        # data[data < 1] = 1
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
