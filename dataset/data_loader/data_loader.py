"""The Base Class for data-loading.

Provides a pytorch-style data-loader for end-to-end training pipelines.
Extend the class to support specific datasets.
Dataset already supported:UBFC(TODO: add more dataset)
"""
import numpy as np
import os
import cv2
import glob
from torch.utils.data import Dataset


class data_loader(Dataset):
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
            "--preprocessing", default=True, type=bool)
        return parser

    def __init__(self, name,data_dirs,config_data):
        """Inits dataloader with lists of files.

        Args:
            name(str): name of the dataloader.
            config_data(CfgNode): data settings(ref:config.py).
        """
        self.name = name
        self.data_dirs = data_dirs
        self.cached_path = os.path.join(config_data.CACHED_PATH, name)
        self.inputs = list()
        self.labels = list()
        self.len = 0
        if config_data.DO_PREPROCESS:
            self.preprocess_dataset(config_data.PREPROCESS)
        else:
            self.load()

    def preprocess_dataset(self, config_preprocess):
        """Parses and preprocesses all data.

        Args:
            config_preprocess(CfgNode): preprocessing settings(ref:config.py).

        """
        pass

    @staticmethod
    def preprocess(frames, bvps, config_preprocess, large_box):
        """Preprocesses a pair of data.

        Args:
            config_preprocess(CfgNode): preprocessing settings(ref:config.py).

        """
        print(config_preprocess)
        frames = data_loader.resize(
            frames,
            config_preprocess.W,
            config_preprocess.H,
            config_preprocess.CROP_FACE,
            large_box)
        frames_clips, bvps_clips = data_loader.chunk(
            frames, bvps, config_preprocess.CLIP_LENGTH)
        return frames_clips, bvps_clips

    @staticmethod
    def facial_detection(frame, larger_box=False):
        """Conducts face detection on a single frame.
        Sets larger_box=True for larger bounding box, e.g. moving trials."""
        detector = cv2.CascadeClassifier(
            './dataset/haarcascade_frontalface_default.xml')
        face_zone = detector.detectMultiScale(frame)
        result = face_zone[0]
        if (len(face_zone) < 1):
            print("ERROR:No Face Detected")
        if (len(face_zone) >= 2):
            result = np.argmax(face_zone, axis=0)
            print("WARN:More than one faces are detected(Only cropping the biggest one.)")
            result = face_zone[result[2]]
        if larger_box:
            print("Larger Bounding Box")
            result[0] = max(0, result[0] - 0.4 * result[2])
            result[1] = max(0, result[1] - 0.1 * result[2])
            result[2] = 1.8 * result[2]
            result[3] = 1.2 * result[3]
        return result

    @staticmethod
    def resize(frames, w, h, crop_face=True, larger_box=False):
        """Resizes each frame, crops the face area if flag is true."""
        face_region = data_loader.facial_detection(frames[0], larger_box)
        resize_frames = np.zeros((frames.shape[0], h, w, 3))
        for i in range(0, frames.shape[0]):
            frame = frames[i]
            if crop_face:
                frame = frame[max(face_region[1],
                                  0):min(face_region[1] + face_region[3],
                                         frame.shape[0]),
                        max(face_region[0],
                            0):min(face_region[0] + face_region[2],
                                   frame.shape[1])]
                # view the cropped area.
                # cv2.imshow("frame",frame)
                # cv2.waitKey(0)
            resize_frames[i] = cv2.resize(frame, (w, h))
        return resize_frames

    @staticmethod
    def chunk(frames, bvps, clip_length):
        """Chunks the data into clips."""
        assert (frames.shape[0] == bvps.shape[0])
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
        count = 0
        filename = os.path.split(filename)[-1]
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
        inputs = glob.glob(os.path.join(self.cached_path,"*input*.npy"))
        labels = glob.glob(os.path.join(self.cached_path,"*label*.npy"))
        assert(len(inputs)==len(labels))
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
            normalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / \
                                            (data[j + 1, :, :, :] + data[j, :, :, :])
        normalized_data = normalized_data / np.std(normalized_data)
        return normalized_data

    @staticmethod
    def diff_normalize_label(label):
        """Difference frames and normalization labels"""
        diff_label = np.diff(label, axis=0)
        return diff_label / np.std(diff_label)