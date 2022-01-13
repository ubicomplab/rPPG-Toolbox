"""The Base Class for data-loading.

Provides a pytorch-style data-loader for end-to-end training pipelines.
Extend the class to support specific datasets.
Dataset already supported:UBFC(TODO: add more dataset)
"""
import numpy as np
import os
import cv2
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
            "--cached_dir", default="preprocessed", type=str)
        parser.add_argument(
            "--preprocessing", default=True, type=bool)
        return parser

    def __init__(self, cached_dir):
        """Inits dataloader with lists of files.

        Args:
            cached_dir(str): The directory where preprocessing results are stored.
        """
        self.cached_dir = cached_dir
        self.inputs = []
        self.labels = []
        self.len = 0

    def preprocess(self, w, h, clip_length, crop_face):
        """Preprocesses the raw data.

        Args:
            w,h(int): The shape of each frame after preprocessing.
            clip_length(int): The length of each clip.
            crop__face(bool): Whether to crop the face area
        """
        pass

    def read_video(self, video_path):
        """Reads the file, returns frames."""
        pass

    def read_bvp(self, bvp_path):
        """Reads the file, return bvp signals."""
        pass

    def facial_detection(self, frame):
        """Conducts face detection on a single frame. """
        detector = cv2.CascadeClassifier(
            './dataset/haarcascade_frontalface_default.xml')
        face_zone = detector.detectMultiScale(frame)
        if (len(face_zone) < 1):
            print("ERROR:No Face Detected")
        if (len(face_zone) >= 2):
            print("WARN:More than one faces are detected(Only cropping one face)")
        return face_zone[0]

    def resize(self, frames, w, h, detect_face=True):
        """Resizes each frame, crops the face area if flag is true."""
        face_region = self.facial_detection(frames[0])
        resize_frames = np.zeros((frames.shape[0], h, w, 3))
        for i in range(0, frames.shape[0]):
            frame = frames[i]
            if detect_face:
                frame = frame[face_region[1]:face_region[1] + face_region[3],
                              face_region[0]:face_region[0] + face_region[2]]
            resize_frames[i] = cv2.resize(frame, (w, h))
        return resize_frames

    def chunk(self, frames, bvps, clip_length):
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
        if (not os.path.exists(self.cached_dir)):
            os.mkdir(self.cached_dir)
        count = 0
        filename = os.path.split(filename)[-1]
        for i in range(len(bvps_clips)):
            assert (len(self.inputs) == len(self.labels))
            input_path_name = self.cached_dir + os.sep + \
                "{0}_input{1}.npy".format(filename, str(count))
            label_path_name = self.cached_dir + os.sep + \
                "{0}_label{1}.npy".format(filename, str(count))
            self.inputs.append(input_path_name)
            self.labels.append(label_path_name)
            np.save(input_path_name, frames_clips[i])
            np.save(label_path_name, bvps_clips[i])
            count += 1
        return count
