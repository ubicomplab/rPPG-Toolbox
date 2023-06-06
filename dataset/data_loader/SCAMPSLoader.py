"""The dataloader for SCAMPS datasets.

Details for the SCAMPS Dataset see https://github.com/danmcduff/scampsdataset
If you use this dataset, please cite the following publication:
McDuff, Daniel and Wander, Miah and Liu, Xin and Hill, Brian L and Hernandez, Javier and Lester, Jonathan and Baltrusaitis, Tadas
SCAMPS: Synthetics for Camera Measurement of Physiological Signals
in: Conference on Neural Information Processing Systems' 2022
"""
import glob
import json
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import mat73
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm


class SCAMPSLoader(BaseLoader):
    """The data loader for the SCAMPS Processed dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an SCAMPS Processed dataloader.
            Args:
                data_path(string): path of a folder which stores raw video and ground truth biosignal in mat files.
                Each mat file contains a video sequence of resolution of 72x72 and various ground trugh signal.
                e.g., dXsub -> raw/diffnormalized data; d_ppg -> pulse signal, d_br -> resp signal
                -----------------
                     ProcessedData/
                     |   |-- P000001.mat/
                     |   |-- P000002.mat/
                     |   |-- P000003.mat/
                     ...
                -----------------
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)
        self.cached_path = config_data.CACHED_PATH + "_" + self.dataset_name
        self.file_list_path = config_data.FILE_LIST_PATH.split('.')[0] + "_" + self.dataset_name \
                              + os.path.basename(config_data.FILE_LIST_PATH)  # append split name before .csv ext

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For COHFACE dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "*.mat")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = list()
        for data_dir in data_dirs:
            subject = os.path.split(data_dir)[-1]
            dirs.append({"index": subject, "path": data_dir})
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values"""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs
        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []
        for i in choose_range:
            data_dirs_new.append(data_dirs[i])
        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ Invoked by preprocess_dataset() for multi_process. """
        matfile_path = data_dirs[i]['path']
        saved_filename = data_dirs[i]['index']

        # Read Frames
        frames = self.read_video(matfile_path)
        frames = (np.round(frames * 255)).astype(np.uint8)

        # Read Labels
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            bvps = self.read_wave(matfile_path)

        frames_clips, bvps_clips = self.preprocess(
            frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    def preprocess_dataset_backup(self, data_dirs, config_preprocess):
        """Preprocesses the raw data."""
        file_num = len(data_dirs)
        pbar = tqdm(list(range(file_num)))
        for i in pbar:
            matfile_path = data_dirs[i]['path']
            pbar.set_description("Processing %s" % matfile_path)

            # Read Frames
            frames = self.read_video(matfile_path)

            # Read Labels
            if config_preprocess.USE_PSUEDO_PPG_LABEL:
                bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
            else:
                bvps = self.read_wave(matfile_path)
                
            frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
            self.preprocessed_data_len += self.save(frames_clips, bvps_clips, data_dirs[i]['index'])

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3). """
        mat = mat73.loadmat(video_file)
        frames = mat['Xsub']  # load raw frames
        return np.asarray(frames)

    @staticmethod
    def read_wave(wave_file):
        """Reads a bvp signal file."""
        mat = mat73.loadmat(wave_file)
        ppg = mat['d_ppg']  # load raw frames
        return np.asarray(ppg)
