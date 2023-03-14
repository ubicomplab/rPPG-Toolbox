""" The dataloader for MMPD datasets.

"""

import os
import cv2
import glob
import numpy as np
import re

from scipy.__config__ import get_info
from .BaseLoader import BaseLoader
from multiprocessing import Pool, Process, Value, Array, Manager
from tqdm import tqdm
import pandas as pd
import scipy.io as sio
import sys
import itertools
from warnings import simplefilter
# from dataset.data_loader.process_tool import *
from scipy.signal import butter, filtfilt

simplefilter(action='ignore', category=FutureWarning)

class MMPDLoader(BaseLoader):
    """The data loader for the UBFC dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an UBFC dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be mat_dataset" for below dataset structure:
                -----------------
                     mat_dataset/
                     |   |-- subject1/
                     |       |-- p1_0.mat
                     |       |-- p1_1.mat
                     |       |...
                     |   |-- subject2/
                     |       |-- p2_0.mat
                     |       |-- p2_1.mat
                     |       |...
                     |...
                     |   |-- subjectn/
                     |       |-- pn_0.mat
                     |       |-- pn_1.mat
                     |       |...
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, raw_data_path):
        """Returns data directories under the path(For MMPD dataset)."""

        data_dirs = glob.glob(raw_data_path + os.sep + 'subject*')
        if not data_dirs:
            raise ValueError(self.dataset_name + ' data paths empty!')
        dirs = list()
        for data_dir in data_dirs:
            subject = int(os.path.split(data_dir)[-1][7:])
            mat_dirs = os.listdir(data_dir)
            for mat_dir in mat_dirs:
                index = mat_dir.split('_')[-1].split('.')[0]
                dirs.append({'index': index, 
                             'path': data_dir+os.sep+mat_dir,
                             'subject': subject})
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping subjects between splits"""

        # return the full directory
        if begin == 0 and end == 1:
            return data_dirs
        
        data_info = dict()
        for data in data_dirs:
            index = data['index']
            data_dir = data['path']
            subject = data['subject']
            # creates a dictionary of data_dirs indexed by subject number
            if subject not in data_info:  # if subject not in the data info dictionary
                data_info[subject] = list()  # make an emplty list for that subject
            # append a tuple of the filename, subject num, trial num, and chunk num
            data_info[subject].append(data)
        subj_list = list(data_info.keys())  # all subjects by number ID
        subj_list = sorted(subj_list)
        num_subjs = len(subj_list)  # number of unique subjects

        # get split of data set (depending on start / end)
        subj_range = list(range(num_subjs))
        if begin != 0 or end != 1:
            subj_range = list(range(int(begin*num_subjs), int(end*num_subjs)))
        print('used subject ids for split:', [subj_list[i] for i in subj_range])

        # compile file list
        data_dirs_new = list()
        for i in subj_range:
            subj_num = subj_list[i]
            data_dirs_new += data_info[subj_num]

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """  Invoked by preprocess_dataset for multi_process.   """
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = 'subject' + str(data_dirs[i]['subject']) + '_' + str(data_dirs[i]['index'])

        frames, bvps = self.read_mat(data_dirs[i]['path'])
        frames = (np.round(frames * 255)).astype(np.uint8)
        target_length = frames.shape[0]
        bvps = BaseLoader.resample_ppg(bvps, target_length)
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)

        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_mat(mat_file):
        try:
            mat = sio.loadmat(mat_file)
        except:
            for _ in range(20):
                print(mat_file)
        frames = np.array(mat['video'])
        bvps = np.array(mat['GT_ppg']).T.reshape(-1)
        return frames, bvps

