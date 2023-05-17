"""The dataloader for BP4D+ datasets.

Details for the BP4D+ Dataset see https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html
If you use this dataset, please cite the following publications:
Xing Zhang, Lijun Yin, Jeff Cohn, Shaun Canavan, Michael Reale, Andy Horowitz, Peng Liu, and Jeff Girard
“BP4D-Spontaneous: A high resolution spontaneous 3D dynamic facial expression database”
Image and Vision Computing, 32 (2014), pp. 692-706  (special issue of the Best of FG13)

AND

Xing Zhang, Lijun Yin, Jeff Cohn, Shaun Canavan, Michael Reale, Andy Horowitz, and Peng Liu
“A high resolution spontaneous 3D dynamic facial expression database”
The 10th IEEE International Conference on Automatic Face and Gesture Recognition (FG13),  April, 2013. 
"""

import glob
import zipfile
import os
import re

import cv2
from skimage.util import img_as_float
import numpy as np
import pandas as pd
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm

from dataset.data_loader.BaseLoader import BaseLoader


class BP4DPlusLoader(BaseLoader):
    """The data loader for the BP4D+ dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an BP4D+ dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                    RawData/
                    |   |-- 2D+3D/
                    |       |-- F001.zip/
                    |       |-- F002.zip
                    |       |...
                    |   |-- 2DFeatures/
                    |       |-- F001_T1.mat
                    |       |-- F001_T2.mat
                    |       |...
                    |   |-- 3DFeatures/
                    |       |-- F001_T1.mat
                    |       |-- F001_T2.mat
                    |       |...
                    |   |-- AUCoding/
                    |       |-- AU_INT/
                    |            |-- AU06/
                    |               |-- F001_T1_AU06.csv
                    |               |...
                    |           |...
                    |       |-- AU_OCC/
                    |           |-- F00_T1.csv 
                    |           |...
                    |   |-- IRFeatures/
                    |       |-- F001_T1.txt
                    |       |...
                    |   |-- Physiology/
                    |       |-- F001/
                    |           |-- T1/
                    |               |-- BP_mmHg.txt
                    |               |-- microsiemens.txt
                    |               |--LA Mean BP_mmHg.txt
                    |               |--LA Systolic BP_mmHg.txt
                    |               |-- BP Dia_mmHg.txt
                    |               |-- Pulse Rate_BPM.txt
                    |               |-- Resp_Volts.txt
                    |               |-- Respiration Rate_BPM.txt
                    |       |...
                    |   |-- Thermal/
                    |       |-- F001/
                    |           |-- T1.mv
                    |           |...
                    |       |...
                    |   |-- BP4D+UserGuide_v0.2.pdf
                -----------------
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)

    def adjust_data_dirs(self, data_dirs):
        """ Reads data folder and only preprocess files that have not already been preprocessed."""

        cached_path = self.cached_path
        file_list = glob.glob(os.path.join(cached_path, '*label*.npy'))
        trial_list = [f.replace(cached_path, '').split('_')[0].replace(os.sep, '') for f in file_list]
        trial_list = list(set(trial_list)) # get a list of completed video trials

        adjusted_data_dirs = []
        for d in data_dirs:
            idx = d['index']

            if not idx in trial_list: # if trial has already been processed
                adjusted_data_dirs.append(d)

        return adjusted_data_dirs

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For PURE dataset).
        NOTE: There are 5 videos in BP4D+ with length of less than 180 frames."""

        # get all subj trials in dataset
        f_subj_trials = glob.glob(os.path.join(data_path, "Physiology", "F*", "T*"))
        m_subj_trials = glob.glob(os.path.join(data_path, "Physiology", "M*", "T*"))
        subj_trials = f_subj_trials + m_subj_trials

        # split path up into information (subject, trial, etc.)
        data_dirs = list()
        for trial_path in subj_trials:
            trial_data = trial_path.split(os.sep)
            index = trial_data[-2] + trial_data[-1] # should be of format: F008T8
            trial = trial_data[-1] # trial number 
            subj_sex = index[0] # subject biological sex
            subject = index[0:4] # subject number (by sex) F001

            # this filename exists but the file does not... weird...
            if index == 'F042T11':
                continue
            
            # append information to data dirs list
            data_dirs.append({"index": index, "path": data_path, "subject": subject})

        # adjust data directories if dataset already partially preprocessed
        # data_dirs = self.adjust_data_dirs(data_dirs)

        # return data dirs
        return data_dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping subjects between splits"""

        # return the full directory
        if begin == 0 and end == 1:
            return data_dirs

        # get info about the dataset: subject list and num vids per subject
        data_info = dict()
        for data in data_dirs:
            subject = data['subject']
            data_dir = data['path']
            index = data['index']
            # creates a dictionary of data_dirs indexed by subject number
            if subject not in data_info:  # if subject not in the data info dictionary
                data_info[subject] = []  # make an emplty list for that subject
            # append a tuple of the filename, subject num, trial num, and chunk num
            data_info[subject].append({"index": index, "path": data_dir, "subject": subject})

        subj_list = list(data_info.keys())  # all subjects by number ID 
        subj_list = sorted(subj_list)
        num_subjs = len(subj_list)  # number of unique subjects

        # get split of data set (depending on start / end)
        subj_range = list(range(0, num_subjs))
        if begin != 0 or end != 1:
            subj_range = list(range(int(begin * num_subjs), int(end * num_subjs)))

        # compile file list
        data_dirs_new = []
        for i in subj_range:
            subj_num = subj_list[i]
            subj_files = data_info[subj_num]
            data_dirs_new += subj_files 

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ Invoked by preprocess_dataset for multi_process. """
        saved_filename = data_dirs[i]['index']
        
        frames = self.read_video(data_dirs[i], config_preprocess)
        bvps = self.read_wave(data_dirs[i], config_preprocess, frames)

        target_length = frames.shape[0]
        bvps = BaseLoader.resample_ppg(bvps, target_length)
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    def read_video(self, data_dir, config_preprocess):
        """Reads a video file, returns frames(T, H, W, 3) """

        video_file = os.path.join(data_dir['path'], '2D+3D', data_dir['subject']+'.zip') # video fname
        trial = data_dir['index'].split('T')[-1] # trial number (1-10) of form
        trial = 'T' + trial

        # grab each frame from zip file
        imgzip = open(video_file)
        zipfile_path = video_file

        cnt = 0
        frames=list()
        with zipfile.ZipFile(zipfile_path, "r") as zippedImgs:
            for ele in zippedImgs.namelist():
                ext = os.path.splitext(ele)[-1]
                ele_task = str(ele).split('/')[1]

                if ext == '.jpg' and ele_task == trial:
                    data = zippedImgs.read(ele)
                    frame = cv2.imdecode(np.fromstring(data, np.uint8), cv2.IMREAD_COLOR)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # downsample frames (otherwise processing time becomes WAY TOO LONG)
                    dim_w = min(2*config_preprocess.W, frame.shape[1])
                    dim_h = int(dim_w * frame.shape[0]/frame.shape[1])
                    frame = cv2.resize(frame, (dim_w,dim_h), interpolation=cv2.INTER_AREA)
                    frame = np.expand_dims(frame, axis=0)

                    if cnt == 0:
                        frames = frame
                    else:
                        frames = np.concatenate((frames, frame), axis=0)
                    cnt += 1
        
        if cnt == 0:
            raise ValueError('EMPTY VIDEO', data_dir['index'])
        return np.asarray(frames)

    def read_wave(self, data_dir, config_preprocess, frames):
        """Reads a bvp signal file."""

        # generate PPG psuedo labels
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            label = self.generate_pos_psuedo_labels(frames, fs=25)

        # read in physiological labels txt file data
        else:
            data_path = data_dir['path']
            subject = data_dir['subject'] # of format F008
            trial = data_dir['index'].split('T')[-1] # of format T5 or T10
            trial = 'T' + trial
            base_path = os.path.join(data_path, "Physiology", subject, trial)
            label = np.array(pd.read_csv(os.path.join(base_path, "BP_mmHg.txt")).to_numpy().flatten())

        return label
