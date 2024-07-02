""" The dataloader for MMPD datasets.

"""

import os
import cv2
import glob
import numpy as np
import re

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
    """The data loader for the MMPD dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an MMPD dataloader.
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
        self.info = config_data.INFO
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
        frames, bvps, light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup \
            = self.read_mat(data_dirs[i]['path'])

        saved_filename = 'subject' + str(data_dirs[i]['subject'])
        saved_filename += f'_L{light}_MO{motion}_E{exercise}_S{skin_color}_GE{gender}_GL{glasser}_H{hair_cover}_MA{makeup}'

        frames = (np.round(frames * 255)).astype(np.uint8)
        target_length = frames.shape[0]
        bvps = BaseLoader.resample_ppg(bvps, target_length)
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)

        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    def read_mat(self, mat_file):
        try:
            mat = sio.loadmat(mat_file)
        except:
            for _ in range(20):
                print(mat_file)
        frames = np.array(mat['video'])

        if self.config_data.PREPROCESS.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else: 
            bvps = np.array(mat['GT_ppg']).T.reshape(-1)

        light = mat['light']
        motion = mat['motion']
        exercise = mat['exercise']
        skin_color = mat['skin_color']
        gender = mat['gender']
        glasser = mat['glasser']
        hair_cover = mat['hair_cover']
        makeup = mat['makeup']
        information = [light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup]
        
        light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup = self.get_information(information)

        return frames, bvps, light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup
    
    def load_preprocessed_data(self):
        """ Loads the preprocessed data listed in the file list.

        Args:
            None
        Returns:
            None
        """
        file_list_path = self.file_list_path  # get list of files in
        file_list_df = pd.read_csv(file_list_path)
        inputs_temp = file_list_df['input_files'].tolist()
        inputs = []
        for each_input in inputs_temp:
            info = each_input.split(os.sep)[-1].split('_')
            light = int(info[1][-1])
            motion = int(info[2][-1])
            exercise = int(info[3][-1])
            skin_color = int(info[4][-1])
            gender = int(info[5][-1])
            glasser = int(info[6][-1])
            hair_cover = int(info[7][-1])
            makeup = int(info[8][-1])
            if (light in self.info.LIGHT) and (motion in self.info.MOTION) and \
                (exercise in self.info.EXERCISE) and (skin_color in self.info.SKIN_COLOR) and \
                (gender in self.info.GENDER) and (glasser in self.info.GLASSER) and \
                (hair_cover in self.info.HAIR_COVER) and (makeup in self.info.MAKEUP):
                inputs.append(each_input)
        if not inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error!')
        inputs = sorted(inputs)  # sort input file name list
        labels = [input_file.replace("input", "label") for input_file in inputs]
        self.inputs = inputs
        self.labels = labels
        self.preprocessed_data_len = len(inputs)

    @staticmethod
    def get_information(information):
        light = ''
        if information[0] == 'LED-low':
            light = 1
        elif information[0] == 'LED-high':
            light = 2
        elif information[0] == 'Incandescent':
            light = 3
        elif information[0] == 'Nature':
            light = 4
        else:
            raise ValueError("Error with MMPD or Mini-MMPD dataset labels! "
            "The following lighting label is not supported: {0}".format(information[0]))

        motion = ''
        if information[1] == 'Stationary' or information[1] == 'Stationary (after exercise)':
            motion = 1
        elif information[1] == 'Rotation':
            motion = 2
        elif information[1] == 'Talking':
            motion = 3
        # 'Watching Videos' is an erroneous label from older versions of the MMPD dataset, 
        #  it should be handled as 'Walking'.
        elif information[1] == 'Walking' or information[1] == 'Watching Videos':
            motion = 4
        else:
            raise ValueError("Error with MMPD or Mini-MMPD dataset labels! " 
            "The following motion label is not supported: {0}".format(information[1]))
        
        exercise = ''
        if information[2] == 'True':
            exercise = 1
        elif information[2] == 'False':
            exercise = 2
        else:
            raise ValueError("Error with MMPD or Mini-MMPD dataset labels! "
            "The following exercise label is not supported: {0}".format(information[2]))

        skin_color = information[3][0][0]

        if skin_color != 3 and skin_color != 4 and skin_color != 5 and skin_color != 6:
            raise ValueError("Error with MMPD or Mini-MMPD dataset labels! "
            "The following skin_color label is not supported: {0}".format(information[3][0][0]))

        gender = ''
        if information[4] == 'male':
            gender = 1
        elif information[4] == 'female':
            gender = 2
        else:
            raise ValueError("Error with MMPD or Mini-MMPD dataset labels! "
            "The following gender label is not supported: {0}".format(information[4]))

        glasser = ''
        if information[5] == 'True':
            glasser = 1
        elif information[5] == 'False':
            glasser = 2
        else:
            raise ValueError("Error with MMPD or Mini-MMPD dataset labels! "
            "The following glasser label is not supported: {0}".format(information[5]))

        hair_cover = ''
        if information[6] == 'True':
            hair_cover = 1
        elif information[6] == 'False':
            hair_cover = 2
        else:
            raise ValueError("Error with MMPD or Mini-MMPD dataset labels! "
            "The following hair_cover label is not supported: {0}".format(information[6]))
        
        makeup = ''
        if information[7] == 'True':
            makeup = 1
        elif information[7] == 'False':
            makeup = 2
        else:
            raise ValueError("Error with MMPD or Mini-MMPD dataset labels! "
            "The following makeup label is not supported: {0}".format(information[7]))

        return light, motion ,exercise, skin_color, gender, glasser, hair_cover, makeup

