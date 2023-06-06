"""The dataloader for the UBFC-PHYS dataset.

Details for the UBFC-PHYS Dataset see https://sites.google.com/view/ybenezeth/ubfc-phys.
If you use this dataset, please cite this paper:
R. Meziati Sabour, Y. Benezeth, P. De Oliveira, J. Chappé, F. Yang. 
"UBFC-Phys: A Multimodal Database For Psychophysiological Studies Of Social Stress", 
IEEE Transactions on Affective Computing, 2021.
"""
import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
import csv
import pandas as pd

class UBFCPHYSLoader(BaseLoader):
    """The data loader for the UBFC-PHYS dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an UBFC-PHYS dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- s1/
                     |       |-- vid_s1_T1.avi
                     |       |-- vid_s1_T2.avi
                     |       |-- vid_s1_T3.avi
                     |       |...
                     |       |-- bvp_s1_T1.csv
                     |       |-- bvp_s1_T2.csv
                     |       |-- bvp_s1_T3.csv
                     |   |-- s2/
                     |       |-- vid_s2_T1.avi
                     |       |-- vid_s2_T2.avi
                     |       |-- vid_s2_T3.avi
                     |       |...
                     |       |-- bvp_s2_T1.csv
                     |       |-- bvp_s2_T2.csv
                     |       |-- bvp_s2_T3.csv
                     |...
                     |   |-- sn/
                     |       |-- vid_sn_T1.avi
                     |       |-- vid_sn_T2.avi
                     |       |-- vid_sn_T3.avi
                     |       |...
                     |       |-- bvp_sn_T1.csv
                     |       |-- bvp_sn_T2.csv
                     |       |-- bvp_sn_T3.csv
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        self.filtering = config_data.FILTERING
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For UBFC-PHYS dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "s*" + os.sep + "*.avi")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = [{"index": re.search(
            'vid_(.*).avi', data_dir).group(1), "path": data_dir} for data_dir in data_dirs]
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """   invoked by preprocess_dataset for multi_process.   """
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        # Read Frames
        frames = self.read_video(
            os.path.join(data_dirs[i]['path']))

        # Read Labels
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            bvps = self.read_wave(
                os.path.join(os.path.dirname(data_dirs[i]['path']),"bvp_{0}.csv".format(saved_filename)))

        bvps = BaseLoader.resample_ppg(bvps, frames.shape[0])
            
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    def load_preprocessed_data(self):
        """ Loads the preprocessed data listed in the file list.

        Args:
            None
        Returns:
            None
        """
        file_list_path = self.file_list_path  # get list of files in
        file_list_df = pd.read_csv(file_list_path)
        base_inputs = file_list_df['input_files'].tolist()
        filtered_inputs = []

        for input in base_inputs:
            input_name = input.split(os.sep)[-1].split('.')[0].rsplit('_', 1)[0]
            if self.filtering.USE_EXCLUSION_LIST and input_name in self.filtering.EXCLUSION_LIST :
                # Skip loading the input as it's in the exclusion list
                continue
            if self.filtering.SELECT_TASKS and not any(task in input_name for task in self.filtering.TASK_LIST):
                # Skip loading the input as it's not in the task list
                continue
            filtered_inputs.append(input)

        if not filtered_inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error!')
        
        filtered_inputs = sorted(filtered_inputs)  # sort input file name list
        labels = [input_file.replace("input", "label") for input_file in filtered_inputs]
        self.inputs = filtered_inputs
        self.labels = labels
        self.preprocessed_data_len = len(filtered_inputs)

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T,H,W,3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        bvp = []
        with open(bvp_file, "r") as f:
            d = csv.reader(f)
            for row in d:
                bvp.append(float(row[0]))
        return np.asarray(bvp)
