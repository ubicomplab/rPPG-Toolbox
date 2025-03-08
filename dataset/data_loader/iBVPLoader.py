"""The dataloader for iBVP datasets.

Details for the iBVP Dataset see https://doi.org/10.3390/electronics13071334
If you use this dataset, please cite the following publications:

Joshi, Jitesh, and Youngjun Cho. 2024. "iBVP Dataset: RGB-Thermal rPPG Dataset with High Resolution Signal Quality Labels" Electronics 13, no. 7: 1334. https://doi.org/10.3390/electronics13071334 

Joshi, Jitesh, Katherine Wang, and Youngjun Cho. 2023. "PhysioKit: An Open-Source, Low-Cost Physiological Computing Toolkit for Single- and Multi-User Studies" Sensors 23, no. 19: 8244. https://doi.org/10.3390/s23198244 

"""
import glob
import os
import re

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
import pandas as pd


class iBVPLoader(BaseLoader):
    """The data loader for the iBVP dataset."""

    def __init__(self, name, data_path, config_data, device=None):
        """Initializes an iBVP dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "iBVP_Dataset" for below dataset structure:
                -----------------
                     iBVP_Dataset/
                     |   |-- p01_a/
                     |      |-- p01_a_rgb/
                     |      |-- p01_a_t/
                     |      |-- p01_a_bvp.csv
                     |   |-- p01_b/
                     |      |-- p01_b_rgb/
                     |      |-- p01_b_t/
                     |      |-- p01_b_bvp.csv
                     |...
                     |   |-- pii_x/
                     |      |-- pii_x_rgb/
                     |      |-- pii_x_t/
                     |      |-- pii_x_bvp.csv

                -----------------
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For iBVP dataset)."""

        data_dirs = glob.glob(data_path + os.sep + "*_*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = list()
        for data_dir in data_dirs:
            subject_trail_val = os.path.split(data_dir)[-1].replace('_', '')
            index = subject_trail_val
            subject = subject_trail_val[0:3]
            dirs.append({"index": index, "path": data_dir, "subject": subject})
        return dirs

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
            data_dirs_new += subj_files  # add file information to file_list (tuple of fname, subj ID, trial num,
            # chunk num)

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ Invoked by preprocess_dataset for multi_process. """
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        # Read Frames
        if 'None' in config_preprocess.DATA_AUG:
            # Utilize dataset-specific function to read video
            
            if config_preprocess.IBVP.DATA_MODE == "T":
                frames = self.read_video(
                    os.path.join(data_dirs[i]['path'], "{0}_t".format(filename), ""))
            
            elif config_preprocess.IBVP.DATA_MODE == "RGBT":
                rgb_frames = self.read_video(
                    os.path.join(data_dirs[i]['path'], "{0}_rgb".format(filename), ""))

                thermal_frames = self.read_thermal_video(
                    os.path.join(data_dirs[i]['path'], "{0}_t".format(filename), ""))
            else:
                frames = self.read_video(
                    os.path.join(data_dirs[i]['path'], "{0}_rgb".format(filename), ""))

        elif 'Motion' in config_preprocess.DATA_AUG:
            # Utilize general function to read video in .npy format
            frames = self.read_npy_video(
                glob.glob(os.path.join(data_dirs[i]['path'], filename, '*.npy')))
        else:
            raise ValueError(f'Unsupported DATA_AUG specified for {self.dataset_name} dataset! Received {config_preprocess.DATA_AUG}.')

        # Read Labels
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            bvps, sq_vec = self.read_wave(
                os.path.join(data_dirs[i]['path'], "{0}_bvp.csv".format(filename)))

        if "RGBT" in config_preprocess.IBVP.DATA_MODE:
            rgb_length, rgb_height, rgb_width, rgb_ch = rgb_frames.shape
            thermal_length, t_height, t_width, t_ch = thermal_frames.shape
            target_length = min(rgb_length, thermal_length)
            rgb_frames = rgb_frames[:target_length, ...]
            thermal_frames = thermal_frames[:target_length, :rgb_height, :, :]      #rgb_height = 480, thermal height = 512, so reducing thermal height to match with RGB.
            frames = np.concatenate([rgb_frames, thermal_frames], axis=-1)
        else:
            target_length = frames.shape[0]

        bvps = BaseLoader.resample_ppg(bvps, target_length)
        sq_vec = BaseLoader.resample_ppg(sq_vec, target_length)

        # print(type(frames), frames.shape)
        # print(type(bvps), bvps.shape)
        # exit()
        # Discard frames based on Signal Quality
        del_idx = sq_vec <= 0.3
        frames = np.delete(frames, del_idx, axis=0)
        bvps = np.delete(bvps, del_idx, axis=0)
        sq_vec = np.delete(sq_vec, del_idx, axis=0)

        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        frames = list()
        all_bmp = sorted(glob.glob(video_file + '*.bmp'))
        for bmp_path in all_bmp:
            img = cv2.imread(bmp_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        return np.asarray(frames)

    @staticmethod
    def read_thermal_video(video_file):
        """Reads a video file, returns frames(T, H, W, 1) """
        im_width = 640
        im_height = 512
        frames = list()
        all_raw = sorted(glob.glob(video_file + '*.raw'))
        for raw_path in all_raw:
            thermal_matrix = np.fromfile(raw_path, dtype=np.uint16, count=im_width *
                                        im_height).reshape(im_height, im_width)
            thermal_matrix = thermal_matrix.astype(np.float32)
            thermal_matrix = (thermal_matrix * 0.04) - 273.15
            frames.append(thermal_matrix)
        frames = np.expand_dims(frames, axis=-1)
        return np.asarray(frames)


    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with open(bvp_file, "r") as f:
            labels = pd.read_csv(f).to_numpy()
            waves = labels[:, 0]
            sq_vec = labels[:, 3]   #SQ2
        return waves, sq_vec
