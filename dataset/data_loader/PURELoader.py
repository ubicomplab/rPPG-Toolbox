"""The dataloader for PURE datasets.

Details for the PURE Dataset see https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure
If you use this dataset, please cite the following publication:
Stricker, R., Müller, S., Gross, H.-M.
Non-contact Video-based Pulse Rate Measurement on a Mobile Service Robot
in: Proc. 23st IEEE Int. Symposium on Robot and Human Interactive Communication (Ro-Man 2014), Edinburgh, Scotland, UK, pp. 1056 - 1062, IEEE 2014
"""
import os
import cv2
import glob
import json
import numpy as np
import re
from dataset.data_loader.BaseLoader import BaseLoader
from utils.utils import sample
import glob
from multiprocessing import Pool, Process, Value, Array, Manager
from tqdm import tqdm


class PURELoader(BaseLoader):
    """The data loader for the PURE dataset."""


    def __init__(self, name, data_path, config_data):
        """Initializes an PURE dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- 01-01/
                     |      |-- 01-01/
                     |      |-- 01-01.json
                     |   |-- 01-02/
                     |      |-- 01-02/
                     |      |-- 01-02.json
                     |...
                     |   |-- ii-jj/
                     |      |-- ii-jj/
                     |      |-- ii-jj.json
                -----------------
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)


    def get_data(self, data_path):
        """Returns data directories under the path(For PURE dataset)."""

        data_dirs = glob.glob(data_path + os.sep + "*-*")
        if not data_dirs:
            raise ValueError(self.name+ " dataset get data error!")
        dirs = list()
        for data_dir in data_dirs:
            subject_trail_val = os.path.split(data_dir)[-1].replace('-', '')
            index = int(subject_trail_val)
            subject = int(subject_trail_val[0:2])
            dirs.append({"index": index, "path": data_dir, "subject": subject})
        return dirs


    def get_data_subset(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping subjects between splits"""

        # get info about the dataset: subject list and num vids per subject
        data_info = dict()
        for data in data_dirs:

            subject = data['subject']
            data_dir = data['path']
            index = data['index']

            # creates a dictionary of data_dirs indexed by subject number
            if subject not in data_info: # if subject not in the data info dictionary
                data_info[subject] = [] # make an emplty list for that subject
            # append a tuple of the filename, subject num, trial num, and chunk num
            data_info[subject].append({"index": index, "path": data_dir, "subject": subject})

        subj_list = list(data_info.keys()) # all subjects by number ID (1-27)
        subj_list.sort()
        num_subjs = len(subj_list) # number of unique subjects

        # get split of data set (depending on start / end)
        subj_range = list(range(0,num_subjs))
        if (begin !=0 or end !=1):
            subj_range = list(range(int(begin*num_subjs), int(end*num_subjs)))
        print('used subject ids for split:', [subj_list[i] for i in subj_range])

        # compile file list
        file_info_list = []
        for i in subj_range:
            subj_num = subj_list[i]
            subj_files = data_info[subj_num]
            file_info_list += subj_files # add file information to file_list (tuple of fname, subj ID, trial num, chunk num)
        
        return file_info_list


    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i):
        """   invoked by preprocess_dataset for multi_process.   """
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']
        frames = self.read_video(
            os.path.join(
                data_dirs[i]['path'],
                filename, ""))
        bvps = self.read_wave(
            os.path.join(
                data_dirs[i]['path'],
                "{0}.json".format(filename)))
        bvps = sample(bvps, frames.shape[0])
        frames_clips, bvps_clips = self.preprocess(
            frames, bvps, config_preprocess, config_preprocess.LARGE_FACE_BOX)
        count, input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)

    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):
        """Preprocesses the raw data."""
        file_num = len(data_dirs)
        print("file_num:", file_num)
        choose_range = range(0, file_num)

        if begin != 0 or end != 1:
            #choose_range = range(int(begin * file_num), int(end * file_num))
            data_dirs = self.get_data_subset(data_dirs, begin, end)
            choose_range = range(0, len(data_dirs))
        print(choose_range)

        pbar = tqdm(list(choose_range))
        # multi_process
        p_list = []
        running_num = 0
        for i in choose_range:
            process_flag = True
            while process_flag:            # ensure that every i creates a process
                if running_num < 16:       # in case of too many processes
                    p = Process(target=self.preprocess_dataset_subprocess, args=(data_dirs,config_preprocess,i))
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
        # append all data path and update the length of data
        inputs = glob.glob(os.path.join(self.cached_path, "*input*.npy"))
        if not inputs:
            raise ValueError(self.name + ' dataset loading data error!')
        labels = [input.replace("input", "label") for input in inputs]
        assert (len(inputs) == len(labels))
        self.inputs = inputs
        self.labels = labels
        self.len = len(inputs)

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T,H,W,3) """
        frames = list()
        all_png = sorted(glob.glob(video_file + '*.png'))
        for png_path in all_png:
            img = cv2.imread(png_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with open(bvp_file, "r") as f:
            labels = json.load(f)
            waves = [label["Value"]["waveform"]
                     for label in labels["/FullPackage"]]
        return np.asarray(waves)
