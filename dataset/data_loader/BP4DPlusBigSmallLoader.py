"""The dataloader for BP4D+ Big Small datasets. This dataloader was adapted from the following git repository 
based on rPPG Toolbox: https://github.com/girishvn/BigSmall

Details on the BigSmall model can be found here: https://girishvn.github.io/BigSmall/
Details for the BP4D+ Dataset see https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html

If you use this dataset, please cite the following publications:

Xing Zhang, Lijun Yin, Jeff Cohn, Shaun Canavan, Michael Reale, Andy Horowitz, Peng Liu, and Jeff Girard
“BP4D-Spontaneous: A high resolution spontaneous 3D dynamic facial expression database”
Image and Vision Computing, 32 (2014), pp. 692-706  (special issue of the Best of FG13)

AND

Xing Zhang, Lijun Yin, Jeff Cohn, Shaun Canavan, Michael Reale, Andy Horowitz, and Peng Liu
“A high resolution spontaneous 3D dynamic facial expression database”
The 10th IEEE International Conference on Automatic Face and Gesture Recognition (FG13),  April, 2013. 

If you use the BigSmall model or preprocessing please cite the following publication:

Girish Narayanswamy, Yujia Liu, Yuzhe Yang, Chengqian Ma, Xin Liu, Daniel McDuff, and Shwetak Patel
"BigSmall: Efficient Multi-Task Learning for Disparate Spatial and Temporal Physiological Measurements"
arXiv:2303.11573 (https://arxiv.org/abs/2303.11573)

"""

import glob
import zipfile
import os
import re

import cv2
from skimage.util import img_as_float
import numpy as np
import pandas as pd
import pickle 

from unsupervised_methods.methods import POS_WANG
from unsupervised_methods import utils
from scipy import signal
from scipy import sparse
import math
from math import ceil

from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm

from dataset.data_loader.BaseLoader import BaseLoader


class BP4DPlusBigSmallLoader(BaseLoader):
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

        self.inputs = list()
        self.labels = list()
        self.dataset_name = name
        self.raw_data_path = data_path
        self.cached_path = config_data.CACHED_PATH
        self.file_list_path = config_data.FILE_LIST_PATH
        self.preprocessed_data_len = 0
        self.data_format = config_data.DATA_FORMAT
        self.do_preprocess = config_data.DO_PREPROCESS
        
        assert (config_data.BEGIN < config_data.END)
        assert (config_data.BEGIN > 0 or config_data.BEGIN == 0)
        assert (config_data.END < 1 or config_data.END == 1)

        if config_data.DO_PREPROCESS:
            self.raw_data_dirs = self.get_raw_data(self.raw_data_path)
            self.preprocess_dataset(self.raw_data_dirs, config_data, config_data.BEGIN, config_data.END)
        else:
            if not os.path.exists(self.cached_path):
                raise ValueError(self.dataset_name,
                                 'Please set DO_PREPROCESS to True. Preprocessed directory does not exist!')
            if not os.path.exists(self.file_list_path):
                print('File list does not exist... generating now...')
                self.raw_data_dirs = self.get_raw_data(self.raw_data_path)
                self.build_file_list_retroactive(self.raw_data_dirs, config_data.BEGIN,
                                                 config_data.END, config_data)
                print('File list generated.', end='\n\n')

            self.load()
        print('Cached Data Path', self.cached_path, end='\n\n')
        print('File List Path', self.file_list_path)
        print(f" {self.dataset_name} Preprocessed Dataset Length: {self.preprocessed_data_len}", end='\n\n')


    def preprocess_dataset(self, data_dirs, config_data, begin, end):
        print('Starting Preprocessing...')

        # GET DATASET INFORMATION (PATHS AND OTHER META DATA REGARDING ALL VIDEO TRIALS)
        data_dirs = self.split_raw_data(data_dirs, begin, end)  # partition dataset 

        # REMOVE ALREADY PREPROCESSED SUBJECTS
        data_dirs = self.adjust_data_dirs(data_dirs, config_data)

        # CREATE CACHED DATA PATH
        cached_path = config_data.CACHED_PATH
        if not os.path.exists(cached_path):
            os.makedirs(cached_path, exist_ok=True)

        # READ RAW DATA, PREPROCESS, AND SAVE PROCESSED DATA FILES
        file_list_dict = self.multi_process_manager(data_dirs, config_data)

        self.build_file_list_retroactive(self.raw_data_dirs, config_data.BEGIN,
                                        config_data.END, config_data)  # build file list
        
        self.load()  # load all data and corresponding labels (sorted for consistency)
        print("Total Number of raw files preprocessed:", len(data_dirs), end='\n\n')
        print("Num loaded files", self.preprocessed_data_len)



    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping subjects between splits"""

        # return the full directory
        if begin == 0 and end == 1:
            return data_dirs

        # get info about the dataset: subject list and num vids per subject
        data_info = dict()
        for data in data_dirs:
            idx_subj = data['index'][0:4]
            subject = data['subject']
            data_path = data['path']
            index = data['index']
            trial = data['trial']
            subj_sex = data['sex']
            # creates a dictionary of data_dirs indexed by subject number
            if idx_subj not in data_info:  # if subject not in the data info dictionary
                data_info[idx_subj] = []  # make an emplty list for that subject
            # append a tuple of the filename, subject num, trial num, and chunk num
            data_info[idx_subj].append({"index": index, "path": data_path, "subject": subject, "trial": trial, "sex": subj_sex})

        subj_list = list(data_info.keys())  # all subjects by number ID (1-27)
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
        


    def get_raw_data(self, data_path):
        """Returns data directories under the path(For PURE dataset)."""

        # GET ALL SUBJECT TRIALS IN DATASET
        f_subj_trials = glob.glob(os.path.join(data_path, "Physiology", "F*", "T*"))
        m_subj_trials = glob.glob(os.path.join(data_path, "Physiology", "M*", "T*"))
        subj_trials = f_subj_trials + m_subj_trials

        # SPLIT PATH UP INTO INFORMATION (SUBJECT, TRIAL, ETC.)
        data_dirs = list()
        for trial_path in subj_trials:
            trial_data = trial_path.split(os.sep)
            index = trial_data[-2] + trial_data[-1] # should be of format: F008T8
            trial = trial_data[-1] # trial number 
            subj_sex = index[0] # subject biological sex
            subject = int(index[1:4]) # subject number (by sex)

            # If processesing AU Subset only process trials T1, T6, T7, T8 (only ones that have AU labels)
            if not trial in ['T1', 'T6', 'T7', 'T8']:
                continue

            if index == 'F041T7': # data sample has mismatch length for video frames and AU labels
                continue
            
            # append information to data dirs list
            data_dirs.append({"index": index, "path": data_path, "subject": subject, "trial": trial, "sex": subj_sex})

        # RETURN DATA DIRS 
        return data_dirs

    
    def adjust_data_dirs(self, data_dirs, config_preprocess):
        """ Reads data folder and only preprocess files that have not already been preprocessed."""

        cached_path = config_preprocess.CACHED_PATH
        file_list = glob.glob(os.path.join(cached_path, '*label*.npy'))
        trial_list = [f.replace(cached_path, '').split('_')[0].replace(os.sep, '') for f in file_list]
        trial_list = list(set(trial_list)) # get a list of completed video trials

        adjusted_data_dirs = []
        for d in data_dirs:
            idx = d['index']

            if not idx in trial_list: # if trial has already been processed
                adjusted_data_dirs.append(d)

        return adjusted_data_dirs


    def preprocess_dataset_subprocess(self, data_dirs, config_data, i, file_list_dict):
        """ invoked by preprocess_dataset for multi_process """

        data_dir_info = data_dirs[i] # get data raw data file path 
        saved_filename = data_dirs[i]['index'] # get subject and trial in format of  FXXXTXX

        # CONSTRUCT DATA DICTIONARY FOR VIDEO TRIAL
        data_dict = self.construct_data_dict(data_dir_info, config_data) # construct a dictionary of ALL labels and video frames (of equal length)
        data_dict = self.generate_pos_psuedo_labels(data_dict, fs=config_data.FS)
        
        # SEPERATE DATA INTO VIDEO FRAMES AND LABELS ARRAY
        frames = self.read_video(data_dict) # read in the video frames
        labels = self.read_labels(data_dict) # read in video labels 
        if frames.shape[0] != labels.shape[0]: # check if data and labels are the same length
            raise ValueError(' Preprocessing dataset subprocess: frame and label time axis not the same')

        # PREPROCESS VIDEO FRAMES AND LABELS (eg. DIFF-NORM, RAW_STD)
        big_clips, small_clips, labels_clips = self.preprocess(frames, labels, config_data)

        # SAVE PREPROCESSED FILE CHUNKS
        count, input_name_list, label_name_list = self.save_multi_process(big_clips, small_clips, labels_clips, saved_filename, config_data)

        file_list_dict[i] = input_name_list



    def generate_pos_psuedo_labels(self, data_dict, fs=30):
        """Generated POS-based PPG Psuedo Labels For Training

        Args:
            frames(List[array]): a video frames.
            fs(int or float): Sampling rate of video
        Returns:
            env_norm_bvp: Hilbert envlope normalized POS PPG signal, filtered are HR frequency
        """

        frames = data_dict['X']

        # GENERATE POS PPG SIGNAL
        WinSec = 1.6
        RGB = POS_WANG._process_video(frames)
        N = RGB.shape[0]
        H = np.zeros((1, N))
        l = math.ceil(WinSec * fs)

        for n in range(N):
            m = n - l
            if m >= 0:
                Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
                Cn = np.mat(Cn).H
                S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
                h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
                mean_h = np.mean(h)
                for temp in range(h.shape[1]):
                    h[0, temp] = h[0, temp] - mean_h
                H[0, m:n] = H[0, m:n] + (h[0])

        bvp = H
        bvp = utils.detrend(np.mat(bvp).H, 100)
        bvp = np.asarray(np.transpose(bvp))[0]

        # AGGRESSIVELY FILTER PPG SIGNAL
        hr_arr = data_dict['HR_bpm'] # get hr freq from GT label
        avg_hr_bpm = np.sum(hr_arr)/len(hr_arr) # calculate avg hr for the entire trial
        hr_freq = avg_hr_bpm / 60 # divide beats per min by 60, to get beats pers secone
        halfband = 20 / fs # half bandwith to account for HR variation (accounts for +/- 20 bpm variation from mean HR)

        # MAX BANDWIDTH [0.70, 3]Hz = [42, 180]BPM (BANDWIDTH MAY BE SMALLER)
        min_freq = hr_freq - halfband # calculate min cutoff frequency
        if min_freq < 0.70:
            min_freq = 0.70
        max_freq = hr_freq + halfband # calculate max cutoff frequency
        if max_freq > 3:
            max_freq = 3

        # FILTER POS PPG W/ 2nd ORDER BUTTERWORTH FILTER
        b, a = signal.butter(2, [(min_freq) / fs * 2, (max_freq) / fs * 2], btype='bandpass')
        pos_bvp = signal.filtfilt(b, a, bvp.astype(np.double))

        # APPLY HILBERT NORMALIZATION TO NORMALIZE PPG AMPLITUDE
        analytic_signal = signal.hilbert(pos_bvp)
        amplitude_envelope = np.abs(analytic_signal)
        env_norm_bvp = pos_bvp/amplitude_envelope

        data_dict['pos_bvp'] = pos_bvp
        data_dict['pos_env_norm_bvp'] = env_norm_bvp

        return data_dict # return data dict w/ POS psuedo labels



    def construct_data_dict(self, data_dir_info, config_data):

        # GET TRIAL NUMBER 
        trial = data_dir_info['trial']

        # BUILD DICTIONARY TO STORE FRAMES AND LABELS
        data_dict = dict()

        # READ IN RAW VIDEO FRAMES
        data_dict = self.read_raw_vid_frames(data_dir_info, config_data, data_dict)

        # READ IN RAW PHYSIOLOGICAL SIGNAL LABELS 
        data_dict = self.read_raw_phys_labels(data_dir_info, data_dict)

        # READ IN ACTION UNIT (AU) LABELS (if trial in [1, 6, 7, 8]: trials w/ AU labels)
        if trial in ['T1', 'T6', 'T7', 'T8']:
            data_dict, start_np_idx, end_np_idx = self.read_au_labels(data_dir_info, config_data, data_dict)

            # CROP DATAFRAME W/ AU START END
            data_dict = self.crop_au_subset_data(data_dict, start_np_idx, end_np_idx)

        # FRAMES AND LABELS SHOULD BE OF THE SAME LENGTH
        shape_mismatch = False
        for k in data_dict.keys():
            if not data_dict[k].shape[0] == data_dict['X'].shape[0]:
                print('Shape Mismatch', k, data_dict[k].shape[0], 'Frames Len', data_dict['X'].shape[0])
                shape_mismatch  = True

        if shape_mismatch:        
            raise ValueError('Shape Mismatch')

        return data_dict
    


    def downsample_frame(self, frame, dim_h=144, dim_w=144):

        if dim_h == dim_w: # square crop
            vidLxL = cv2.resize(frame[int((frame.shape[0]-frame.shape[1])):,:,:], (dim_h,dim_w), interpolation=cv2.INTER_AREA)
        else:
            vidLxL = cv2.resize(frame, (dim_h,dim_w), interpolation=cv2.INTER_AREA)

        return vidLxL



    def read_raw_vid_frames(self, data_dir_info, config_data, data_dict):
        data_path = data_dir_info['path']
        subject_trial = data_dir_info['index'][0:4]
        trial = data_dir_info['trial']

        # GRAB EACH FRAME FROM ZIP FILE
        imgzip = open(os.path.join(data_path, '2D+3D', subject_trial+'.zip'))
        zipfile_path = os.path.join(data_path, '2D+3D', subject_trial+'.zip')

        cnt = 0

        with zipfile.ZipFile(zipfile_path, "r") as zippedImgs:
            for ele in zippedImgs.namelist():
                ext = os.path.splitext(ele)[-1]
                ele_task = str(ele).split('/')[1]

                if ext == '.jpg' and ele_task == trial:
                    data = zippedImgs.read(ele)
                    frame = cv2.imdecode(np.fromstring(data, np.uint8), cv2.IMREAD_COLOR)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    dim_h = config_data.PREPROCESS.BIGSMALL.RESIZE.BIG_H
                    dim_w = config_data.PREPROCESS.BIGSMALL.RESIZE.BIG_W

                    frame = self.downsample_frame(frame, dim_h=dim_h, dim_w=dim_w) # downsample frames (otherwise processing time becomes WAY TOO LONG)
                    frame = np.expand_dims(frame, axis=0)

                    # If frames are empty
                    if cnt == 0:
                        frames = frame
                    else:
                        frames = np.concatenate((frames, frame), axis=0)
                    cnt += 1
        
        if cnt == 0:
            raise ValueError('EMPTY VIDEO', data_dir_info['index'])
    
        data_dict['X'] = np.asarray(frames)
        return data_dict


    def read_raw_phys_labels(self, data_dir_info, data_dict):

        data_path = data_dir_info['path']
        subject = data_dir_info['index'][0:4] # of format F008
        trial = data_dir_info['trial'] # of format T05
        base_path = os.path.join(data_path, "Physiology", subject, trial)

        len_Xsub = data_dict['X'].shape[0]

        # READ IN PHYSIOLOGICAL LABELS TXT FILE DATA
        try:
            bp_wave = pd.read_csv(os.path.join(base_path, "BP_mmHg.txt")).to_numpy().flatten()
            HR_bpm = pd.read_csv(os.path.join(base_path, "Pulse Rate_BPM.txt")).to_numpy().flatten()
            resp_wave = pd.read_csv(os.path.join(base_path, "Resp_Volts.txt")).to_numpy().flatten()
            resp_bpm = pd.read_csv(os.path.join(base_path, "Respiration Rate_BPM.txt")).to_numpy().flatten()
            mean_BP = pd.read_csv(os.path.join(base_path, "LA Mean BP_mmHg.txt")).to_numpy().flatten()
            sys_BP = pd.read_csv(os.path.join(base_path, "LA Systolic BP_mmHg.txt")).to_numpy().flatten()
            dia_BP = pd.read_csv(os.path.join(base_path, "BP Dia_mmHg.txt")).to_numpy().flatten()
            eda = pd.read_csv(os.path.join(base_path, "EDA_microsiemens.txt")).to_numpy().flatten()
        except FileNotFoundError:
            print('Label File Not Found At Basepath', base_path)
            return

        # RESIZE SIGNALS TO LENGTH OF X (FRAMES) AND CONVERT TO NPY ARRAY
        bp_wave = np.interp(np.linspace(0, len(bp_wave), len_Xsub), np.arange(0, len(bp_wave)), bp_wave)
        HR_bpm = np.interp(np.linspace(0, len(HR_bpm), len_Xsub), np.arange(0, len(HR_bpm)), HR_bpm)
        resp_wave = np.interp(np.linspace(0, len(resp_wave), len_Xsub), np.arange(0, len(resp_wave)), resp_wave)
        resp_bpm = np.interp(np.linspace(0, len(resp_bpm), len_Xsub), np.arange(0, len(resp_bpm)), resp_bpm)
        mean_BP = np.interp(np.linspace(0, len(mean_BP), len_Xsub), np.arange(0, len(mean_BP)), mean_BP)
        sys_BP = np.interp(np.linspace(0, len(sys_BP), len_Xsub), np.arange(0, len(sys_BP)), sys_BP)
        dia_BP = np.interp(np.linspace(0, len(dia_BP), len_Xsub), np.arange(0, len(dia_BP)), dia_BP)
        eda = np.interp(np.linspace(0, len(eda), len_Xsub), np.arange(0, len(eda)), eda)

        data_dict['bp_wave'] = bp_wave
        data_dict['HR_bpm'] = HR_bpm
        data_dict['mean_bp'] = mean_BP
        data_dict['systolic_bp'] = sys_BP
        data_dict['diastolic_bp'] = dia_BP
        data_dict['resp_wave'] = resp_wave
        data_dict['resp_bpm'] = resp_bpm
        data_dict['eda'] = eda
        return data_dict  



    def read_au_labels(self, data_dir_info, config_data, data_dict):

        # DATA PATH INFO    
        subj_idx = data_dir_info['index']
        base_path = config_data.DATA_PATH
        AU_OCC_url = os.path.join(base_path, 'AUCoding', "AU_OCC", subj_idx[0:4] + '_' + subj_idx[4:] + '.csv')

        # DATA CHUNK LENGTH
        frame_shape = data_dict['X'].shape[0]

        # READ IN AU CSV FILE
        AUs = pd.read_csv(AU_OCC_url, header = 0).to_numpy()

        # NOTE: START AND END FRAMES ARE 1-INDEXED
        start_frame = AUs[0,0] # first frame w/ AU encoding
        end_frame = AUs[AUs.shape[0] - 1, 0] # last frame w/ AU encoding

        # ENCODED AUs
        AU_num = [1, 2, 4, 5, 6, 7, 9, 10, 11, 
                12, 13, 14, 15, 16, 17, 18, 19, 20,
                22, 23, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
        AU_int_num = [6, 10, 12, 14, 17] # AU w/ intensity encoding (0-5)
    
        # ITERATE THROUGH ENCODED AUs
        for au_idx, au in enumerate(AU_num): # Iterate through list of AUs

            # Define AU str name
            if au < 10:
                AU_key = 'AU' + '0' + str(au)
            else:
                AU_key = 'AU' + str(au)

            # GET SPECIFIC ACTION UNIT DATA
            aucoding = AUs[:, au_idx + 1] # indx + 1 as first row/column is index

            if start_frame > 1: # indx + 1 as first row/column is 1-indexed
                # pad the previous frame with -1
                aucoding = np.pad(aucoding, (start_frame - 1, 0), 'constant', constant_values = (-1, -1))
            if end_frame < frame_shape:
                # pad the following frame with -1 as well
                aucoding = np.pad(aucoding, (0, frame_shape - end_frame), 'constant', constant_values = (-1, -1))

            # Save out info to dict
            data_dict[AU_key] = aucoding

            # READ IN INTENSITY (INT) ENCODED AUs
            if au in AU_int_num:
                AU_INT_url = os.path.join(base_path, 'AUCoding', 'AU_INT', AU_key, subj_idx[0:4] + '_' + subj_idx[4:] + '_' + AU_key + '.csv')
                AUs_int = pd.read_csv(AU_INT_url, header = None).to_numpy() # read in each csv file
                assert (AUs_int.shape[0] == AUs.shape[0]) # ensure int encoding same length as binary encoding
                aucoding_int = AUs_int[:, 1]
                if start_frame > 1:
                    # pad the previous frame with -1
                    aucoding_int = np.pad(aucoding_int, (start_frame - 1, 0), 'constant', constant_values = (-1, -1))
                if end_frame < frame_shape:
                    # pad the following frame with -1
                    aucoding_int = np.pad(aucoding_int, (0, frame_shape - end_frame), 'constant', constant_values = (-1, -1))

                # Save out info to dict
                AU_int_key = AU_key + 'int'
                data_dict[AU_int_key] = aucoding_int

        # return start crop index if using AU subset data
        start_np_idx = start_frame - 1 
        end_np_idx = end_frame - 1
        return data_dict, start_np_idx, end_np_idx
        


    def crop_au_subset_data(self, data_dict, start, end):

        keys = data_dict.keys()

        # Iterate through video frames ad labels and crop based off start and end frame
        for k in keys:
            data_dict[k] = data_dict[k][start:end+1] # start and end frames are inclusive 

        return data_dict
    


    # GET VIDEO FRAMES FROM DATA DICTIONARY
    def read_video(self, data_dict):
        """ Reads a video file, returns frames (N,H,W,3) """
        frames = data_dict['X']
        return np.asarray(frames)



    # GET VIDEO LABELS FROM DATA DICTIONARY AND FORMAT AS ARRAY
    def read_labels(self, data_dict):
        """Reads labels corresponding to video file."""
        f = data_dict
        keys = list(f.keys())
        data_len = f['X'].shape[0] # get the video data length
        keys.remove('X') # remove X from the processed keys (not a label)

        # Init labels array
        labels = np.ones((data_len, 49)) # 47 tasks from original dataset, and added psuedo labels: 'pos_bvp','pos_env_norm_bvp'
        labels = -1*labels # make all values -1 originally

        # LABELS BY INDEX IN OUTPUT LABELS NPY ARRAY
        # 0: bp_wave, 1: hr_bpm, 2: systolic_bp, 3: diastolic_bp, 4: mean_bp,
        # 5: resp_wave, 6: resp_bpm, 7: eda, [8,47]: AUs, 'pos_bvp', 'pos_env_norm_bvp'
        labels_order_list = ['bp_wave', 'HR_bpm', 'systolic_bp', 'diastolic_bp', 'mean_bp', 'resp_wave', 'resp_bpm', 'eda', 
                                'AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU06int', 'AU07', 'AU09', 'AU10', 'AU10int', 'AU11', 'AU12', 'AU12int', 
                                'AU13', 'AU14', 'AU14int', 'AU15', 'AU16', 'AU17', 'AU17int', 'AU18', 'AU19', 'AU20', 'AU22', 'AU23', 'AU24', 
                                'AU27', 'AU28', 'AU29', 'AU30', 'AU31', 'AU32', 'AU33', 'AU34', 'AU35', 'AU36', 'AU37', 'AU38', 'AU39', 
                                'pos_bvp','pos_env_norm_bvp']

        # ADDING LABELS TO DATA ARRAY
        # If Label DNE Then Array Is -1 Filled For That Label
        # Note: BP4D does not have AU labels for all trials: These fields are thus COMPLETELY -1 filled for these trials
        for i in range(len(labels_order_list)):
            if labels_order_list[i] in keys:
                labels[:, i] = f[labels_order_list[i]]

        return np.asarray(labels) # Return labels as np array
    


    def preprocess(self, frames, labels, config_data):

        config_preprocess = config_data.PREPROCESS
        
        #######################################
        ########## PROCESSING FRAMES ##########
        #######################################

        # RESIZE FRAMES TO BIG SIZE  (144x144 DEFAULT)
        frames = self.crop_face_resize(
                        frames,
                        config_preprocess.CROP_FACE.DO_CROP_FACE,
                        config_preprocess.CROP_FACE.USE_LARGE_FACE_BOX,
                        config_preprocess.CROP_FACE.LARGE_BOX_COEF,
                        config_preprocess.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION,
                        config_preprocess.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY,
                        config_preprocess.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX,
                        config_preprocess.BIGSMALL.RESIZE.BIG_W,
                        config_preprocess.BIGSMALL.RESIZE.BIG_H)

        # PROCESS BIG FRAMES
        big_data = list()
        for data_type in config_preprocess.BIGSMALL.BIG_DATA_TYPE:
            f_c = frames.copy()
            if data_type == "Raw":
                big_data.append(f_c)
            elif data_type == "DiffNormalized":
                big_data.append(BaseLoader.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                big_data.append(BaseLoader.standardized_data(f_c))
            else:
                raise ValueError("Unsupported data type!")
        big_data = np.concatenate(big_data, axis=-1)  # concatenate all channels

        # PROCESS SMALL FRAMES
        small_data = list()
        for data_type in config_preprocess.BIGSMALL.SMALL_DATA_TYPE:
            f_c = frames.copy()
            if data_type == "Raw":
                small_data.append(f_c)
            elif data_type == "DiffNormalized":
                small_data.append(BaseLoader.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                small_data.append(BaseLoader.standardized_data(f_c))
            else:
                raise ValueError("Unsupported data type!")
        small_data = np.concatenate(small_data, axis=-1)  # concatenate all channels

        # RESIZE SMALL FRAMES TO LOWER RESOLUTION (9x9 DEFAULT)
        small_data = self.crop_face_resize(
                        small_data,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        config_preprocess.BIGSMALL.RESIZE.SMALL_W,
                        config_preprocess.BIGSMALL.RESIZE.SMALL_H)

        ######################################
        ########## PROCESSED LABELS ##########
        ######################################

        # EXTRACT LABELS FROM ARRAY
        bp_wave = labels[:, 0]
        hr = labels[:, 1]
        bp_sys = labels[:, 2]
        bp_dia = labels [:, 3]
        bp_mean = labels [:, 4]
        resp_wave = labels[:, 5]
        rr = labels[:, 6]
        eda = labels[:, 7]
        au = labels[:, 8:47]
        pos_bvp = labels[:, 47]
        pos_env_norm_bvp = labels[:, 48]

        # REMOVE BP OUTLIERS
        bp_sys[bp_sys < 5] = 5
        bp_sys[bp_sys > 250] = 250
        bp_dia[bp_dia < 5] = 5
        bp_dia[bp_dia > 200] = 200

        # REMOVE EDA OUTLIERS
        eda[eda < 1] = 1
        eda[eda > 40] = 40

        # REMOVE AU -1 LABELS IN AU SUBSET
        if np.average(au) != -1:
            au[np.where(au != 0) and np.where(au != 1)] = 0
            labels[:, 8:47] = au

        if config_preprocess['LABEL_TYPE'] == "Raw":
            pass

        elif config_preprocess['LABEL_TYPE'] == "DiffNormalized":

            bp_wave = BaseLoader.diff_normalize_label(bp_wave)
            labels[:, 0] = bp_wave

            resp_wave = BaseLoader.diff_normalize_label(resp_wave)
            labels[:, 5] = resp_wave

            pos_bvp = BaseLoader.diff_normalize_label(pos_bvp)
            labels[:, 47] = pos_bvp

            pos_env_norm_bvp = BaseLoader.diff_normalize_label(pos_env_norm_bvp)
            labels[:, 48] = pos_env_norm_bvp

        elif config_preprocess['LABEL_TYPE'] == "Standardized":

            bp_wave = BaseLoader.standardized_label(bp_wave)
            labels[:, 0] = bp_wave

            resp_wave = BaseLoader.standardized_label(resp_wave)
            labels[:, 5] = resp_wave

            pos_bvp = BaseLoader.standardized_label(pos_bvp)
            labels[:, 47] = pos_bvp

            pos_env_norm_bvp = BaseLoader.standardized_label(pos_env_norm_bvp)
            labels[:, 48] = pos_env_norm_bvp       

        ######################################
        ######## CHUNK DATA / LABELS #########
        ######################################
        
        # Chunk clips and labels
        if config_preprocess.DO_CHUNK:
            chunk_len = config_preprocess.CHUNK_LENGTH
            big_clips, small_clips, labels_clips = self.chunk(big_data, small_data, labels, chunk_len)
        else:
            big_clips = np.array([big_data])
            small_clips = np.array([small_data])
            labels_clips = np.array([labels])

        ######################################
        ########### RETURN CHUNKS ############
        ######################################
        return big_clips, small_clips, labels_clips
    


    def chunk(self, big_frames, small_frames, labels, chunk_len):
        """Chunks the data into clips."""

        clip_num = labels.shape[0] // chunk_len
        big_clips = [big_frames[i * chunk_len:(i + 1) * chunk_len] for i in range(clip_num)]
        small_clips = [small_frames[i * chunk_len:(i + 1) * chunk_len] for i in range(clip_num)]
        labels_clips = [labels[i * chunk_len:(i + 1) * chunk_len] for i in range(clip_num)]

        return np.array(big_clips), np.array(small_clips), np.array(labels_clips)
    


    def save_multi_process(self, big_clips, small_clips, label_clips, filename, config_preprocess):
        """Saves the preprocessing data."""
        cached_path = config_preprocess.CACHED_PATH
        if not os.path.exists(cached_path):
            os.makedirs(cached_path, exist_ok=True)
        count = 0
        input_path_name_list = []
        label_path_name_list = []
        for i in range(len(label_clips)):
            assert (len(big_clips) == len(label_clips) and len(small_clips) == len(label_clips))
            
            input_path_name = cached_path + os.sep + \
                                "{0}_input{1}.pickle".format(filename, str(count))

            label_path_name = cached_path + os.sep + \
                                "{0}_label{1}.npy".format(filename, str(count))

            frames_dict = dict()
            frames_dict[0] = big_clips[i]
            frames_dict[1] = small_clips[i]

            input_path_name_list.append(input_path_name)
            label_path_name_list.append(label_path_name)

            np.save(label_path_name, label_clips[i]) # save out labels npy file
            with open(input_path_name, 'wb') as handle: # save out frame dict pickle file
                pickle.dump(frames_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            count += 1 # count of processed clips

        return count, input_path_name_list, label_path_name_list
    


    def build_file_list_retroactive(self, data_dirs, begin, end, config_data):
        """ If a file list has not already been generated for a specific data split build a list of files 
        used by the dataloader for the data split. Eg. list of files used for 
        train / val / test. Also saves the list to a .csv file.

        Args:
            data_dirs(List[str]): a list of video_files.
            begin(float): index of begining during train/val split.
            end(float): index of ending during train/val split.
        Returns:
            None (this function does save a file-list .csv file to self.file_list_path)
        """

        # get data split based on begin and end indices.
        data_dirs_subset = self.split_raw_data(data_dirs, begin, end)

        if config_data.FOLD.FOLD_NAME and config_data.FOLD.FOLD_PATH:
            data_dirs_subset = self.split_raw_data_by_fold(data_dirs_subset, config_data.FOLD.FOLD_PATH)

        # generate a list of unique raw-data file names
        filename_list = []
        for i in range(len(data_dirs_subset)):
            filename_list.append(data_dirs_subset[i]['index'])
        filename_list = list(set(filename_list))  # ensure all indexes are unique

        # generate a list of all preprocessed / chunked data files
        file_list = []
        for fname in filename_list:
            processed_file_data = list(glob.glob(self.cached_path + os.sep + "{0}_input*.pickle".format(fname)))
            file_list += processed_file_data

        if not file_list:
            raise ValueError(self.dataset_name,
                             'File list empty. Check preprocessed data folder exists and is not empty.')

        file_list_df = pd.DataFrame(file_list, columns=['input_files'])
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)
        file_list_df.to_csv(self.file_list_path)  # save file list to .csv


    def split_raw_data_by_fold(self, data_dirs, fold_path):

        fold_df = pd.read_csv(fold_path)
        fold_subjs = list(set(list(fold_df.subjects)))

        fold_data_dirs = []
        for d in data_dirs:
            idx = d['index']
            subj = idx[0:4]

            if subj in fold_subjs: # if trial has already been processed
                fold_data_dirs.append(d)

        return fold_data_dirs



    def load(self):
        """ Loads the preprocessed data listed in the file list.

        Args:
            None
        Returns:
            None
        """
        file_list_path = self.file_list_path  # get list of files in
        file_list_df = pd.read_csv(file_list_path)
        inputs = file_list_df['input_files'].tolist()
        if not inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error!')
        inputs = sorted(inputs)  # sort input file name list
        labels = [input_file.replace("input", "label").replace('.pickle', '.npy') for input_file in inputs]
        self.inputs = inputs
        self.labels = labels
        self.preprocessed_data_len = len(inputs)


    def __getitem__(self, index):
        """Returns a clip of video(3,T,W,H) and it's corresponding signals(T)."""

        with open(self.inputs[index], 'rb') as handle:
            data = pickle.load(handle)

        # format data shapes
        if self.data_format == 'NDCHW':
            data[0] = np.float32(np.transpose(data[0], (0, 3, 1, 2)))
            data[1] = np.float32(np.transpose(data[1], (0, 3, 1, 2)))
        elif self.data_format == 'NCDHW':
            data[0] = np.float32(np.transpose(data[0], (3, 0, 1, 2)))
            data[1] = np.float32(np.transpose(data[1], (3, 0, 1, 2)))
        elif self.data_format == 'NDHWC':
            pass
        else:
            raise ValueError('Unsupported Data Format!')

        label = np.load(self.labels[index])
        label = np.float32(label)
        
        item_path = self.inputs[index]
        item_path_filename = item_path.split(os.sep)[-1]
        split_idx = item_path_filename.index('_')
        filename = item_path_filename[:split_idx]
        chunk_id = item_path_filename[split_idx + 6:].split('.')[0]
        return data, label, filename, chunk_id

