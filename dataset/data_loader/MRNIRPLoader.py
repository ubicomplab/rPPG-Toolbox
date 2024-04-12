import glob
import os

from tqdm import tqdm
import numpy as np

import zipfile

import cv2

import io
import imageio

from scipy.io import loadmat

from dataset.data_loader.BaseLoader import BaseLoader

import pandas as pd


class MRNIRPLoader(BaseLoader):
    """The data loader for the MR-NIRP Processed dataset."""
    def __init__(self, name, data_path, config_data):
        """Initializes an MR-NIRP dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                    RawData/
                    |   |-- subject1/
                    |       |-- subject1_driving_large_motion_975
                    |          |-- NIR.zip
                    |          |-- RGB.zip
                    |          |-- PulseOX.zip
                    |       |-- subject1_driving_small_motion_975
                    |          |...
                    |       |-- subject1_driving_small_motion_940
                    |          |...
                    |       |-- subject1_driving_still_940
                    |          |...
                    |       |-- subject1_driving_still_975
                    |          |...
                    |       |-- subject1_garage_large_motion_975
                    |          |...
                    |       |-- subject1_garage_large_motion_940
                    |          |...
                    |       |-- subject1_garage_small_motion_975
                    |          |...
                    |       |-- subject1_garage_small_motion_940
                    |          |...
                    |       |-- subject1_garage_still_974
                    |          |...
                    |       |-- subject1_garage_still_940
                    |          |...
                    |   |-- subject2/
                    |       |...
                    |...
                    |   |-- subjectn/
                    |       |...
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        self.filtering = config_data.FILTERING
        super().__init__(name, data_path, config_data)


    def get_raw_data(self, data_path):
        """Returns data directories under the path(For MR-NIRP dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "Subject*" + os.sep + "subject*")

        if not data_dirs:
            raise ValueError("dataset data paths empty!")
        dirs = [{"index": os.path.basename(data_dir), "path": data_dir} for data_dir in data_dirs]
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
    

    def load_preprocessed_data(self):
        """ Loads the preprocessed data listed in the file list.
        """
        file_list_path = self.file_list_path  # get list of files in
        file_list_df = pd.read_csv(file_list_path)
        base_inputs = file_list_df['input_files'].tolist()
        filtered_inputs = []

        for input in base_inputs:
            input_name = input.split(os.sep)[-1].split('.')[0].rsplit('_', 1)[0]
            subject_name = input_name.rsplit('_')[0]
            task = input_name.rsplit('_', 1)[0].split('_', 1)[1]
            subject_task = input_name.rsplit('_', 1)[0]

            if self.filtering.SELECT_TASKS:
                if input_name not in self.filtering.TASK_LIST and subject_name not in self.filtering.TASK_LIST and task not in self.filtering.TASK_LIST and subject_task not in self.filtering.TASK_LIST:
                    continue
                
            if self.filtering.USE_EXCLUSION_LIST:
                if input_name in self.filtering.EXCLUSION_LIST or subject_name in self.filtering.EXCLUSION_LIST or task in self.filtering.EXCLUSION_LIST or subject_task in self.filtering.EXCLUSION_LIST:
                    continue

            # print(input_name)
            filtered_inputs.append(input)

        if not filtered_inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error!')
        
        filtered_inputs = sorted(filtered_inputs)  # sort input file name list
        labels = [input_file.replace("input", "label") for input_file in filtered_inputs]
        self.inputs = filtered_inputs
        self.labels = labels
        self.preprocessed_data_len = len(filtered_inputs)


    @staticmethod
    def read_video(video_file, resize_dim=144):
        """Reads a video file, returns frames(T, H, W, 3) """
        cnt = 0
        frames=list()
        with zipfile.ZipFile(video_file, "r") as zippedImgs:
            for ele in zippedImgs.namelist():
                ext = os.path.splitext(ele)[-1]

                if ext == '.pgm':
                    data = zippedImgs.read(ele)
                    image_file = io.BytesIO(data)
                    frame = np.array(imageio.imread(image_file), dtype=np.uint16)
                    
                    frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2RGB)         # Demosaice rggb to RGB Image
                    frame = (frame >> 8).astype(np.uint8)                       # convert from uint16 to uint8
                    
                    # downsample frames (otherwise processing time becomes WAY TOO LONG)
                    if resize_dim is not None:
                        dim_w = min(resize_dim, frame.shape[1])
                        dim_h = int(dim_w * frame.shape[0] / frame.shape[1])
                        frame = cv2.resize(frame, (dim_w, dim_h), interpolation=cv2.INTER_AREA)
                        
                    frame = np.expand_dims(frame, axis=0)

                    if cnt == 0:
                        frames = frame
                    else:
                        frames = np.concatenate((frames, frame), axis=0)
                    cnt += 1
                    
        if cnt == 0:
            raise ValueError('EMPTY VIDEO', video_file)
        
        return np.asarray(frames)


    @staticmethod
    def read_wave(wave_file):
        """Reads a bvp signal file."""
        with zipfile.ZipFile(wave_file, 'r') as wave_archive:
            mat = loadmat(wave_archive.open('PulseOX/pulseOx.mat'))
            ppg = mat['pulseOxRecord'][0]
            timestamps = (mat['pulseOxTime'][0] - mat['pulseOxTime'][0][0])

        return ppg, timestamps
    
    
    @staticmethod
    def read_video_unzipped(video_file):
        frames = list()
        all_pgm = sorted(glob.glob(os.path.join(video_file, "Frame*.pgm")))
        for pgm_path in all_pgm:
            try:
                frame = cv2.imread(pgm_path, cv2.IMREAD_UNCHANGED)          # read 10bit raw image (in uint16 format)
                frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2RGB)         # Demosaice rggb to RGB Image
            except:
                print("Error in reading frame:", pgm_path)
                continue
            
            frame = (frame >> 8).astype(np.uint8)                       # convert from uint16 to uint8

            frames.append(frame)
            
        return np.asarray(frames, dtype=np.uint8)


    @staticmethod
    def read_wave_unzipped(wave_file):
        """Reads a bvp signal file."""
        raw_data = loadmat(wave_file + os.sep + "pulseOx.mat")
        
        timestamps = (raw_data['pulseOxTime'][0] - raw_data['pulseOxTime'][0][0])
        ppg = raw_data['pulseOxRecord'][0]
        
        return ppg, timestamps
    
    @staticmethod
    def correct_irregular_sampling(ppg, timestamps, target_fs=30):
        """Resampling functionality borrowed from: https://github.com/ToyotaResearchInstitute/RemotePPG"""
        resampled_ppg = []
        for curr_time in np.arange(0.0, timestamps[-1], 1.0/target_fs):
            time_diff = timestamps - curr_time
            stop_idx = np.argmax(time_diff > 0)
            start_idx = stop_idx - 1 if stop_idx > 0 else stop_idx
            
            time_span = time_diff[stop_idx] - time_diff[start_idx]
            weight = - time_diff[start_idx] / time_span if time_span != 0 else 0
            
            interpolated_ppg = ppg[start_idx] * (1 - weight) + ppg[stop_idx] * weight
            resampled_ppg.append(interpolated_ppg)
        
        return np.array(resampled_ppg)
    
    
    @staticmethod
    def match_length(ppg, frames):
        target_length = min(ppg.shape[0], frames.shape[0])
        ppg = ppg[:target_length]
        frames = frames[:target_length]
        return ppg, frames
    
    
    # def preprocess_dataset(self, data_dirs, config_preprocess, begin=0, end=1):
    #     """Preprocesses the raw data."""
    #     file_num = len(data_dirs)
                
    #     for i in tqdm(range(file_num)):
    #         # Skip the subject2_garage_small_motion_940 corrupted video
    #         if data_dirs[i]['index'] == "subject2_garage_small_motion_940":
    #             continue
            
    #         # Read Video Frames
    #         # frames = self.read_video(os.path.join(data_dirs[i]['path'], "RGB.zip"))
    #         frames = self.read_video_unzipped(os.path.join(data_dirs[i]['path'], "RGB"))

    #         if self.config_data.PREPROCESS.USE_PSUEDO_PPG_LABEL:
    #             bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
    #         else: 
    #             # bvps = self.read_wave(os.path.join(data_dirs[i]['path'], "PulseOx.zip"))
    #             bvps, timestamps = self.read_wave_unzipped(os.path.join(data_dirs[i]['path'], "PulseOX"))
                        
    #         bvps = self.correct_irregular_sampling(bvps, timestamps, target_fs=self.config_data.FS)
    #         bvps, frames = self.match_length(bvps, frames)
                        
    #         # target_length = frames.shape[0]
    #         # bvps = BaseLoader.resample_ppg(bvps, target_length)

    #         frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
    #         self.preprocessed_data_len += self.save(frames_clips, bvps_clips, data_dirs[i]["index"])


    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ invoked by preprocess_dataset for multi_process."""        
        # Skip corrupted frames
        if data_dirs[i]['index'] == "subject2_garage_small_motion_940":
            return
        # frames = self.read_video(os.path.join(data_dirs[i]['path'], "RGB.zip"))
        frames = self.read_video_unzipped(os.path.join(data_dirs[i]['path'], "RGB"))

        if self.config_data.PREPROCESS.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else: 
            # bvps = self.read_wave(os.path.join(data_dirs[i]['path'], "PulseOx.zip"))
            bvps, timestamps = self.read_wave_unzipped(os.path.join(data_dirs[i]['path'], "PulseOX"))
                    
        bvps = self.correct_irregular_sampling(bvps, timestamps, target_fs=self.config_data.FS)
        bvps, frames = self.match_length(bvps, frames)
                    
        # target_length = frames.shape[0]
        # bvps = BaseLoader.resample_ppg(bvps, target_length)

        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, _ = self.save_multi_process(frames_clips, bvps_clips, data_dirs[i]['index'])
        file_list_dict[i] = input_name_list
