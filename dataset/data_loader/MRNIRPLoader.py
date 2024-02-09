import glob
import os
import re

from tqdm import tqdm
import numpy as np

import zipfile

import cv2
import io
import imageio

from scipy.io import loadmat

from dataset.data_loader.BaseLoader import BaseLoader

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
                     |       |-- NIR.zip
                     |       |-- RGB.zip
                     |       |-- PulseOX.zip
                     |   |-- subject2/
                     |       |-- NIR.zip
                     |       |-- RGB.zip
                     |       |-- PulseOX.zip
                     |...
                     |   |-- subjectn/
                     |       |-- NIR.zip
                     |       |-- RGB.zip
                     |       |-- PulseOX.zip
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)


    def get_raw_data(self, data_path):
        """Returns data directories under the path(For UBFC-rPPG dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "subject*" + os.sep + "*_garage_still_975")

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
                    
                    frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
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
            ppg = mat['pulseOxRecord']

        return np.asarray(ppg).flatten()
    
    
    @staticmethod
    def read_video_unzipped(video_file):
        frames = list()
        all_pgm = sorted(glob.glob(os.path.join(video_file, "Frame*.pgm")))
        for pgm_path in all_pgm:
            frame = cv2.imread(pgm_path, cv2.IMREAD_UNCHANGED)          # read 10bit raw image
            frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_RG2RGB)         # Demosaice RGB Image
            frame = cv2.convertScaleAbs(frame, alpha=(255.0/65535.0))   # convert from uint16 to uint8

            frames.append(frame)
            
        return np.asarray(frames, dtype=np.uint8)

    
    @staticmethod
    def read_wave_unzipped(wave_file):
        """Reads a bvp signal file."""
        mat = loadmat(wave_file + os.sep + "pulseOx.mat")
        return np.squeeze(np.asarray(mat['pulseOxRecord']))
    

    def preprocess_dataset(self, data_dirs, config_preprocess, begin=0, end=1):
        """Preprocesses the raw data."""
        file_num = len(data_dirs)
        for i in tqdm(range(file_num)):
            # Read Video Frames
            # frames = self.read_video(os.path.join(data_dirs[i]['path'], "RGB.zip"))
            frames = self.read_video_unzipped(os.path.join(data_dirs[i]['path'], "RGB"))

            # bvps = self.read_wave(os.path.join(data_dirs[i]['path'], "PulseOx.zip"))
            bvps = self.read_wave_unzipped(os.path.join(data_dirs[i]['path'], "PulseOX"))
            
            target_length = frames.shape[0]
            bvps = BaseLoader.resample_ppg(bvps, target_length)
            frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
            self.preprocessed_data_len += self.save(frames_clips, bvps_clips, data_dirs[i]["index"])

        

    # def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
    #     """ invoked by preprocess_dataset for multi_process."""
    #     saved_filename = data_dirs[i]['index']
        
    #     frames = self.read_video(os.path.join(data_dirs[i]['path'], "RGB"))

    #     # Read Labels
    #     if config_preprocess.USE_PSUEDO_PPG_LABEL:
    #         bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
    #     else:
    #         bvps = self.read_wave(os.path.join(data_dirs[i]['path'], "PulseOX", "PulseOx.mat"))
        
    #     target_length = frames.shape[0]
    #     bvps = BaseLoader.resample_ppg(bvps, target_length)
        
    #     frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
    #     input_name_list, _ = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
    #     file_list_dict[i] = input_name_list