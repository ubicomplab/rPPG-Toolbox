"""The dataloader for  PhysDrive datasets.

"""
import glob
import glob
import json
import os
import re
import scipy.io as sio
import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
import threading


class  PhysDriveLoader(BaseLoader):
    """The data loader for the  PhysDrive dataset."""

    def __init__(self, name, data_path, config_data, device=None):
        """Initializes an  PhysDrive dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "On-Road-rPPG" for below dataset structure:
                -----------------
                     On-Road-rPPG/
                     |   |-- AFH1/
                     |      |-- A1/
                     |          |-- Align/
                     |              |-- ...png
                     |          |-- Label/
                     |              |-- BVP.mat
                     |              |-- ECG.mat
                     |              |-- RESP.mat
                     |              |-- SPO2.mat
                     |      |-- A2/...
                     |      |-- B1/...
                     |      |-- B2/...
                     |      |-- C1/...
                     |      |-- C2/...
                     |   |-- AFH2/
                     ...
                     |   |-- CMZ2/

                -----------------
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For  PhysDrive dataset)."""
        subject_dirs = glob.glob(os.path.join(data_path, "*"))
        print(f"Found {len(subject_dirs)} subject directories")
        data_dirs = []
        subject_id_map = {}

        for subject_dir in subject_dirs:
            subject_name = os.path.basename(subject_dir)  # 如 "AFH1"
            if subject_name == "processed":
                continue  # 跳过 processed 目录
            if subject_name not in subject_id_map:
                subject_id_map[subject_name] = len(subject_id_map) + 1
            # 获取Subject下的所有Session目录（A1, A2等）
            session_dirs = glob.glob(os.path.join(subject_dir, "*"))
            for session_dir in session_dirs:
                session_name = os.path.basename(session_dir)  # 如 "A1"
                unique_id = f"{subject_name}_{session_name}"  # 唯一标识符
                data_dirs.append({
                    "index": unique_id,  # 或使用哈希生成整数
                    "path": session_dir,
                    "subject": subject_id_map[subject_name]  # 数字Subject ID
                })
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

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ Invoked by preprocess_dataset for multi_process. """
        try:
            session_info = data_dirs[i]
            session_path = session_info['path']
            saved_filename = session_info['index']

            # ========== 路径检查 ==========
            align_dir = os.path.join(session_path, "Align")
            label_dir = os.path.join(session_path, "Label")
            # print("session info:", session_info, "session path:", session_path, "align_dir:", align_dir, "label_dir:", label_dir)

            if not os.path.isdir(align_dir):
                raise NotADirectoryError(f"Align dir missing: {align_dir}")

            # ========== 读取帧数据 ==========
            if 'None' in config_preprocess.DATA_AUG:
                frames = self.read_video(align_dir)
            elif 'Motion' in config_preprocess.DATA_AUG:
                npy_files = glob.glob(os.path.join(session_path, '*.npy'))
                if not npy_files:
                    raise FileNotFoundError(f"No .npy files in {session_path}")
                frames = self.read_npy_video(npy_files)
            else:
                raise ValueError(f'Unsupported DATA_AUG: {config_preprocess.DATA_AUG}')

            if frames.size == 0:
                raise ValueError(f"Empty frames: {align_dir}")

            # ========== 读取标签 ==========
            if config_preprocess.USE_PSUEDO_PPG_LABEL:
                bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
            else:
                bvp_path = os.path.join(label_dir, "BVP.mat")
                if not os.path.isfile(bvp_path):
                    raise FileNotFoundError(f"BVP.mat missing: {bvp_path}")

                bvps = self.read_wave(bvp_path)
                if bvps.size == 0:
                    raise ValueError(f"Empty BVP: {bvp_path}")

            # ========== 数据对齐 ==========
            target_length = frames.shape[0]
            bvps = BaseLoader.resample_ppg(bvps, target_length)
            if len(bvps) != target_length:
                raise ValueError(f"Resampling failed: BVP {len(bvps)} vs frames {target_length}")

            # ========== 预处理与保存 ==========
            frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
            input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)

            # 使用线程安全的方式更新共享字典
            with threading.Lock():  # 如果使用多线程
                file_list_dict[i] = input_name_list

        except Exception as e:
            print(f"[Error] Failed to process {session_path}: {str(e)}")
            # 可选：将错误信息记录到日志文件

    @staticmethod
    def read_video(video_file):
        """Reads a video file (actually a directory of aligned frames), returns frames(T, H, W, 3)"""
        png_files = sorted(glob.glob(os.path.join(video_file, "*.png")))  # 正确使用 video_file 作为路径
        frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in png_files]
        return np.array(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        waves = sio.loadmat(bvp_file)["BVP"].flatten() #(1,len)
        return np.asarray(waves)
