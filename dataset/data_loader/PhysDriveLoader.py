"""The dataloader for PhysDrive datasets.
PhysDrive: https://github.com/WJULYW/PhysDrive-Dataset
Jiyao Wang, Xiao Yang, Qingyong Hu, Jiankai Tang, Can Liu, Dengbo He, Yuntao Wang, Ying-Cong Chen, Kaishun Wu. (2025) PhysDrive: A Multimodal Remote Physiological Measurement Dataset for In-vehicle Driver Monitoring
"""

import glob
import os
import re
import scipy.io as sio
import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
import threading
from neurokit2 import ppg_peaks, ppg_quality, NeuroKitWarning
import warnings

warnings.filterwarnings("ignore", category=NeuroKitWarning)

class PhysDriveLoader(BaseLoader):
    """The data loader for the PhysDrive dataset."""

    def __init__(self, name, data_path, config_data, device=None):
        """Initializes a PhysDrive dataloader.
            Args:
                data_path (str): Path to a folder containing raw video and BVP data.
                For example, data_path should be "On-Road-rPPG" for the following structure:
                -----------------
                     On-Road-rPPG/
                     |-- AFH1/
                         |-- A1/
                             |-- Align/
                                 |-- ...png
                             |-- Label/
                                 |-- BVP.mat
                                 |-- ECG.mat
                                 |-- RESP.mat
                                 |-- SPO2.mat
                         |-- A2/...
                         |-- B1/...
                         |-- B2/...
                         |-- C1/...
                         |-- C2/...
                     |-- AFH2/
                     ...
                     |-- CMZ2/
                -----------------
                name (str): Name of the dataloader.
                config_data (CfgNode): Data settings (ref: config.py).
        """
        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        """Returns data directories under the given path (for PhysDrive dataset)."""
        subject_dirs = glob.glob(os.path.join(data_path, "*"))
        print(f"Found {len(subject_dirs)} subject directories")
        data_dirs = []
        subject_id_map = {}

        for subject_dir in subject_dirs:
            subject_name = os.path.basename(subject_dir)  # e.g., "AFH1"
            if subject_name == "processed":
                continue  # Skip the 'processed' directory
            if subject_name not in subject_id_map:
                subject_id_map[subject_name] = len(subject_id_map) + 1
            # Retrieve all session directories (e.g., A1, A2) under the subject
            session_dirs = glob.glob(os.path.join(subject_dir, "*"))
            for session_dir in session_dirs:
                session_name = os.path.basename(session_dir)  # e.g., "A1"
                unique_id = f"{subject_name}_{session_name}"
                data_dirs.append({
                    "index": unique_id,
                    "path": session_dir,
                    "subject": subject_id_map[subject_name]
                })
        return data_dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data_dirs based on begin and end values,
        ensuring no overlapping subjects between splits."""

        if begin == 0 and end == 1:
            return data_dirs

        data_info = dict()
        for data in data_dirs:
            subject = data['subject']
            data_dir = data['path']
            index = data['index']
            if subject not in data_info:
                data_info[subject] = []
            data_info[subject].append({"index": index, "path": data_dir, "subject": subject})

        subj_list = sorted(list(data_info.keys()))
        num_subjs = len(subj_list)

        subj_range = list(range(0, num_subjs))
        if begin != 0 or end != 1:
            subj_range = list(range(int(begin * num_subjs), int(end * num_subjs)))

        data_dirs_new = []
        for i in subj_range:
            subj_num = subj_list[i]
            subj_files = data_info[subj_num]
            data_dirs_new += subj_files

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """Called by preprocess_dataset for multiprocessing."""
        try:
            session_info = data_dirs[i]
            session_path = session_info['path']
            saved_filename = session_info['index']

            # ========== Path check ==========
            align_dir = os.path.join(session_path, "Align")
            label_dir = os.path.join(session_path, "Label")

            if not os.path.isdir(align_dir):
                raise NotADirectoryError(f"Align dir missing: {align_dir}")

            # ========== Read frame data ==========
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

            # ========== Read label ==========
            if config_preprocess.USE_PSUEDO_PPG_LABEL:
                bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
            else:
                bvp_path = os.path.join(label_dir, "BVP.mat")
                if not os.path.isfile(bvp_path):
                    raise FileNotFoundError(f"BVP.mat missing: {bvp_path}")

                bvps = self.read_wave(bvp_path)
                if bvps.size == 0:
                    raise ValueError(f"Empty BVP: {bvp_path}")

            # ========== Data alignment ==========
            target_length = frames.shape[0]
            bvps = BaseLoader.resample_ppg(bvps, target_length)
            if len(bvps) != target_length:
                raise ValueError(f"Resampling failed: BVP {len(bvps)} vs frames {target_length}")

            # ========== Slice before preprocessing and saving ==========
            frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)

            # ========== Quality filtering for each clip ==========
            qualified_frames = []
            qualified_bvps = []
            skipped_count = 0

            for idx, (f_clip, b_clip) in enumerate(zip(frames_clips, bvps_clips)):
                quality = self.single_signal_quality_assessment(b_clip, fs=self.config_data.FS)
                if quality < 0.5:
                    print(
                        f"[Warning] Skipping low-quality clip {saved_filename}/{idx + 1}/{len(bvps_clips)}: quality={quality:.3f}")
                    skipped_count += 1
                    continue
                qualified_frames.append(f_clip)
                qualified_bvps.append(b_clip)

            # Check if there are any valid clips
            if len(qualified_frames) == 0:
                print(f"[Warning] All clips in {saved_filename} are low quality. Skipping whole session.")
                return
            else:
                print(f"{skipped_count}/{len(frames_clips)} clips skipped.")

            # ========== Save filtered clips ==========
            input_name_list, label_name_list = self.save_multi_process(qualified_frames, qualified_bvps, saved_filename)

            # Thread-safe update of shared dictionary
            with threading.Lock():
                file_list_dict[i] = input_name_list

        except Exception as e:
            print(f"[Error] Failed to process {session_path}: {str(e)}")

    @staticmethod
    def read_video(video_file):
        """Reads a video file (actually a directory of aligned frames), returns frames (T, H, W, 3)"""
        png_files = sorted(glob.glob(os.path.join(video_file, "*.png")))
        frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in png_files]
        return np.array(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a BVP signal file."""
        waves = sio.loadmat(bvp_file)["BVP"].flatten()
        return np.asarray(waves)

    @staticmethod
    def single_signal_quality_assessment(signal, fs=30, method_quality='templatematch', method_peaks='elgendi'):
        assert method_quality in ['templatematch', 'dissimilarity'], "method_quality must be one of ['templatematch', 'dissimilarity']"

        signal_filtered = signal

        if len(signal_filtered) < 10 or np.all(signal_filtered == signal_filtered[0]):
            print("Warning: Signal is too short or constant. Skipping quality assessment.")
            return 0

        if method_quality in ['templatematch', 'dissimilarity']:
            method_quality = 'dissimilarity' if method_quality == 'dissimilarity' else method_quality

            try:
                _, peak_info = ppg_peaks(
                    signal_filtered,
                    sampling_rate=fs,
                    method=method_peaks
                )

                if peak_info["PPG_Peaks"].size == 0:
                    print("No peaks detected in the signal. Skipping quality assessment.")
                    return 0

                quality = ppg_quality(
                    signal_filtered,
                    ppg_pw_peaks=peak_info["PPG_Peaks"],
                    sampling_rate=fs,
                    method=method_quality
                )

                quality = np.nanmean(quality)

            except ValueError as e:
                print(f"Error in ppg_quality function: {e}")
                quality = 0

            return quality
