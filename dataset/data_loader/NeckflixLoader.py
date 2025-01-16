"""The dataloader for the Neckflix dataset.

Details for the Neckflix Dataset see ####
If you use this dataset, please cite this paper:
C. Arrow, M. Ward, J. Eshraghian, G. Dwivedi.
"Neckflix:"
"""
import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager
import subprocess
import numpy as np

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
import csv
import pandas as pd
import av

class NeckflixLoader(BaseLoader):
    """The data loader for the Neckflix dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an Neckflix dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- P001_S01_R1_0_D/
                     |       |-- video_start_end_times.csv
                     |       |-- K1_Depth.mkv
                     |       |-- K1_IR.mkv
                     |       |-- K1_RGB.mkv
                     |       |-- K2_Depth.mkv
                     |       |-- K2_IR.mkv
                     |       |-- K2_RGB.mkv
                     |       |...
                     |       |-- trace_data.csv
                     |   |-- P001_S01_R2_0_N/
                     |       |-- video_start_end_times.csv
                     |       |-- K1_RGB.mkv
                     |       |-- K2_RGB.mkv
                     |       |...
                     |       |-- trace_data.csv
                     |...
                     |   |-- Px_Sy_Rz/
                     |       |-- video_start_end_times.csv
                     |       |-- K1_Depth.mkv
                     |       |-- K1_IR.mkv
                     |       |-- K1_RGB.mkv
                     |       |-- K2_Depth.mkv
                     |       |-- K2_IR.mkv
                     |       |-- K2_RGB.mkv
                     |       |...
                     |       |-- trace_data.csv
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        self.filtering = config_data.FILTERING
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For iBVP dataset)."""
        data_dirs = sorted(glob.glob(os.path.join(data_path, "*", "*.mkv")) + glob.glob(os.path.join(data_path, "*", "*.hdf5")))
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = list()
        for data_dir in data_dirs:
            index = os.sep.join(data_dir.split(os.sep)[-2:]).split('.')[0]
            participant = data_dir.split(os.sep)[-2].split('_')[0]
            pos = data_dir.split(os.sep)[-2].split('_')[-2]
            if pos == '0':
                position = 'supine'
            elif pos == '45':
                position = 'recumbent'
            elif pos == '90':
                position = 'sitting'
            else:
                raise ValueError(f"Position {pos} not recognized!")
            dirs.append({"index": index, "path": data_dir, "participant": participant, "position": position})
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping participants between splits"""

        # return the full directory
        if begin == 0 and end == 1:
            return data_dirs

        # get info about the dataset: participant list and num vids per participant
        data_info = dict()
        for data in data_dirs:
            participant = data['participant']
            data_dir = data['path']
            index = data['index']
            position = data['position']
            # creates a dictionary of data_dirs indexed by participant number
            if participant not in data_info:  # if participant not in the data info dictionary
                data_info[participant] = []  # make an emplty list for that participant
            # append a tuple of the filename, participant num, trial num, and chunk num
            data_info[participant].append({"index": index, "path": data_dir, "participant": participant, "position": position})

        participant_list = sorted(list(data_info.keys()))  # all participants by number ID
        num_participants = len(participant_list)  # number of unique participants

        # get split of data set (depending on start / end)
        participant_range = list(range(0, num_participants))
        if begin != 0 or end != 1:
            participant_range = list(range(int(begin * num_participants), int(end * num_participants)))

        # compile file list
        data_dirs_new = []
        for i in participant_range:
            participant_num = participant_list[i]
            participant_files = data_info[participant_num]
            data_dirs_new += participant_files  # add file information to file_list (tuple of fname, participant ID, trial num, chunk num)

        return data_dirs_new

    # def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
    #     """ Invoked by preprocess_dataset for multi_process. """
    #     filename = os.path.split(data_dirs[i]['path'])[-1]
    #     saved_filename = data_dirs[i]['index']

    #     # Read Frames
    #     if 'None' in config_preprocess.DATA_AUG:
    #         # Utilize dataset-specific function to read video
            
    #         if config_preprocess.IBVP.DATA_MODE == "T":
    #             frames = self.read_video(
    #                 os.path.join(data_dirs[i]['path'], "{0}_t".format(filename), ""))
            
    #         elif config_preprocess.IBVP.DATA_MODE == "RGBT":
    #             rgb_frames = self.read_video(
    #                 os.path.join(data_dirs[i]['path'], "{0}_rgb".format(filename), ""))

    #             thermal_frames = self.read_thermal_video(
    #                 os.path.join(data_dirs[i]['path'], "{0}_t".format(filename), ""))
    #         else:
    #             frames = self.read_video(
    #                 os.path.join(data_dirs[i]['path'], "{0}_rgb".format(filename), ""))

    #     elif 'Motion' in config_preprocess.DATA_AUG:
    #         # Utilize general function to read video in .npy format
    #         frames = self.read_npy_video(
    #             glob.glob(os.path.join(data_dirs[i]['path'], filename, '*.npy')))
    #     else:
    #         raise ValueError(f'Unsupported DATA_AUG specified for {self.dataset_name} dataset! Received {config_preprocess.DATA_AUG}.')

    #     # Read Labels
    #     if config_preprocess.USE_PSUEDO_PPG_LABEL:
    #         bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
    #     else:
    #         bvps, sq_vec = self.read_wave(
    #             os.path.join(data_dirs[i]['path'], "{0}_bvp.csv".format(filename)))

    #     if "RGBT" in config_preprocess.IBVP.DATA_MODE:
    #         rgb_length, rgb_height, rgb_width, rgb_ch = rgb_frames.shape
    #         thermal_length, t_height, t_width, t_ch = thermal_frames.shape
    #         target_length = min(rgb_length, thermal_length)
    #         rgb_frames = rgb_frames[:target_length, ...]
    #         thermal_frames = thermal_frames[:target_length, :rgb_height, :, :]      #rgb_height = 480, thermal height = 512, so reducing thermal height to match with RGB.
    #         frames = np.concatenate([rgb_frames, thermal_frames], axis=-1)
    #     else:
    #         target_length = frames.shape[0]

    #     bvps = BaseLoader.resample_ppg(bvps, target_length)
    #     sq_vec = BaseLoader.resample_ppg(sq_vec, target_length)

    #     # print(type(frames), frames.shape)
    #     # print(type(bvps), bvps.shape)
    #     # exit()
    #     # Discard frames based on Signal Quality
    #     del_idx = sq_vec <= 0.3
    #     frames = np.delete(frames, del_idx, axis=0)
    #     bvps = np.delete(bvps, del_idx, axis=0)
    #     sq_vec = np.delete(sq_vec, del_idx, axis=0)

    #     frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
    #     input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
    #     file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file, data_dict):
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
        VidObj.release()
        data_dict['RGB'] = np.asarray(frames)
        return data_dict

    @staticmethod
    def read_16bit_video(video_file, data_dict):
        """Reads a 16-bit grayscale FFV1 MKV file, returns frames (T, H, W, 1) as uint16"""
        container = av.open(video_file)
        video_stream = container.streams.video[0]
        frames = []

        for frame in container.decode(video_stream):
            # Convert to a NumPy array in 16-bit grayscale
            frame_np = frame.to_ndarray(format="gray16le")
            # PyAV returns grayscale frames with shape (H, W)
            frames.append(frame_np)

        # Return all frames as a single NumPy array of shape (T, H, W)
        frames = np.asarray(frames, dtype=np.uint16)
        if 'IR' in video_file:
            data_dict['IR'] = np.expand_dims(frames,axis=-1)
        elif 'Depth' in video_file:
            data_dict['Depth'] = np.expand_dims(frames,axis=-1)
        return data_dict

    # @staticmethod
    # def read_wave(bvp_file):
    #     """Reads a bvp signal file."""
    #     with open(bvp_file, "r") as f:
    #         labels = pd.read_csv(f).to_numpy()
    #         waves = labels[:, 0]
    #         sq_vec = labels[:, 3]   #SQ2
    #     return waves, sq_vec


# Variables
data_path = "/Volumes/SMED-VEFCVP-001/CVP_Dataset/Processed_Dataset"
begin = 0
end = 0.8

### Testing functions
def get_raw_data(data_path):
    """Returns data directories under the path(For iBVP dataset)."""
    data_dirs = sorted(glob.glob(os.path.join(data_path, "*", "*.mkv")) + glob.glob(os.path.join(data_path, "*", "*.hdf5")))
    # if not data_dirs:
    #     raise ValueError(self.dataset_name + " data paths empty!")
    dirs = list()
    for data_dir in tqdm(data_dirs):
        index = os.sep.join(data_dir.split(os.sep)[-2:]).split('.')[0]
        video_type = index.split(os.sep)[-1].split(os.sep)[-1]
        participant = data_dir.split(os.sep)[-2].split('_')[0]
        pos = data_dir.split(os.sep)[-2].split('_')[-2]
        if pos == '0':
            position = 'supine'
        elif pos == '45':
            position = 'recumbent'
        elif pos == '90':
            position = 'sitting'
        else:
            raise ValueError(f"Position {pos} not recognized!")
        data_dir_info = {"index": index,
                         "video_type": video_type, 
                         "path": data_dir, 
                         "participant": participant, 
                         "position": position}
        dirs.append(data_dir_info)
    return dirs

def split_raw_data(data_dirs, begin, end):
    """Returns a subset of data dirs, split with begin and end values, 
    and ensures no overlapping participants between splits"""

    # return the full directory
    if begin == 0 and end == 1:
        return data_dirs

    # get info about the dataset: participant list and num vids per participant
    data_info = dict()
    for data in data_dirs:
        participant = data['participant']
        video_type = data['video_type']
        data_dir = data['path']
        index = data['index']
        position = data['position']
        # creates a dictionary of data_dirs indexed by participant number
        if participant not in data_info:  # if participant not in the data info dictionary
            data_info[participant] = []  # make an emplty list for that participant
        # append a tuple of the filename, participant num, trial num, and chunk num
        data_info[participant].append({"index": index,
                                       "video_type": video_type, 
                                       "path": data_dir, 
                                       "participant": participant, 
                                       "position": position})

    participant_list = sorted(list(data_info.keys()))  # all participants by number ID
    num_participants = len(participant_list)  # number of unique participants

    # get split of data set (depending on start / end)
    participant_range = list(range(0, num_participants))
    if begin != 0 or end != 1:
        participant_range = list(range(int(begin * num_participants), int(end * num_participants)))

    # compile file list
    data_dirs_new = []
    for i in participant_range:
        participant_num = participant_list[i]
        participant_files = data_info[participant_num]
        data_dirs_new += participant_files  # add file information to file_list (tuple of fname, participant ID, trial num, chunk num)

    return data_dirs_new

data_dirs = get_raw_data(data_path)
data_dirs_subset = split_raw_data(data_dirs, begin, end)
data_dir_info = data_dirs_subset[1] # 0 for EV, 1 for Depth, 2 for IR, 3 for RGB
#### Defining the construct_data_dict function
# Get the instance for the specific video recording
instance = data_dir_info['index'].split(os.sep)[0]
# define an empty dict
data_dict = dict()

if 'RGB' in data_dir_info['video_type']:
    data_dict = read_video(data_dir_info['path'], data_dict)
    print('Loaded RGB Video')
elif ('IR' in data_dir_info['video_type']) | ('Depth' in data_dir_info['video_type']):
    frames = read_16bit_video(data_dir_info['path'])
    print('Loaded IR/Depth Video')
elif 'EV' in video_type:
    print('EV Video not yet supported')
else:
    raise ValueError(f"Video type {video_type} not recognized!")

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


# read depth video
ev_file = data_dirs[0]['path']
depth_file = data_dirs[1]['path']
ir_file = data_dirs[2]['path']
rgb_file = data_dirs[3]['path']


#TODO: Read ev video