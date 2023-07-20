# Dataset to MP4 Videos Conversion Script
# 
# This is a simple script that serves as an example to convert 
# an rPPG dataset of interest into a folder of MP4 video files 
# that can ultimately be used with OpenFace for further analysis.
#
# See comments and the motion_analysis folder README For more details.

import os, glob
import cv2
import numpy as np
from scipy import io as scio

# Functions for reading rPPG media of interest and saving frames
def read_video(video_file):
    """Reads a video file, returns frames(T, H, W, 3) """
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

def read_png_frames(video_file):
    """Reads a video file, returns frames(T, H, W, 3) """
    frames = list()
    all_png = sorted(glob.glob(video_file + '*.png'))
    for png_path in all_png:
        img = cv2.imread(png_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    return np.asarray(frames)

def read_mat(mat_file):
    """Reads a video file in the MATLAB format (.mat), returns frames(T,H,W,3)"""
    try:
        mat = scio.loadmat(mat_file)
    except:
        for _ in range(20):
            print(mat_file)
    frames = np.array(mat['video'])
    return frames

def read_npy_video(self, video_file):
    """Reads a video file in the numpy format (.npy), returns frames(T,H,W,3)"""
    frames = np.load(video_file[0])
    if np.issubdtype(frames.dtype, np.integer) and np.min(frames) >= 0 and np.max(frames) <= 255:
        processed_frames = [frame.astype(np.uint8)[..., :3] for frame in frames]
    elif np.issubdtype(frames.dtype, np.floating) and np.min(frames) >= 0.0 and np.max(frames) <= 1.0:
        processed_frames = [(np.round(frame * 255)).astype(np.uint8)[..., :3] for frame in frames]
    else:
        print("Failed!")
    return np.asarray(processed_frames)

def save_video_frames(frames, video_name, save_path):
    """Saves video frames as an mp4 video file in the save path"""
    os.makedirs(save_path, exist_ok=True)
    height, width, _ = frames[0].shape
    video_name = video_name + ".mp4"
    video_file = os.path.join(save_path, video_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_file, fourcc, 30.0, (width, height))
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()
    print(f"Video saved: {video_file}")

# Dataset specific processing functions
def process_ubfc_folder(folder_path, save_path):
    data_dirs = glob.glob(folder_path + os.sep + "subject*")
    for dir in data_dirs:
        frames = read_video(os.path.join(dir,"vid.avi"))
        subject_name = os.path.split(dir)[-1]
        save_video_frames(frames, subject_name, save_path)
    print("All videos saved!")

def process_phys_folder(folder_path, save_path):
    data_dirs = glob.glob(folder_path + os.sep + "s*" + os.sep + "*.avi")
    for dir in data_dirs:
        frames = read_video(dir)
        subject_name = os.path.split(dir)[-1]
        subject_name = os.path.splitext(subject_name)[0]
        save_video_frames(frames, subject_name, save_path)
    print("All videos saved!")

def process_pure_folder(folder_path, save_path):
    data_dirs = glob.glob(folder_path + os.sep + "*-*")
    for dir in data_dirs:
        subject_trail_val = os.path.split(dir)[-1]
        video_name = subject_trail_val
        frames = read_png_frames(os.path.join(dir, "", subject_trail_val, ""))
        save_video_frames(frames, video_name, save_path)
    print("All videos saved!")

def process_afrl_folder(folder_path, save_path):
    data_dirs = glob.glob(folder_path + os.sep + "*.avi")
    for dir in data_dirs:
        frames = read_video(dir)
        subject_name = os.path.split(dir)[-1]
        subject_name = os.path.splitext(subject_name)[0]
        save_video_frames(frames, subject_name, save_path)
    print("All videos saved!")

def process_mmpd_folder(folder_path, save_path):
    data_dirs = glob.glob(folder_path + os.sep + 'subject*')
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
    for dir in dirs:
        frames = read_mat(dir['path'])
        frames = (np.round(frames * 255)).astype(np.uint8)
        subject_name = os.path.split(dir['path'])[-1]
        subject_name = subject_name.split('.')[0]
        save_video_frames(frames, subject_name, save_path)
    print("All videos saved!")


if __name__ == "__main__":
    # The below code is an example of using the above functions to convert the UBFC-rPPG dataset
    # into a folder of MP4s for subsequent analysis by OpenFace.

    # Dataset Paths
    # Change this to point to the location of your dataset
    dataset_path = '/path/to/UBFC-rPPG'

    # Save Paths
    # Change this to point to the location you'd like to save the MP4 videos
    save_path = '/path/to/converted_mp4_videos'

    # Dataset Processing
    process_ubfc_folder(dataset_path, os.path.join(save_path))
