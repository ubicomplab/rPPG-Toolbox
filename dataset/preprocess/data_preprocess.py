import cv2
import numpy as np
from dataset.synchronize import green_channel, align

W = 128
H = 128


def facial_detection(frame):
    '''facial detection function for one face'''
    detector = cv2.CascadeClassifier(
        '/home/dlenv/xiaoyu/Toolbox/dataset/haarcascade_frontalface_default.xml')
    face_zone = detector.detectMultiScale(frame)
    if(len(face_zone) < 1):
        print("ERROR:No Face Detected")
    if(len(face_zone) >= 2):
        print("WARN:More than one faces are detected(Only cropping one face)")
    x, y, w, h = face_zone[0]
    return x, y, w, h


def crop_face_down_sample(frame, x, y, w, h):
    '''crop one face and resize the frame to 128*128'''
    frame = frame[y:y + h, x:x + w]
    frame = cv2.resize(frame, (W, H))
    return frame


def down_sample(a, len):
    '''down sampling'''
    return np.interp(np.linspace(1, a.shape[0], len), np.linspace(1, a.shape[0], a.shape[0]), a)


def resize(frames, H, W, face_region):
    '''Downsample every frame into 128x128 and normalize the image'''
    resize_frames = np.zeros((frames.shape[0], H, W, 3))
    for i in range(0, frames.shape[0]):
        frame = frames[i]
        frame = frame[face_region[1]:face_region[1]+face_region[3],
                      face_region[0]:face_region[0]+face_region[2]]
        resize_frames[i] = cv2.resize(frame, (W, H))
    return resize_frames


def synchronize(frames, bvps, plot_flag):
    '''synchronize the video frames and the bvps'''
    face_region = facial_detection(frames[0])
    greenchannel, bvp = green_channel.green_channel(
        frames, bvps, face_region, False)
    # # downsample
    greenchannel_ds = green_channel.down_sample(greenchannel, bvp.shape[0])
    # # do align
    final_shift = align.corr_relate_align(
        greenchannel_ds, bvp)
    sync_frames = frames[:-final_shift]
    sync_bvps = bvps[final_shift:]
    green_channel.green_channel(sync_frames, sync_bvps, face_region, plot_flag)
    return sync_frames, sync_bvps


def ubfc_chunk(frames, bvps, clip_len):
    '''wrap certain length of data into chunks'''
    """
    size:(w,h)
    """
    assert(frames.shape[0] == bvps.shape[0])
    clip_num = frames.shape[0]//clip_len
    x_sub = []
    y_sub = []
    for i in range(clip_num):
        x_sub.append(frames[i*clip_len:(i+1)*clip_len])
        y_sub.append(bvps[i*clip_len:(i+1)*clip_len])
    return x_sub, y_sub

# gt_file = "./data_example/gt_HR.csv"
# time_file = "./data_example/time.txt"
# bvp_file = "./data_example/wave.csv"
# time = file_io.read_time(time_file)
# bvp = file_io.read_wave(bvp_file)
# gt = file_io.read_gt(gt_file)
# fps = len(time)/(time[len(time)-1]-time[0])*1000
# clip_length = 300


# videodata = file_io.read_video("./data_example/video001.avi")
# # videodata = skvideo.io.vread("./data_example/video001.avi")
# x, y, w, h = facial_detection(videodata[0])
# frames = np.zeros((videodata.shape[0],H,W,3))

# for i in range(0,videodata.shape[0]):
#     frame = videodata[i]
#     cropped_face = crop_face_down_sample(frame,x,y,w,h)
#     frames[i] = cropped_face

# if not os.path.exists("res"):
#     os.mkdir("res")

# # divide_clip(frames,bvp,gt,fps,clip_length,"res/example",True)


# print()
# # read the ground-true ecg
