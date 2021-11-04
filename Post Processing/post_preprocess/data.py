import file_io
import numpy as np
import cv2
import green_channel
import synchronize
import data_preprocess
class VIPL_data:
    def __init__(self,video_file,bvp_file):
        self.frames = file_io.read_video(video_file)
        self.bvps = file_io.read_wave(bvp_file)
    def resize(self,H,W):
        resize_frames = np.zeros((self.frames.shape[0], H, W, 3))
        for i in range(0, self.frames.shape[0]):
            frame = self.frames[i]
            resize_frames[i] = cv2.resize(frame, (W,H))
        return resize_frames
    def synchronize(self):
        face_region = data_preprocess.facial_detection(self.frames[0])
        greenchannel, bvp = green_channel.green_channel(self.frames, self.bvps,face_region)
        # # downsample
        greenchannel_ds = green_channel.down_sample(greenchannel, bvp.shape[0])
        # # do align
        final_shift = synchronize.corr_relate_align(greenchannel_ds, bvp)
        sync_frames = self.frames[:-final_shift]
        sync_bvps = self.bvps[final_shift:]
        green_channel.green_channel(sync_frames, sync_bvps,face_region)
        return sync_frames, sync_bvps



bvp_file = "./data_example/wave.csv"
video_file = "./data_example/video001.avi"

data = VIPL_data(video_file,bvp_file)
data.synchronize()