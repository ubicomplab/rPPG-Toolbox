import cv2
import math
import numpy as np
from scipy import io as scio



def read_time(path):
    time = []
    with open(path,"r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            time.append(int(line))
    return np.asarray(time)

def read_gt(path):
    gt = []
    with open(path,"r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            if line == "HR":
                continue
            gt.append(int(line))
    return np.asarray(gt)


def read_wave(path):
    wave = []
    with open(path,"r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            if line == "Wave":
                continue
            wave.append(int(line))
    return np.asarray(wave)

def read_video(VideoFile):
    #Standard:
    VidObj = cv2.VideoCapture(VideoFile)
    VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
    FrameRate = VidObj.get(cv2.CAP_PROP_FPS)
    FN = 0
    success, frame = VidObj.read()
    RGB = []

    while(success):
        #TODO: if different region
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
        frame = np.asarray(frame)

        RGB.append(frame)
        success, frame = VidObj.read()
    return np.asarray(RGB)

# def ECG_from_mat(ECGFile):
#     ECG = scio.loadmat(ECGFile)["ECG"]
#     data = ECG[0,0]["data"]
#     return data[:2500,0]

