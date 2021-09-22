import cv2
import math
import numpy as np
from scipy import io as scio

def normalization(data):
    _range = np.max(abs(data))
    return data / _range

def process_video(VideoFile,StartTime,Duration):
    #Standard:
    w = 140*4
    h = 280*4
    VidObj = cv2.VideoCapture(VideoFile)
    VidObj.set(cv2.CAP_PROP_POS_MSEC, StartTime * 1000)
    FrameRate = VidObj.get(cv2.CAP_PROP_FPS)
    FramesNumToRead = math.ceil(Duration * FrameRate)+1  # TODO:cell?

    T = np.zeros((FramesNumToRead, 1))

    FN = 0
    success, frame = VidObj.read()
    RGB = np.zeros((FramesNumToRead,w,h, 3))
    CurrentTime = VidObj.get(cv2.CAP_PROP_POS_MSEC)
    EndTime = StartTime + Duration

    while(success and ( CurrentTime <= (EndTime*1000) )):
        T[FN] = CurrentTime
        #TODO: if different region
        frame = cv2.cvtColor(np.array(frame).astype('float32'), cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame,(h, w))
        frame = np.asarray(frame)

        RGB[FN] = frame
        success, frame = VidObj.read()

        CurrentTime = VidObj.get(cv2.CAP_PROP_POS_MSEC)
        FN+=1
    return T[:FN],RGB[:FN]

def ECG_from_mat(ECGFile):
    ECG = scio.loadmat(ECGFile)["ECG"]
    data = ECG[0,0]["data"]
    return data[:2500,0]


def read_wave(path):
    wave = []
    with open(path,"r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            if line == "Wave":
                continue
            wave.append(int(line))
    return np.asarray(wave)