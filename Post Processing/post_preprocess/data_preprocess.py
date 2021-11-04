import numpy as np
import skvideo.io
import cv2
import math
import numpy as np
import file_io
from green_channel import gc_and_align
import os
W = 128
H = 128


def facial_detection(frame):
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_zone = detector.detectMultiScale(frame)
    if(len(face_zone) < 1):
        print("ERROR:No Face Detected")
    if(len(face_zone) >= 2):
        print("WARN:More than one faces are detected(Only cropping one face)")
    x,y,w,h = face_zone[0]
    return x,y,w,h

def crop_face_down_sample(frame,x,y,w,h):
    frame = frame[y:y + h, x:x + w]
    frame = cv2.resize(frame, (W,H))
    return frame

def down_sample(a,len):
    return np.interp(np.linspace(1,a.shape[0],len),np.linspace(1,a.shape[0],a.shape[0]),a)







gt_file = "./data_example/gt_HR.csv"
time_file = "./data_example/time.txt"
bvp_file = "./data_example/wave.csv"
time = file_io.read_time(time_file)
bvp = file_io.read_wave(bvp_file)
gt = file_io.read_gt(gt_file)
fps = len(time)/(time[len(time)-1]-time[0])*1000
clip_length = 300


videodata = file_io.read_video("./data_example/video001.avi")
# videodata = skvideo.io.vread("./data_example/video001.avi")
x, y, w, h = facial_detection(videodata[0])
frames = np.zeros((videodata.shape[0],H,W,3))

for i in range(0,videodata.shape[0]):
    frame = videodata[i]
    cropped_face = crop_face_down_sample(frame,x,y,w,h)
    frames[i] = cropped_face

if not os.path.exists("res"):
    os.mkdir("res")

# divide_clip(frames,bvp,gt,fps,clip_length,"res/example",True)


print()
# read the ground-true ecg

