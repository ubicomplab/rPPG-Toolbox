    

import os
import numpy as np

import glob
import zipfile
import os
import re

import cv2
from skimage.util import img_as_float
import numpy as np
import pandas as pd
import pickle 


data_path = "F:\\BP4D+_v0.2" 
subject_trial = 'F001'
trial = 'T1'

# GRAB EACH FRAME FROM ZIP FILE
imgzip = open(os.path.join(data_path, '2D+3D', subject_trial+'.zip'))
zipfile_path = os.path.join(data_path, '2D+3D', subject_trial+'.zip')

cnt = 0 # frame count / index
with zipfile.ZipFile(zipfile_path, "r") as zippedImgs:
    for ele in zippedImgs.namelist():
        ext = os.path.splitext(ele)[-1]
        ele_task = str(ele).split('/')[1]

        if ext == '.jpg' and ele_task == trial:
            data = zippedImgs.read(ele)
            frame = cv2.imdecode(np.fromstring(data, np.uint8), cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # TODO WACV should this be a float 32???

            path = './bp4d_sample_frame.npy'
            np.save(path, frame)

            break