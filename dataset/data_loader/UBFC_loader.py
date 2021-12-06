import os
from dataset.data_loader import data_loader
from dataset.preprocess import data_preprocess
import numpy as np
import cv2

class UBFC_loader(data_loader):
    def __init__(self, video_file, bvp_file, name):
        super(UBFC_loader, self).__init__(video_file, bvp_file, name)

    def __len__(self):
        assert(self.preprocess)
        return self.count

    def __getitem__(self, index):
        x = np.load(self.xs[index])
        y = np.load(self.ys[index])
        x = np.transpose(x, (3, 0, 1, 2))
        return x, y
        # return super().__getitem__(index)

    def preprocessing(self):
        # TODO: add configure
        self.preprocess = True
        file_num = len(self.bvp_files)
        for i in range(file_num):
            self.read_video(self.video_files[i])
            self.read_wave(self.bvp_files[i])
            # self.synchronize()
            self.resize(128, 128)
            self.chunk(64)
            self.save()
        print(self.name, "data preprocess done")

    def read_video(self, video_file):
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        FrameRate = VidObj.get(cv2.CAP_PROP_FPS)
        FN = 0
        success, frame = VidObj.read()
        RGB = []

        while(success):
            # TODO: if different region
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)

            RGB.append(frame)
            success, frame = VidObj.read()
        self.frames = np.asarray(RGB)

    def read_wave(self, bvp_file):
        with open(bvp_file, "r") as f:
            str1 = f.read()
            str1 = str1.split("\n")
            bvp = str1[0]
            bvp = [float(x) for x in bvp.split()]
            hr = str1[1]
            hr = [float(x) for x in hr.split()]
            times = str1[2]
            times = [float(x) for x in times.split()]
            fs = len(times) / times[-1]
        self.bvps = np.asarray(bvp)

    def resize(self, H, W):
        face_region = data_preprocess.facial_detection(self.frames[0])
        self.frames = data_preprocess.resize(self.frames, H, W, face_region)

    def synchronize(self):
        self.frames, self.bvps = data_preprocess.synchronize(
            self.frames, self.bvps, True)

    def chunk(self, clip_length):
        self.frames, self.bvps = data_preprocess.ubfc_chunk(
            self.frames, self.bvps, clip_length)

    def save(self):
        if(not os.path.exists("preprocessed")):
            os.mkdir("preprocessed")
        for i in range(len(self.bvps)):
            assert(len(self.xs) == len(self.ys))
            assert(len(self.xs) == (self.count))
            x_path_name = "preprocessed"+os.sep + \
                str(self.name)+str(self.count)+"_x.npy"
            y_path_name = "preprocessed"+os.sep + \
                str(self.name)+str(self.count)+"_y.npy"
            self.xs.append(x_path_name)
            self.ys.append(y_path_name)
            np.save(x_path_name, self.frames[i])
            np.save(y_path_name, self.bvps[i])
            self.count += 1