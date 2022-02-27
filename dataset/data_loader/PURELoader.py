"""The dataloader for PURE datasets.

Details for the PURE Dataset see https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure
If you use this dataset, please cite the following publication:
Stricker, R., Müller, S., Gross, H.-M.
Non-contact Video-based Pulse Rate Measurement on a Mobile Service Robot
in: Proc. 23st IEEE Int. Symposium on Robot and Human Interactive Communication (Ro-Man 2014), Edinburgh, Scotland, UK, pp. 1056 - 1062, IEEE 2014
"""
import os
import cv2
import json
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from utils.utils import sample


class PURELoader(BaseLoader):
    """The data loader for the PURE dataset."""

    def __init__(self, name, data_dirs, config_data):
        """Initializes an PURE dataloader.
            Args:
                data_dirs(list): A list of paths storing raw video and bvp data.
                Each contains 4 one-minute videos for one subject.
                e.g. [RawData/01-01,RawData/01-02,...,RawData/ii-jj] for below dataset structure:
                -----------------
                     RawData/
                     |   |-- 01-01/
                     |      |-- 01-01/
                     |      |-- 01-01.json
                     |   |-- 01-02/
                     |      |-- 01-02/
                     |      |-- 01-02.json
                     |...
                     |   |-- ii-jj/
                     |      |-- ii-jj/
                     |      |-- ii-jj.json
                -----------------
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_dirs, config_data)

    def preprocess_dataset(self, config_preprocess):
        """Preprocesses the raw data."""
        file_num = len(self.data_dirs)
        for i in range(file_num):
            print(i)
            filename = os.path.split(self.data_dirs[i]['path'])[-1]
            frames = self.read_video(
                os.path.join(
                    self.data_dirs[i]['path'],
                    filename, ""))
            bvps = self.read_wave(
                os.path.join(
                    self.data_dirs[i]['path'],
                    "{0}.json".format(filename)))
            bvps = sample(bvps, frames.shape[0])
            # Slow Translation and Fast Translation setups.
            if (filename[-2:] == "03") or (filename[-2:] == "04"):
                larger_box = True
            else:
                larger_box = False
            frames_clips, bvps_clips = self.preprocess(
                frames, bvps, config_preprocess, larger_box)
            self.len += self.save(frames_clips, bvps_clips,
                                  self.data_dirs[i]['index'])

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T,H,W,3) """
        frames = list()
        for _, _, files in os.walk(video_file):
            for file in files:
                img = cv2.imread(os.path.join(video_file, file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img)
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with open(bvp_file, "r") as f:
            labels = json.load(f)
            waves = [label["Value"]["waveform"]
                     for label in labels["/FullPackage"]]
        return np.asarray(waves)
