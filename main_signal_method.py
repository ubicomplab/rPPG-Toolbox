import argparse
from tarfile import PAX_NUMBER_FIELDS
from xmlrpc.client import boolean
import cv2
import math
import numpy as np
from signal_methods.ICA import *
from signal_methods.POS_WANG import *
from signal_methods.CHROME_DEHAAN import *
from dataset.data_loader.UBFCLoader import UBFCLoader
from dataset.data_loader.PURELoader import PURELoader
from utils.utils import *

UBFC_PATH = "/data1/toolbox_data/UBFC/RawData/"
UBFC_FS = 30
METRICS = ["MAE", "RMSE", "MAPE", "Pearson"]

PURE_PATH = "/data1/toolbox_data/PURE/RawData/"
PURE_FS = 30


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default="chrome",
                        choices=["ica", "pos", "chrome"])
    parser.add_argument("--dataset", type=str, default="UBFC")
    parser.add_argument("--plotTF", type=boolean, default=False)
    parser.add_argument("--fs", type=int, default=30)
    args = parser.parse_args()
    FS = 0

    if(args.dataset == "UBFC"):
        data_loader = UBFCLoader
        FS = UBFC_FS
        PATH = UBFC_PATH
    elif(args.dataset == "PURE"):
        data_loader = PURELoader
        FS = PURE_FS
        PATH = PURE_PATH

    data_dirs = data_loader.get_data_signal_method(PATH)
    predict_hr_peak_all = []
    gt_hr_peak_all = []

    for i in range(len(data_dirs)):
        frames = data_loader.read_video(
            data_dirs[i]['frame_path'])
        bvps = data_loader.read_wave(
            data_dirs[i]['wave_path'])
        frames = resize(frames)
        if(args.method == "pos"):
            BVP, PR, PR_gt = POS_WANG(frames, bvps, FS, False)
        elif(args.method == "chrome"):
            BVP, PR, PR_gt = CHROME_DEHAAN(frames, bvps, FS, False)
        elif(args.method == "ica"):
            BVP, PR, PR_gt = ICA_POH(frames, bvps, FS, False)

        gt_hr, pre_hr = calculate_metric_peak_per_video(BVP, bvps, fs=FS)
        gt_fft_hr,pre_fft_hr = calculate_metric_per_video(BVP,bvps,fs=FS)
        print(gt_hr,pre_hr)
        predict_hr_peak_all.append(pre_hr)
        gt_hr_peak_all.append(gt_hr)

    predict_hr_peak_all = np.array(predict_hr_peak_all)
    gt_hr_peak_all = np.array(gt_hr_peak_all)
    for metric in METRICS:
        if metric == "MAE":
            MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
            print("Peak MAE (Peak Label):{0}".format(MAE_PEAK))
        elif metric == "RMSE":
            RMSE_PEAK = np.sqrt(
                np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
            print("PEAK RMSE (Peak Label):{0}".format(RMSE_PEAK))
        elif metric == "MAPE":
            MAPE_PEAK = np.mean(
                np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
            print("PEAK MAPE (Peak Label):{0}".format(MAPE_PEAK))
        elif metric == "Pearson":
            Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
            print("PEAK Pearson  (Peak Label):{0}".format(Pearson_PEAK[0][1]))
        else:
            raise ValueError("Wrong Test Metric Type")
