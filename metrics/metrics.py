import argparse
import glob
import os
import torch
import re
import pandas as pd
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from config import get_evaluate_config
from dataset import data_loader
from eval.post_process import *
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.model.TS_CAN import TSCAN
from neural_methods.model.DeepPhys import DeepPhys
from collections import OrderedDict
import random
import numpy as np

def read_label(dataset):
    df = pd.read_csv("label/{0}_Comparison.csv".format(dataset))
    out_dict = df.to_dict(orient='index')
    out_dict = {str(value['VideoID']): value for key,
                                                 value in out_dict.items()}
    return out_dict


def read_hr_label(feed_dict, index):
    # For UBFC only
    if index[:7] == 'subject':
        index = index[7:]
    video_dict = feed_dict[index]
    if video_dict['Preferred'] == 'Peak Detection':
        hr = video_dict['Peak Detection']
    elif video_dict['Preferred'] == 'FFT':
        hr = video_dict['FFT']
    else:
        hr = video_dict['Peak Detection']
    return index, hr

def reform_data_from_dict(data):
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)
    return np.reshape(sort_data.cpu(), (-1))



def calculate_metrics(predictions, labels, config):
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    label_hr = list()
    label_dict = read_label(config.DATA.DATASET)
    white_list = []
    for index in predictions.keys():
        if index in white_list:
            continue
        prediction = reform_data_from_dict(predictions[index])
        label = reform_data_from_dict(labels[index])
        gt_hr_fft, pred_hr_fft = calculate_metric_per_video(
            prediction, label, fs=config.DATA.FS)
        # print(predictions[i]['prediction'], labels[i]['prediction'])
        gt_hr_peak, pred_hr_peak = calculate_metric_peak_per_video(
            prediction, label, fs=config.DATA.FS)
        gt_hr_fft_all.append(gt_hr_fft)
        predict_hr_fft_all.append(pred_hr_fft)
        predict_hr_peak_all.append(pred_hr_peak)
        gt_hr_peak_all.append(gt_hr_peak)
        video_index, GT_HR = read_hr_label(label_dict, index)
        label_hr.append(GT_HR)
        if abs(GT_HR - pred_hr_fft) > 10:
            print('Video Index: ', video_index)
            print('GT HR: ', GT_HR)
            print('Pred HR: ', pred_hr_fft)
    predict_hr_peak_all = np.array(predict_hr_peak_all)
    predict_hr_fft_all = np.array(predict_hr_fft_all)
    gt_hr_peak_all = np.array(gt_hr_peak_all)
    gt_hr_fft_all = np.array(gt_hr_fft_all)
    label_hr_all_manual = np.array(label_hr)
    for metric in config.TEST.METRICS:
        if metric == "MAE":
            MAE_FFT = np.mean(np.abs(predict_hr_fft_all - label_hr_all_manual))
            MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - label_hr_all_manual))
            print("FFT MAE:{0}".format(MAE_FFT))
            print("Peak MAE:{0}".format(MAE_PEAK))

            MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_peak_all))
            MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
            print("FFT MAE (Peak Label):{0}".format(MAE_FFT))
            print("Peak MAE (Peak Label):{0}".format(MAE_PEAK))

            MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
            MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_fft_all))
            print("FFT MAE (FFT Label):{0}".format(MAE_FFT))
            print("Peak MAE (FFT Label):{0}".format(MAE_PEAK))

        elif metric == "RMSE":
            RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - label_hr_all_manual)))
            RMSE_PEAK = np.sqrt(np.mean(np.square(predict_hr_peak_all - label_hr_all_manual)))
            print("FFT RMSE:{0}".format(RMSE_FFT))
            print("PEAK RMSE:{0}".format(RMSE_PEAK))

            RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_peak_all)))
            RMSE_PEAK = np.sqrt(np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
            print("FFT RMSE (Peak Label):{0}".format(RMSE_FFT))
            print("PEAK RMSE (Peak Label):{0}".format(RMSE_PEAK))

            RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
            RMSE_PEAK = np.sqrt(np.mean(np.square(predict_hr_peak_all - gt_hr_fft_all)))
            print("FFT RMSE (FFT Label):{0}".format(RMSE_FFT))
            print("PEAK RMSE (FFT Label):{0}".format(RMSE_PEAK))

        elif metric == "MAPE":
            MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - label_hr_all_manual) / label_hr_all_manual)) * 100
            MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - label_hr_all_manual) / label_hr_all_manual)) * 100
            print("FFT MAPE:{0}".format(MAPE_FFT))
            print("PEAK MAPE:{0}".format(MAPE_PEAK))

            MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
            MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
            print("FFT MAPE (Peak Label):{0}".format(MAPE_FFT))
            print("PEAK MAPE (Peak Label):{0}".format(MAPE_PEAK))

            MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
            MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
            print("FFT MAPE (FFT Label):{0}".format(MAPE_FFT))
            print("PEAK MAPE (FFT Label):{0}".format(MAPE_PEAK))

        elif metric == "Pearson":
            Pearson_FFT = np.corrcoef(predict_hr_fft_all, label_hr_all_manual)
            Pearson_PEAK = np.corrcoef(predict_hr_peak_all, label_hr_all_manual)
            print("FFT Pearson:{0}".format(abs(Pearson_FFT[1, 0])))
            print("PEAK Pearson:{0}".format(abs(Pearson_PEAK[1, 0])))
            # print("FFT Pearson:{0}".format(Pearson_FFT[0][1]))
            # print("PEAK Pearson:{0}".format(Pearson_PEAK[0][1]))

            Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_peak_all)
            Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
            print("FFT Pearson (Peak Label):{0}".format(Pearson_FFT[0][1]))
            print("PEAK Pearson  (Peak Label):{0}".format(Pearson_PEAK[0][1]))

            Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
            Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_fft_all)
            print("FFT Pearson (FFT Label):{0}".format(Pearson_FFT[0][1]))
            print("PEAK Pearson (FFT Label):{0}".format(Pearson_PEAK[0][1]))

        else:
            raise ValueError("Wrong Test Metric Type")