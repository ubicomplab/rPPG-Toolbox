from neural_methods.trainer.BaseTrainer import BaseTrainer
import torch
from neural_methods.model.DeepPhys import DeepPhys
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
import logging
from metrics.metrics import calculate_metrics
from collections import OrderedDict
import glob
from math import ceil
import argparse
import glob
import os
import torch
import re
import pandas as pd
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from dataset import data_loader
from eval.post_process import *
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.model.TS_CAN import TSCAN
from neural_methods.model.DeepPhys import DeepPhys
from collections import OrderedDict
import random
import numpy as np




activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook

def getitem(inputs,labels,index):
    """Returns a clip of video(3,T,W,H) and it's corresponding signals(T)."""
    data = np.load(inputs[index])
    label = np.load(labels[index])
    data = np.transpose(data, (0, 3, 1, 2))
    data = np.float32(data)
    label = np.float32(label)
    item_path = inputs[index]
    item_path_filename = item_path.split('/')[-1]
    split_idx = item_path_filename.index('_')
    filename = item_path_filename[:split_idx]
    chunk_id = item_path_filename[split_idx+6:].split('.')[0]
    return data, label, filename, chunk_id

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



def calculate_metrics(predictions, labels):
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    label_hr = list()
    label_dict = read_label("UBFC")
    white_list = []
    for index in predictions.keys():
        if index in white_list:
            continue
        prediction = reform_data_from_dict(predictions[index])
        label = reform_data_from_dict(labels[index])
        gt_hr_fft, pred_hr_fft = calculate_metric_per_video(
            prediction, label, fs=30)
        # print(predictions[i]['prediction'], labels[i]['prediction'])
        gt_hr_peak, pred_hr_peak = calculate_metric_peak_per_video(
            prediction, label, fs=30)
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
    for metric in ['MAE', 'RMSE', 'MAPE', 'Pearson']:
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







if __name__ == "__main__":
    cached_path = "/data1/acsp/Yuzhe_Zhang/Toolbox_master_2/rPPG-Toolbox/PreprocessedData/UBFC_SizeW72_SizeH72_ClipLength180_DataTypeNormalized_Standardized_LabelTypeNormalized_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len180/"
    inputs_data = glob.glob(os.path.join(cached_path, "subject8_input*.npy"))
    labels_data = [input.replace("input", "label") for input in inputs_data]
    assert (len(inputs_data) == len(labels_data))
    len = len(inputs_data)
    print("load lens:", len)

    model = DeepPhys(img_size=72).to("cuda")
    model = torch.nn.DataParallel(model, device_ids=list(range(4)))
    person_model_paths = "/data1/acsp/Yuzhe_Zhang/Toolbox_master_2/rPPG-Toolbox/PreTrainedModels/deepphys_synthetics_10epoch_geforce2080ti_Epoch9.pth"
    model.load_state_dict(torch.load(person_model_paths, map_location=torch.device('cuda')))


    # model.attn_mask_1.register_forward_hook(get_activation('attn_mask_1'))
    # mask = activation['attn_mask_1'][0].view(34, 34)
    # mask = mask / torch.max(mask)

    predictions = dict()
    labels = dict()
    with torch.no_grad():
        for idx in range(len):
            data, label, filename, chunk_id = getitem(inputs_data,labels_data,idx)
            data = torch.from_numpy(data)
            label = torch.from_numpy(label)
            data_test, labels_test = data.to("cuda"), label.to("cuda")
            D, C, H, W = data_test.shape
            data_test = data_test.contiguous().view(D, C, H, W)
            labels_test = labels_test.contiguous().view(-1, 1)
            pred_ppg_test = model(data_test)
            subj_index = filename
            sort_index = int(chunk_id)
            if subj_index not in predictions.keys():
                predictions[subj_index] = dict()
                labels[subj_index] = dict()
            predictions[subj_index][sort_index] = pred_ppg_test
            labels[subj_index][sort_index] = labels_test
    calculate_metrics(predictions, labels)

    cached_path = "/data1/acsp/Yuzhe_Zhang/Toolbox_master_2/rPPG-Toolbox/PreprocessedData/UBFC_SizeW72_SizeH72_ClipLength180_DataTypeNormalized_Standardized_LabelTypeNormalized_Large_boxTrue_Large_size2.0_Dyamic_DetFalse_det_len180/"
    inputs_data_con = glob.glob(os.path.join(cached_path, "subject8_input*.npy"))
    labels_data_con = [input.replace("input", "label") for input in inputs_data_con]
    assert (len(inputs_data_con) == len(labels_data_con))
    len = len(inputs_data_con)
    print("load lens:", len)

    predictions_con = dict()
    labels_con = dict()
    with torch.no_grad():
        for idx in range(len):
            data, label, filename, chunk_id = getitem(inputs_data_con,labels_data_con,idx)
            data = torch.from_numpy(data)
            label = torch.from_numpy(label)
            data_test, labels_test = data.to("cuda"), label.to("cuda")
            D, C, H, W = data_test.shape
            data_test = data_test.contiguous().view(D, C, H, W)
            labels_test = labels_test.contiguous().view(-1, 1)
            pred_ppg_test = model(data_test)
            subj_index = filename
            sort_index = int(chunk_id)
            if subj_index not in predictions_con.keys():
                predictions_con[subj_index] = dict()
                labels_con[subj_index] = dict()
            predictions_con[subj_index][sort_index] = pred_ppg_test
            labels_con[subj_index][sort_index] = labels_test
    calculate_metrics(predictions_con, labels_con)



# for name, layer in model.named_modules():
# ...     if isinstance(layer, torch.nn.Conv2d):
# ...             print(name, layer)