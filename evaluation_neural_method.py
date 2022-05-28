""" The main function of rPPG deep model evaluation pipeline.

TODO: Adds detailed description for models and datasets supported.
An evaluation pipleine for neural network methods, including model loading, inference and ca
  Typical usage example:

  python evaluation_neural_method.py --data_path /mnt/data0/COHFACE/RawData --model_path store_model/physnet.pth --preprocess
  You should edit predict (model,data_loader,config) and add functions for definition,e.g, define_Physnet_model to support your models.
"""

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
from neural_methods.model.ts_can import TSCAN
from eval.post_process import *
from collections import OrderedDict

def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument('--config_file', required=False,
                        default="configs/PURE_PHYSNET_EVALUATION.yaml", type=str, help="The name of the model.")
    parser.add_argument(
        '--device',
        default=None,
        type=int,
        help="An integer to specify which gpu to use, -1 for cpu.")
    parser.add_argument(
        '--model_path', required=False, type=str)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--test_data_path', default=None, required=False,
                        type=str, help='The path of the data directory.')
    parser.add_argument('--log_path', default=None, type=str)
    return parser


def define_Physnet_model(config):
    model = PhysNet_padding_Encoder_Decoder_MAX(
        frames=config.MODEL.PHYSNET.FRAME_NUM).to(config.DEVICE)  # [3, T, 128,128]
    return model


def define_TSCAN_model(config):
    model = TSCAN(frame_depth=config.MODEL.TSCAN.FRAME_DEPTH,
                  img_size=config.DATA.PREPROCESS.H)
    return model


def load_model(model, config):
    if config.NUM_OF_GPU_TRAIN > 1:
        checkpoint = torch.load(config.INFERENCE.MODEL_PATH)
        state_dict = checkpoint
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' of dataparallel
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(torch.load(config.INFERENCE.MODEL_PATH))
    model = model.to(config.DEVICE)
    return model


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
    return hr


def physnet_predict(model, data_loader, config):
    """

    """
    predictions = list()
    labels = list()
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            data, label = batch[0].to(
                config.DEVICE), batch[1].to(config.DEVICE)
            prediction, _, _, _ = model(data)
            predictions.extend(prediction.to("cpu").numpy())
            labels.extend(label.to("cpu").numpy())
    return np.reshape(np.array(predictions), (-1)), np.reshape(np.array(labels), (-1))


def tscan_predict(model, data_loader, config):
    """ Model evaluation on the testing dataset."""
    print(" ====Testing===")
    predictions = dict()
    labels = dict()
    model.eval()
    with torch.no_grad():
        for _, test_batch in enumerate(data_loader):
            subj_index = test_batch[2][0]
            sort_index = int(test_batch[3][0])
            data_test, labels_test = test_batch[0].to(
                config.DEVICE), test_batch[1].to(config.DEVICE)
            N, D, C, H, W = data_test.shape
            data_test = data_test.view(N * D, C, H, W)
            labels_test = labels_test.view(-1, 1)
            data_test = data_test[:(
                                           N * D) // config.MODEL.TSCAN.FRAME_DEPTH * config.MODEL.TSCAN.FRAME_DEPTH]
            labels_test = labels_test[:(
                                               N * D) // config.MODEL.TSCAN.FRAME_DEPTH * config.MODEL.TSCAN.FRAME_DEPTH]
            pred_ppg_test = model(data_test)
            if subj_index not in predictions.keys():
                predictions[subj_index] = dict()
                labels[subj_index] = dict()
            predictions[subj_index][sort_index] = pred_ppg_test
            labels[subj_index][sort_index] = labels_test
    # return np.reshape(np.array(predictions), (-1)), np.reshape(np.array(labels), (-1))
    return predictions, labels


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
    white_list = ['601']
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
        label_hr.append(read_hr_label(label_dict, index))
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
            print("FFT Pearson:{0}".format(Pearson_FFT[0][1]))
            print("PEAK Pearson:{0}".format(Pearson_PEAK[0][1]))

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


def eval(loader, config):
    if config.MODEL.NAME == "Physnet":
        model = define_Physnet_model(config)
    elif config.MODEL.NAME == "Tscan":
        model = define_TSCAN_model(config)
    model = load_model(model, config)
    data = loader(name="inference",
                  data_path=config.DATA.TEST_DATA_PATH, config_data=config.DATA)
    data_loader = DataLoader(
        dataset=data, num_workers=2, batch_size=config.INFERENCE.BATCH_SIZE, shuffle=False)
    predictions, labels = tscan_predict(
        model, data_loader, config)
    calculate_metrics(predictions, labels, config)


if __name__ == "__main__":
    # parses arguments.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    # forms configurations.
    config = get_evaluate_config(args)
    print(config)

    writer = SummaryWriter(config.LOG.PATH)

    # loads data
    if config.DATA.DATASET == "COHFACE":
        loader = data_loader.COHFACELoader.COHFACELoader
    elif config.DATA.DATASET == "UBFC":
        loader = data_loader.UBFCLoader.UBFCLoader
    elif config.DATA.DATASET == "PURE":
        loader = data_loader.PURELoader.PURELoader
    elif config.DATA.DATASET == "SYNTHETICS":
        loader = data_loader.SyntheticsLoader.SyntheticsLoader
    else:
        raise ValueError(
            "Unsupported dataset! Currently supporting COHFACE, UBFC and PURE.")
    eval(loader, config)
