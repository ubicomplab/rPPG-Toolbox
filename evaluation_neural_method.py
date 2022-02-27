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


def get_UBFC_data(config):
    """Returns directories for train sets, validation sets and test sets.
    For the dataset structure, see dataset/dataloader/UBFC_dataloader.py """
    data_dirs = glob.glob(config.DATA.DATA_PATH + os.sep + "subject*")
    dirs = [{"index": re.search(
        'subject(\d+)', data_dir).group(0), "path": data_dir} for data_dir in data_dirs]
    return dirs


def get_COHFACE_data(config):
    """Returns directories for train sets, validation sets and test sets.
    For the dataset structure, see dataset/dataloader/COHFACE_dataloader.py """
    data_dirs = glob.glob(config.DATA.DATA_PATH + os.sep + "*")
    dirs = list()
    for data_dir in data_dirs:
        for i in range(4):
            subject = os.path.split(data_dir)[-1]
            dirs.append({"index": '{0}0{1}'.format(subject, i),
                        "path": os.path.join(data_dir, str(i))})
    return dirs[:10]


def get_PURE_data(config):
    """Returns directories for train sets, validation sets and test sets.
    For the dataset structure, see dataset/dataloader/PURE_dataloader.py """
    data_dirs = glob.glob(config.DATA.DATA_PATH + os.sep + "*-*")
    dirs = list()
    for data_dir in data_dirs:
        subject = os.path.split(data_dir)[-1].replace('-', '')
        dirs.append({"index": subject, "path": data_dir})
    return dirs


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
        '--model_path', required=True, type=str)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--data_path', default="G:\\COHFACE\\RawData", required=False,
                        type=str, help='The path of the data directory.')
    parser.add_argument('--log_path', default=None, type=str)
    return parser


def define_Physnet_model(config):
    model = PhysNet_padding_Encoder_Decoder_MAX(
        frames=config.MODEL.PHYSNET.FRAME_NUM).to(config.DEVICE)  # [3, T, 128,128]
    return model


def load_model(model, config):
    model.load_state_dict(torch.load(
        config.INFERENCE.MODEL_PATH))
    model = model.to(config.DEVICE)

    return model


def read_label(dataset):
    df = pd.read_csv("label/{0}_Comparison.csv".format(dataset))
    out_dict = df.to_dict(orient='index')
    out_dict = {str(value['Sample']): value for key, value in out_dict.items()}
    return out_dict


def read_hr_label(feed_dict, index):
    dict = feed_dict[index]
    if dict['Preferred'] == 'Peak Detection':
        hr = dict['Peak Detection']
    elif dict['Preferred'] == 'FFT':
        hr = dict['FFT']
    else:
        hr = dict['Peak Detection']
    return hr


def predict(model, data_loader, config):
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


def calculate_metrics(predictions, labels, config):
    predict_hr = list()
    rppg_hr = list()
    label_hr = list()
    label_dict = read_label(config.DATA.DATASET)
    for i in range(predictions.shape[0]):
        gt_hr, p_hr = calculate_metric_per_video(
            predictions[i]['prediction'], labels[i]['prediction'], fs=config.DATA.FS)
        rppg_hr.append(gt_hr)
        predict_hr.append(p_hr)
        label_hr.append(read_hr_label(label_dict, predictions[i]['index']))
    predict_hr = np.array(predict_hr)
    rppg_hr = np.array(rppg_hr)
    label_hr = np.array(label_hr)
    print(label_hr)
    for metric in config.TEST.METRICS:
        if metric == "MAE":
            MAE = np.mean(np.abs(predict_hr - label_hr))
            print("MAE:{0}".format(MAE))
        elif metric == "RMSE":
            RMSE = np.sqrt(np.mean(np.square(predict_hr - label_hr)))
            print("RMSE:{0}".format(RMSE))


def eval(data_files, loader, config):
    physnet_model = define_Physnet_model(config)
    physnet_model = load_model(physnet_model, config)
    predictions = list()
    labels = list()
    for file in data_files:
        data = loader(
            name="inference{0}".format(file['index']),
            data_dirs=[file],
            config_data=config.DATA)
        # TODO: add num_works to config
        data_loader = DataLoader(
            dataset=data, num_workers=2, batch_size=config.TRAIN.BATCH_SIZE, shuffle=True)
        prediction_per_video, label_per_video = predict(
            physnet_model, data_loader, config)
        predictions.append(
            {'prediction': prediction_per_video, 'index': file['index']})
        labels.append({'prediction': label_per_video, 'index': file['index']})

    calculate_metrics(np.array(predictions), np.array(labels), config)


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
        data_files = get_COHFACE_data(config)
        loader = data_loader.COHFACELoader.COHFACELoader
    elif config.DATA.DATASET == "UBFC":
        data_files = get_UBFC_data(config)
        loader = data_loader.UBFCLoader.UBFCLoader
    elif config.DATA.DATASET == "PURE":
        data_files = get_PURE_data(config)
        loader = data_loader.PURELoader.PURELoader
    else:
        raise ValueError(
            "Unsupported dataset! Currently supporting COHFACE, UBFC and PURE.")
    eval(data_files, loader, config)
