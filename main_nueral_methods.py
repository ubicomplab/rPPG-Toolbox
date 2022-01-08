# TODO: Add descriptions for the file


# Lint as: python3
import numpy as np
import torch
from torch.utils.data import DataLoader
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from neural_methods.model.PhysNet import *
from neural_methods.model.rPPGNet import *
from dataset.data_loader.UBFC_loader import UBFC_loader
from tensorboardX import SummaryWriter
import argparse
import glob
from neural_methods.trainer.physnet_trainer import physnet_trainer
import os
import time


def get_data():
    bvp_files = glob.glob(args.data_dir + os.sep + "subject*/*.txt")
    video_files = glob.glob(args.data_dir + os.sep + "subject*/*.avi")

    return {
        "bvp": {
            "train": bvp_files,
            "valid": bvp_files,
            "test": bvp_files},
        "video": {
            "train": video_files,
            "valid": video_files,
            "test": video_files}}


def add_args(parser):
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument(
        '--device',
        default=0,
        type=int,
        help="an integer to specify which gpu to use, -1 for cpu")
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--model_name', default="physnet", type=str)
    parser.add_argument('--data_dir', default="G:\\UBFC_data",
                        type=str, help='The path of the data directory')
    t = time.localtime()
    parser.add_argument(
        '--name', default="{0}-{1}".format(str(t.tm_hour), str(t.tm_min)), type=str)
    return parser


def main(args, writer, data_loader):
    trainner = trainner_name(args, writer)
    trainner.train(dataloader)
    print("End")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    trainner_name = eval('{0}_trainer'.format(args.model_name))
    parser = trainner_name.add_trainer_args(parser)
    args = parser.parse_args()
    writer = SummaryWriter('runs/exp/' + args.name)

    # load data
    data_files = get_data()
    train_data = UBFC_loader(
        data_files["video"]["train"],
        data_files["bvp"]["train"],
        "train")
    train_data.preprocessing()
    valid_data = UBFC_loader(
        data_files["video"]["valid"],
        data_files["bvp"]["valid"],
        "valid")
    valid_data.preprocessing()
    test_data = UBFC_loader(
        data_files["video"]["test"],
        data_files["bvp"]["test"],
        "test")
    test_data.preprocessing()

    dataloader = {
        "train": DataLoader(
            dataset=train_data,
            num_workers=2,
            batch_size=args.batch_size,
            shuffle=True),
        "valid": DataLoader(
            dataset=valid_data,
            num_workers=2,
            batch_size=args.batch_size,
            shuffle=True),
        "test": DataLoader(dataset=test_data, num_workers=2,
                           batch_size=args.batch_size, shuffle=True)
    }
    main(args, writer, dataloader)
