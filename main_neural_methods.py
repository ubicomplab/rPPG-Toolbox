""" The main function for the rPPG deep learning pipeline.

TODO: Adds detailed description for models and datasets supported.
An end-to-end training pipleine for neural network methods.
  Typical usage example:

  python main.py --model_name physnet --data_dir "G:\\UBFC_data"
"""

import argparse
import glob
import os
import time
import logging
from config import get_config
from torch.utils.data import DataLoader
from dataset.data_loader.data_loader import data_loader
from dataset.data_loader.UBFC_loader import UBFC_loader
from dataset.data_loader.COHFACE_loader import COHFACE_loader
from dataset.data_loader.PURE_loader import PURE_loader
from tensorboardX import SummaryWriter
from neural_methods.trainer.trainer import trainer
from neural_methods.trainer.physnet_trainer import physnet_trainer

def get_UBFC_data(config):
    """Returns directories for train sets, validation sets and test sets.
    For the dataset structure, see dataset/dataloader/UBFC_dataloader.py """
    data_dirs = glob.glob(config.DATA.DATA_PATH + os.sep + "subject*")
    return {
        "train": data_dirs[:-2],
        "valid": data_dirs[-2:-1],
        "test": data_dirs[-1:]
    }


def get_COHFACE_data(config):
    """Returns directories for train sets, validation sets and test sets.
    For the dataset structure, see dataset/dataloader/COHFACE_dataloader.py """
    data_dirs = glob.glob(config.DATA.DATA_PATH + os.sep + "*")
    return {
        "train": data_dirs[:2],
        "valid": data_dirs[-2:-1],
        "test": data_dirs[-1:]
    }


def get_PURE_data(config):
    """Returns directories for train sets, validation sets and test sets.
    For the dataset structure, see dataset/dataloader/PURE_dataloader.py """
    data_dirs = glob.glob(config.DATA.DATA_PATH + os.sep + "*-*")
    return {
        "train": data_dirs[:-2],
        "valid": data_dirs[-2:-1],
        "test": data_dirs[-1:]
    }

def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument('--cfg', default="configs/COHFACE_PHYSNET_BASIC.yaml", type=str, help="The name of the model.")
    parser.add_argument('--model_name', default=None, type=str, help="The name of the model.")
    parser.add_argument('--dataset', default=None,choices=["COHFACE","PURE","UBFC"], type=str, help="The Dataset. Supporting UBFC/PURE/COHFACE.")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument(
        '--device',
        default=None,
        help="An integer to specify which gpu to use, -1 for cpu.")
    parser.add_argument('--batch_size', default=None,type=int)
    parser.add_argument('--data_path', default=None,required=True,
                        type=str, help='The path of the data directory.')
    parser.add_argument('--epochs',default=None,type=int)
    parser.add_argument('--log_path',default=None, type=str)
    return parser


def main(config, writer, data_loader):
    """Trains the model."""
    trainer_name = eval('{0}_trainer'.format(config.MODEL.NAME))
    trainner = trainer_name(config, writer)
    trainner.train(data_loader)


if __name__ == "__main__":
    # parses arguments.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.add_trainer_args(parser)
    parser = data_loader.add_data_loader_args(parser)
    args = parser.parse_args()

    print(args)
    # forms configurations.
    config = get_config(args)
    print(config)

    writer = SummaryWriter(config.LOG.PATH)

    # loads data
    data_files = eval("get_{0}_data".format(config.DATA.DATASET))(config)
    loader = eval("{0}_loader".format(config.DATA.DATASET))
    train_data = loader(
        name="train",
        data_dirs=data_files["train"],
        config_data=config.DATA)
    valid_data = loader(
        name="valid",
        data_dirs=data_files["valid"],
        config_data=config.DATA)
    test_data = loader(
        name="test",
        data_dirs=data_files["test"],
        config_data=config.DATA)
    dataloader = {
        "train": DataLoader(
            dataset=train_data,
            num_workers=2,
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=True),
        "valid": DataLoader(
            dataset=valid_data,
            num_workers=2,
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=True),
        "test": DataLoader(dataset=test_data, num_workers=2,
                           batch_size=config.TRAIN.BATCH_SIZE, shuffle=True)
    }
    main(config, writer, dataloader)
