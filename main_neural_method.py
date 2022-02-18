""" The main function of rPPG deep learning pipeline.

TODO: Adds detailed description for models and datasets supported.
An end-to-end training pipleine for neural network methods.
  Typical usage example:

  python main_neural_method.py --config_file configs/COHFACE_PHYSNET_BASIC.yaml --data_path "G:\\COHFACE" --preprocess
  python main_neural_method.py --config_file configs/TSCAN_COHFACE_BASIC.yaml --data_path "G:\\COHFACE"
"""

import argparse
import glob
import os
import time
import logging
from config import get_config
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from dataset import data_loader
from neural_methods import trainer


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
    parser.add_argument('--config_file', required=False,
                        default="configs/COHFACE_PHYSNET_BASIC.yaml", type=str, help="The name of the model.")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument(
        '--device',
        default=None,
        type=int,
        help="An integer to specify which gpu to use, -1 for cpu.")
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--data_path', default="G:\\COHFACE\\RawData", required=False,
                        type=str, help='The path of the data directory.')
    parser.add_argument('--epochs', default=None, type=int)
    parser.add_argument('--log_path', default=None, type=str)
    parser.add_argument('--model_dir', default=None, type=str)
    return parser


def train(config, writer, data_loader):
    """Trains the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, writer)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config, writer)

    model_trainer.train(data_loader)


if __name__ == "__main__":
    # parses arguments.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    # forms configurations.
    config = get_config(args)
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
    train(config, writer, dataloader)
