""" The main function of rPPG deep learning pipeline.

TODO: Adds detailed description for models and datasets supported.
An end-to-end training pipleine for neural network methods.
  Typical usage example:

<<<<<<< HEAD:main_neural_methods.py
  python main_neural_methods.py --config_file configs/COHFACE_PHYSNET_BASIC.yaml --data_path "G:\\COHFACE" --preprocess
  python main_neural_methods.py --config_file configs/TSCAN_COHFACE_BASIC.yaml --data_path "G:\\COHFACE"
=======

  python main_neural_method.py --config_file configs/COHFACE_TSCAN_BASIC.yaml --data_path "G:\\COHFACE"
>>>>>>> e647fc4a27f9c434369cf5cb498e82142ce7137e:main_neural_method.py
"""

import argparse
import glob
import os
import time
import logging
import re
from config import get_config
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from dataset import data_loader
from neural_methods import trainer


def get_UBFC_data(config):
    """Returns directories for train sets, validation sets and test sets.
    For the dataset structure, see dataset/dataloader/UBFC_dataloader.py """
    data_dirs = glob.glob(config.DATA.DATA_PATH + os.sep + "subject*")
    dirs = [{"index":re.search('subject(\d+)',data_dir).group(0),"path":data_dir} for data_dir in data_dirs]
    return dirs



def get_COHFACE_data(config):
    """Returns directories for train sets, validation sets and test sets.
    For the dataset structure, see dataset/dataloader/COHFACE_dataloader.py """
    data_dirs = glob.glob(config.DATA.DATA_PATH + os.sep + "*")
    dirs = list()
    for data_dir in data_dirs:
        for i in range(4):
            subject = os.path.split(data_dir)[-1]
            dirs.append({"index": '{0}0{1}'.format(subject,i), "path": os.path.join(data_dir,str(i))})
    return dirs[:10]



def get_PURE_data(config):
    """Returns directories for train sets, validation sets and test sets.
    For the dataset structure, see dataset/dataloader/PURE_dataloader.py """
    data_dirs = glob.glob(config.DATA.DATA_PATH + os.sep + "*-*")
    dirs = list()
    for data_dir in data_dirs:
        subject = os.path.split(data_dir)[-1].replace('-','')
        dirs.append({"index": subject, "path": data_dir})
    return dirs


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

    print(data_files)
    train_data = loader(
        name="train",
        data_dirs=data_files,
        config_data=config.DATA)
    valid_data = loader(
        name="valid",
        data_dirs=data_files[:2],
        config_data=config.DATA)
    test_data = loader(
        name="test",
        data_dirs=data_files[:2],
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
