""" The main function of rPPG deep learning pipeline.

TODO: Adds detailed description for models and datasets supported.
An end-to-end training pipleine for neural network methods.
  Typical usage example:


  python main_neural_method.py --config_file configs/COHFACE_TSCAN_BASIC.yaml --data_path "G:\\COHFACE"
"""

import argparse
import glob
import os
import time
import logging
import re
import sys
import tqdm
from config import get_config
from torch.utils.data import DataLoader
from dataset import data_loader
from neural_methods import trainer


def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument('--config_file', required=False,
                        default="configs/PURE_PHYSNET_BASIC.yaml", type=str, help="The name of the model.")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument(
        '--device',
        default=None,
        type=int,
        help="An integer to specify which gpu to use, -1 for cpu.")
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--train_data_path', default=None, required=False,
                        type=str, help='The path of the data directory.')
    parser.add_argument('--valid_data_path', default=None, required=False,
                        type=str, help='The path of the data directory.')
    parser.add_argument('--epochs', default=None, type=int)
    parser.add_argument('--log_level', default="DEBUG", type=str)
    parser.add_argument('--log_path', default="terminal", type=str)
    parser.add_argument('--model_dir', default=None, type=str)
    return parser


def train(config, data_loader):
    """Trains the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(
            config)

    model_trainer.train(data_loader)


if __name__ == "__main__":
    # parse arguments.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    # configurations.
    config = get_config(args)
    #logging
    if args.log_path=="terminal":
        if args.log_level=="DEBUG" :
            logging.basicConfig(level=logging.DEBUG)
        elif args.log_level=="INFO":
            logging.basicConfig(level=logging.INFO)
        elif args.log_level=="WARNING":
            logging.basicConfig(level=logging.WARNING)
        elif args.log_level=="ERROR":
            logging.basicConfig(level=logging.ERROR)
        elif args.log_level=="CRITICAL":
            logging.basicConfig(level=logging.CRITICAL)
    else:
        if args.log_level=="DEBUG" :
            logging.basicConfig(level=logging.DUBUG,filemode='w',filename=args.log_path)
        elif args.log_level=="INFO":
            logging.basicConfig(level=logging.INFO,filemode='w',filename=args.log_path)
        elif args.log_level=="WARNING":
            logging.basicConfig(level=logging.WARNING,filemode='w',filename=args.log_path)
        elif args.log_level=="ERROR":
            logging.basicConfig(level=logging.ERROR,filemode='w',filename=args.log_path)
        elif args.log_level=="CRITICAL":
            logging.basicConfig(level=logging.CRITICAL,filemode='w',filename=args.log_path)
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

    data_loader = dict()
    train_data_loader = loader(
        name="train",
        data_path=config.DATA.TRAIN_DATA_PATH,
        config_data=config.DATA)
    data_loader['train'] = DataLoader(
        dataset=train_data_loader,
        num_workers=2,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True
    )
    if config.DATA.VALID_DATA_PATH:
        valid_data = loader(
            name="valid",
            data_path=config.DATA.VALID_DATA_PATH,
            config_data=config.DATA)
        data_loader["valid"] = DataLoader(
            dataset=valid_data,
            num_workers=2,
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=True)
    else:
        data_loader['valid'] = None
    train(config, data_loader)
