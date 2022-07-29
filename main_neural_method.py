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
import torch
import random
import numpy as np
import time

RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
g = torch.Generator()
g.manual_seed(RANDOM_SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument('--config_file', required=False,
                        default="configs/PURE_PURE_UBFC_PHYSNET_BASIC.yaml", type=str, help="The name of the model.")
    # SCAMPS_SCAMPS_UBFC_TSCAN_BASIC.yaml
    # SCAMPS_SCAMPS_UBFC_DEEPPHYS_BASIC.yaml
    # PURE_PURE_UBFC_TSCAN_BASIC.yaml
    # PURE_PURE_UBFC_DEEPPHYS_BASIC
    # PURE_PURE_UBFC_PHYSNET_BASIC.yaml
    # UBFC_UBFC_PURE_TSCAN_BASIC.yaml
    return parser


def train_and_test(config, data_loader):
    """Trains the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(config)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config)
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    model_trainer.train(data_loader)
    model_trainer.test(data_loader)

def test(config, data_loader):
    """Tests the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(config)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config)
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    model_trainer.test(data_loader)



if __name__ == "__main__":
    # parse arguments.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    # configurations.
    config = get_config(args)
    print(config)

    # train_loader
    if config.TRAIN.DATA.DATASET == "COHFACE":
        train_loader = data_loader.COHFACELoader.COHFACELoader
    elif config.TRAIN.DATA.DATASET == "UBFC":
        train_loader = data_loader.UBFCLoader.UBFCLoader
    elif config.TRAIN.DATA.DATASET == "PURE":
        train_loader = data_loader.PURELoader.PURELoader
    elif config.TRAIN.DATA.DATASET == "SYNTHETICS":
        train_loader = data_loader.SyntheticsLoader.SyntheticsLoader
    else:
        raise ValueError(
            "Unsupported dataset! Currently supporting COHFACE, UBFC and PURE.")

    # valid_loader
    if config.VALID.DATA.DATASET == "COHFACE":
        valid_loader = data_loader.COHFACELoader.COHFACELoader
    elif config.VALID.DATA.DATASET == "UBFC":
        valid_loader = data_loader.UBFCLoader.UBFCLoader
    elif config.VALID.DATA.DATASET == "PURE":
        valid_loader = data_loader.PURELoader.PURELoader
    elif config.VALID.DATA.DATASET == "SYNTHETICS":
        valid_loader = data_loader.SyntheticsLoader.SyntheticsLoader
    else:
        raise ValueError(
            "Unsupported dataset! Currently supporting COHFACE, UBFC and PURE.")

    # test_loader
    if config.TEST.DATA.DATASET == "COHFACE":
        test_loader = data_loader.COHFACELoader.COHFACELoader
    elif config.TEST.DATA.DATASET == "UBFC":
        test_loader = data_loader.UBFCLoader.UBFCLoader
    elif config.TEST.DATA.DATASET == "PURE":
        test_loader = data_loader.PURELoader.PURELoader
    elif config.TEST.DATA.DATASET == "SYNTHETICS":
        test_loader = data_loader.SyntheticsLoader.SyntheticsLoader
    else:
        raise ValueError(
            "Unsupported dataset! Currently supporting COHFACE, UBFC and PURE.")
    data_loader = dict()
    if config.TRAIN.DATA.DATA_PATH:
        train_data_loader = train_loader(
            name="train",
            data_path=config.TRAIN.DATA.DATA_PATH,
            config_data=config.TRAIN.DATA)
        data_loader['train'] = DataLoader(
            dataset=train_data_loader,
            num_workers=4,
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g
        )
    else:
        data_loader['train'] = None
    if config.TRAIN.DATA.DATA_PATH:
        valid_data = valid_loader(
            name="valid",
            data_path=config.VALID.DATA.DATA_PATH,
            config_data=config.VALID.DATA)
        data_loader["valid"] = DataLoader(
            dataset=valid_data,
            num_workers=4,
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g
        )
    else:
        data_loader['valid'] = None

    if config.TEST.DATA.DATA_PATH:
        test_data = test_loader(
            name="test",
            data_path=config.TEST.DATA.DATA_PATH,
            config_data=config.TEST.DATA)
        data_loader["test"] = DataLoader(
            dataset=test_data,
            num_workers=4,
            batch_size=config.INFERENCE.BATCH_SIZE,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g
        )
    else:
        data_loader['test'] = None
    if config.TRAIN_OR_TEST == "train_and_test":
        train_and_test(config, data_loader)
    elif config.TRAIN_OR_TEST == "only_test":
        test(config, data_loader)
    else:
        print("TRAIN_OR_TEST only support train_and_test or only_test !")
