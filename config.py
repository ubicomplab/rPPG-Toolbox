# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN

# TODO : do train/ do preprocessing
_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Path to preprocessing data, could be overwritten by command line argument
_C.DATA.CACHED_PATH = ''
# Dataset name, coule be overwritten by command line argument
_C.DATA.DATASET = ''
_C.DATA.DO_PREPROCESS = True

# -----------------------------------------------------------------------------
# Data preprocessing
# TODO: add other preprocessing configs
# -----------------------------------------------------------------------------
_C.DATA.PREPROCESS = CN()
_C.DATA.PREPROCESS.CROP_FACE = True
_C.DATA.PREPROCESS.W = 128
_C.DATA.PREPROCESS.H = 128
_C.DATA.PREPROCESS.CLIP_LENGTH = 64


# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model name
_C.MODEL.NAME = 'physnet'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
_C.MODEL.MODEL_PATH = 'store_model'

# Specific parameters for physnet parameters
_C.MODEL.PHYSNET = CN()
_C.MODEL.PHYSNET.FRAME_NUM = 64

#TODO: Specific parameters for CAN


# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 50
_C.TRAIN.BATCH_SIZE = 4
_C.TRAIN.LR = 1e-4
# Optimizer
_C.TRAIN.OPTIMIZER = CN()
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-4
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()

# -----------------------------------------------------------------------------
# Device settings
# -----------------------------------------------------------------------------
_C.DEVICE = "cuda:0"

# -----------------------------------------------------------------------------
# Log settings
# -----------------------------------------------------------------------------
_C.LOG = CN()
_C.LOG.PATH = "runs/exp"

def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    #TODO:add config file
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.model_name:
        config.MODEL.NAME = args.model_name
    if args.dataset:
        config.DATA.DATASET = args.dataset
    if args.device:
        if args.device >= 0:
            config.DEVICE = "cuda:" + str(args.device)
        else:
            config.DEVICE = "cpu"
    if args.batch_size:
        config.TRAIN.BATCH_SIZE = args.batch_size
    if args.epochs:
        config.TRAIN.EPOCHS = args.epochs
    if args.cached_path:
        config.DATA.CACHED_PATH = args.cached_path
    if args.lr:
        config.TRAIN.LR = args.lr
    if args.model_path:
        config.MODEL.MODEL_PATH = args.model_path

    config.LOG.PATH = os.path.join(config.LOG.PATH,"-".join([config.DATA.DATASET,config.MODEL.NAME]))
    config.DATA.CACHED_PATH = os.path.join(config.DATA.CACHED_PATH,"-".join([config.DATA.DATASET,config.MODEL.NAME]))
    config.DATA.DATA_PATH = args.data_path
    config.freeze()


def get_config(args):
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
