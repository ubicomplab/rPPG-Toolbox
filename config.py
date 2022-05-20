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

_C.TRAIL_NAME = ''

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.FS = 0
# Path to dataset, could be overwritten by command line argument
_C.DATA.TRAIN_DATA_PATH = ''
_C.DATA.VALID_DATA_PATH = ''
_C.DATA.TEST_DATA_PATH = ''
# Path to preprocessing data, could be overwritten by command line argument
_C.DATA.CACHED_PATH = 'PreprocessedData'
# Dataset name, coule be overwritten by command line argument
_C.DATA.DATASET = ''
_C.DATA.DO_PREPROCESS = False
_C.DATA.DATA_FORMAT = 'NDCHW'
# -----------------------------------------------------------------------------
# Data preprocessing
# TODO: add other preprocessing configs
# -----------------------------------------------------------------------------
_C.DATA.PREPROCESS = CN()
_C.DATA.PREPROCESS.DO_CHUNK = True
_C.DATA.PREPROCESS.CROP_FACE = True
_C.DATA.PREPROCESS.LARGE_FACE_BOX = True
_C.DATA.PREPROCESS.FACE_DETECT = True
_C.DATA.PREPROCESS.W = 128
_C.DATA.PREPROCESS.H = 128
_C.DATA.PREPROCESS.CLIP_LENGTH = 64
_C.DATA.PREPROCESS.DATA_TYPE = ['']
_C.DATA.PREPROCESS.LABEL_TYPE = ''

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model name
_C.MODEL.NAME = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
_C.MODEL.MODEL_DIR = 'PreTrainedModels'

# Specific parameters for physnet parameters
_C.MODEL.PHYSNET = CN()
_C.MODEL.PHYSNET.FRAME_NUM = 64

# -----------------------------------------------------------------------------
# Model Settings for TS-CAN
# -----------------------------------------------------------------------------
_C.MODEL.TSCAN = CN()
_C.MODEL.TSCAN.FRAME_DEPTH = 10

# -----------------------------------------------------------------------------
# Model Settings for EfficientPhys
# -----------------------------------------------------------------------------
_C.MODEL.EFFICIENTPHYS = CN()
_C.MODEL.EFFICIENTPHYS.FRAME_DEPTH = 10

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
_C.TRAIN.MODEL_FILE_NAME = ''

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.METRICS = []

# -----------------------------------------------------------------------------
# Inference settings
# -----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.BATCH_SIZE = 4
_C.INFERENCE.MODEL_PATH = ''


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
    _update_config_from_file(config, args.config_file)

    config.defrost()
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
    if args.model_dir:
        config.MODEL.MODEL_DIR = args.model_dir
    if args.preprocess:
        config.DATA.DO_PREPROCESS = args.preprocess
    if args.train_data_path:
        config.DATA.TRAIN_DATA_PATH = args.train_data_path
    if args.valid_data_path:
        config.DATA.VALID_DATA_PATH = args.valid_data_path

    if config.TRAIL_NAME == '':
        config.TRAIL_NAME= "_".join([config.DATA.DATASET, config.MODEL.NAME, "SizeW{0}".format(
        str(config.DATA.PREPROCESS.W)), "SizeH{0}".format(str(config.DATA.PREPROCESS.W)), "ClipLength{0}".format(
        str(config.DATA.PREPROCESS.CLIP_LENGTH)), "DataType{0}".format("_".join(config.DATA.PREPROCESS.DATA_TYPE)), "LabelType{0}".format(config.DATA.PREPROCESS.LABEL_TYPE)])

    config.LOG.PATH = os.path.join(
        config.LOG.PATH, config.TRAIL_NAME)
    config.DATA.CACHED_PATH = os.path.join(
        config.DATA.CACHED_PATH, config.TRAIL_NAME)
    config.MODEL.MODEL_DIR = os.path.join(config.MODEL.MODEL_DIR, config.TRAIL_NAME)
    config.freeze()
    return


def update_evaluate_config(config, args):

    _update_config_from_file(config, args.config_file)
    config.defrost()

    if args.device:
        if args.device >= 0:
            config.DEVICE = "cuda:" + str(args.device)
        else:
            config.DEVICE = "cpu"
    if args.batch_size:
        config.TRAIN.BATCH_SIZE = args.batch_size
    if args.cached_path:
        config.DATA.CACHED_PATH = args.cached_path
    if args.preprocess:
        config.DATA.DO_PREPROCESS = args.preprocess
    if args.test_data_path:
        config.DATA.TEST_DATA_PATH = args.test_data_path
    if args.model_path:
        config.INFERENCE_MODEL_PATH = args.model_path
    config.LOG.PATH = os.path.join(
        config.LOG.PATH, config.TRAIL_NAME)
    config.DATA.CACHED_PATH = os.path.join(
        config.DATA.CACHED_PATH, config.TRAIL_NAME)

    config.freeze()
    return


def get_config(args):
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config


def get_evaluate_config(args):
    config = _C.clone()
    update_evaluate_config(config, args)

    return config
