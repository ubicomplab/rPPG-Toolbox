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
# Train settings
# -----------------------------------------------------------------------------\
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
### Train.Data settings
_C.TRAIN.DATA = CN()
_C.TRAIN.DATA.FS = 0
# Path to dataset, could be overwritten by command line argument
_C.TRAIN.DATA.DATA_PATH = ''
_C.TRAIN.DATA.EXP_DATA_NAME = ''
# Path to preprocessing data, could be overwritten by command line argument
_C.TRAIN.DATA.CACHED_PATH = 'PreprocessedData'
# Dataset name, coule be overwritten by command line argument
_C.TRAIN.DATA.DATASET = ''
_C.TRAIN.DATA.DO_PREPROCESS = False
_C.TRAIN.DATA.DATA_FORMAT = 'NDCHW'
# -----------------------------------------------------------------------------
# Data preprocessing
# TODO: add other preprocessing configs
# -----------------------------------------------------------------------------
_C.TRAIN.DATA.PREPROCESS = CN()
_C.TRAIN.DATA.PREPROCESS.DO_CHUNK = True
_C.TRAIN.DATA.PREPROCESS.CLIP_LENGTH = 180
_C.TRAIN.DATA.PREPROCESS.DYNAMIC_DETECTION = True
_C.TRAIN.DATA.PREPROCESS.DETECTION_LENGTH = 180
_C.TRAIN.DATA.PREPROCESS.CROP_FACE = True
_C.TRAIN.DATA.PREPROCESS.LARGE_FACE_BOX = True
_C.TRAIN.DATA.PREPROCESS.LARGER_BOX_SIZE = 1.5
_C.TRAIN.DATA.PREPROCESS.W = 128
_C.TRAIN.DATA.PREPROCESS.H = 128
_C.TRAIN.DATA.PREPROCESS.DATA_TYPE = ['']
_C.TRAIN.DATA.PREPROCESS.LABEL_TYPE = ''

# -----------------------------------------------------------------------------
# Valid settings
# -----------------------------------------------------------------------------\
_C.VALID = CN()
# Valid.Data settings
_C.VALID.DATA = CN()
_C.VALID.DATA.FS = 0
# Path to dataset, could be overwritten by command line argument
_C.VALID.DATA.DATA_PATH = ''
_C.VALID.DATA.EXP_DATA_NAME = ''
# Path to preprocessing data, could be overwritten by command line argument
_C.VALID.DATA.CACHED_PATH = 'PreprocessedData'
# Dataset name, coule be overwritten by command line argument
_C.VALID.DATA.DATASET = ''
_C.VALID.DATA.DO_PREPROCESS = False
_C.VALID.DATA.DATA_FORMAT = 'NDCHW'
# Valid Data preprocessing
_C.VALID.DATA.PREPROCESS = CN()
_C.VALID.DATA.PREPROCESS.DO_CHUNK = True
_C.VALID.DATA.PREPROCESS.CLIP_LENGTH = 180
_C.VALID.DATA.PREPROCESS.DYNAMIC_DETECTION = True
_C.VALID.DATA.PREPROCESS.DETECTION_LENGTH = 180
_C.VALID.DATA.PREPROCESS.CROP_FACE = True
_C.VALID.DATA.PREPROCESS.LARGE_FACE_BOX = True
_C.VALID.DATA.PREPROCESS.LARGER_BOX_SIZE = 1.5
_C.VALID.DATA.PREPROCESS.W = 128
_C.VALID.DATA.PREPROCESS.H = 128
_C.VALID.DATA.PREPROCESS.DATA_TYPE = ['']
_C.VALID.DATA.PREPROCESS.LABEL_TYPE = ''


# -----------------------------------------------------------------------------
# Test settings
# -----------------------------------------------------------------------------\
_C.TEST = CN()
_C.TEST.METRICS = []
# Test.Data settings
_C.TEST.DATA = CN()
_C.TEST.DATA.FS = 0
# Path to dataset, could be overwritten by command line argument
_C.TEST.DATA.DATA_PATH = ''
_C.TEST.DATA.EXP_DATA_NAME = ''
# Path to preprocessing data, could be overwritten by command line argument
_C.TEST.DATA.CACHED_PATH = 'PreprocessedData'
# Dataset name, coule be overwritten by command line argument
_C.TEST.DATA.DATASET = ''
_C.TEST.DATA.DO_PREPROCESS = False
_C.TEST.DATA.DATA_FORMAT = 'NDCHW'
# Valid Data preprocessing
_C.TEST.DATA.PREPROCESS = CN()
_C.TEST.DATA.PREPROCESS.DO_CHUNK = True
_C.TEST.DATA.PREPROCESS.CLIP_LENGTH = 180
_C.TEST.DATA.PREPROCESS.DYNAMIC_DETECTION = True
_C.TEST.DATA.PREPROCESS.DETECTION_LENGTH = 180
_C.TEST.DATA.PREPROCESS.CROP_FACE = True
_C.TEST.DATA.PREPROCESS.LARGE_FACE_BOX = True
_C.TEST.DATA.PREPROCESS.LARGER_BOX_SIZE = 1.5
_C.TEST.DATA.PREPROCESS.W = 128
_C.TEST.DATA.PREPROCESS.H = 128
_C.TEST.DATA.PREPROCESS.DATA_TYPE = ['']
_C.TEST.DATA.PREPROCESS.LABEL_TYPE = ''


#
# # -----------------------------------------------------------------------------
# # Data settings
# # -----------------------------------------------------------------------------
# _C.DATA = CN()
# _C.DATA.FS = 0
# # Path to dataset, could be overwritten by command line argument
# _C.DATA.TRAIN_DATA_PATH = ''
# _C.DATA.VALID_DATA_PATH = ''
# _C.DATA.TEST_DATA_PATH = ''
# _C.DATA.EXP_DATA_NAME = ''
# # Path to preprocessing data, could be overwritten by command line argument
# _C.DATA.CACHED_PATH = 'PreprocessedData'
# # Dataset name, coule be overwritten by command line argument
# _C.DATA.DATASET = ''
# _C.DATA.DO_PREPROCESS = False
# _C.DATA.DATA_FORMAT = 'NDCHW'
# # -----------------------------------------------------------------------------
# # Data preprocessing
# # TODO: add other preprocessing configs
# # -----------------------------------------------------------------------------
# _C.DATA.PREPROCESS = CN()
# _C.DATA.PREPROCESS.DO_CHUNK = True
# _C.DATA.PREPROCESS.CLIP_LENGTH = 180
# _C.DATA.PREPROCESS.DYNAMIC_DETECTION = True
# _C.DATA.PREPROCESS.DETECTION_LENGTH = 180
# _C.DATA.PREPROCESS.CROP_FACE = True
# _C.DATA.PREPROCESS.LARGE_FACE_BOX = True
# _C.DATA.PREPROCESS.LARGER_BOX_SIZE = 1.5
# _C.DATA.PREPROCESS.W = 128
# _C.DATA.PREPROCESS.H = 128
# _C.DATA.PREPROCESS.DATA_TYPE = ['']
# _C.DATA.PREPROCESS.LABEL_TYPE = ''
#
# # -----------------------------------------------------------------------------
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
# Inference settings
# -----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.BATCH_SIZE = 4
_C.INFERENCE.MODEL_PATH = ''

# -----------------------------------------------------------------------------
# Device settings
# -----------------------------------------------------------------------------
_C.DEVICE = "cuda:0"
_C.NUM_OF_GPU_TRAIN = 1

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
    print('=> Merging a config file from {}'.format(cfg_file))
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
    # if args.cached_path:
    #     config.DATA.CACHED_PATH = args.cached_path
    if args.lr:
        config.TRAIN.LR = args.lr
    if args.model_dir:
        config.MODEL.MODEL_DIR = args.model_dir
    # if args.preprocess:
    #     config.DATA.DO_PREPROCESS = args.preprocess
    if args.train_data_path:
        config.TRAIN.DATA.DATA_PATH = args.train_data_path
    if args.valid_data_path:
        config.VALID.DATA.DATA_PATH = args.valid_data_path

    if config.TRAIN.DATA.EXP_DATA_NAME == '':
        config.TRAIN.DATA.EXP_DATA_NAME = "_".join([config.TRAIN.DATA.DATASET, "SizeW{0}".format(
            str(config.TRAIN.DATA.PREPROCESS.W)), "SizeH{0}".format(str(config.TRAIN.DATA.PREPROCESS.W)), "ClipLength{0}".format(
            str(config.TRAIN.DATA.PREPROCESS.CLIP_LENGTH)), "DataType{0}".format("_".join(config.TRAIN.DATA.PREPROCESS.DATA_TYPE)),
                                      "LabelType{0}".format(config.TRAIN.DATA.PREPROCESS.LABEL_TYPE),
                                      "Large_box{0}".format(config.TRAIN.DATA.PREPROCESS.LARGE_FACE_BOX),
                                      "Large_size{0}".format(config.TRAIN.DATA.PREPROCESS.LARGER_BOX_SIZE),
                                      "Dyamic_Det{0}".format(config.TRAIN.DATA.PREPROCESS.DYNAMIC_DETECTION),
                                        "det_len{0}".format(config.TRAIN.DATA.PREPROCESS.DETECTION_LENGTH)
                                              ])
    config.TRAIN.DATA.CACHED_PATH = os.path.join(config.TRAIN.DATA.CACHED_PATH, config.TRAIN.DATA.EXP_DATA_NAME)

    if config.VALID.DATA.EXP_DATA_NAME == '':
        config.VALID.DATA.EXP_DATA_NAME = "_".join([config.VALID.DATA.DATASET, "SizeW{0}".format(
            str(config.VALID.DATA.PREPROCESS.W)), "SizeH{0}".format(str(config.VALID.DATA.PREPROCESS.W)), "ClipLength{0}".format(
            str(config.VALID.DATA.PREPROCESS.CLIP_LENGTH)), "DataType{0}".format("_".join(config.VALID.DATA.PREPROCESS.DATA_TYPE)),
                                      "LabelType{0}".format(config.VALID.DATA.PREPROCESS.LABEL_TYPE),
                                      "Large_box{0}".format(config.VALID.DATA.PREPROCESS.LARGE_FACE_BOX),
                                      "Large_size{0}".format(config.VALID.DATA.PREPROCESS.LARGER_BOX_SIZE),
                                      "Dyamic_Det{0}".format(config.VALID.DATA.PREPROCESS.DYNAMIC_DETECTION),
                                        "det_len{0}".format(config.VALID.DATA.PREPROCESS.DETECTION_LENGTH)
                                              ])
    config.VALID.DATA.CACHED_PATH = os.path.join(config.VALID.DATA.CACHED_PATH, config.VALID.DATA.EXP_DATA_NAME)

    if config.TEST.DATA.EXP_DATA_NAME == '':
        config.TEST.DATA.EXP_DATA_NAME = "_".join([config.TEST.DATA.DATASET, "SizeW{0}".format(
            str(config.TEST.DATA.PREPROCESS.W)), "SizeH{0}".format(str(config.TEST.DATA.PREPROCESS.W)), "ClipLength{0}".format(
            str(config.TEST.DATA.PREPROCESS.CLIP_LENGTH)), "DataType{0}".format("_".join(config.TEST.DATA.PREPROCESS.DATA_TYPE)),
                                      "LabelType{0}".format(config.TEST.DATA.PREPROCESS.LABEL_TYPE),
                                      "Large_box{0}".format(config.TEST.DATA.PREPROCESS.LARGE_FACE_BOX),
                                      "Large_size{0}".format(config.TEST.DATA.PREPROCESS.LARGER_BOX_SIZE),
                                      "Dyamic_Det{0}".format(config.TEST.DATA.PREPROCESS.DYNAMIC_DETECTION),
                                        "det_len{0}".format(config.TEST.DATA.PREPROCESS.DETECTION_LENGTH)
                                              ])
    config.TEST.DATA.CACHED_PATH = os.path.join(config.TEST.DATA.CACHED_PATH, config.TEST.DATA.EXP_DATA_NAME)

    config.LOG.PATH = os.path.join(
        config.LOG.PATH, config.VALID.DATA.EXP_DATA_NAME)

    config.MODEL.MODEL_DIR = os.path.join(config.MODEL.MODEL_DIR, config.TRAIN.DATA.EXP_DATA_NAME)
    config.freeze()
    return



def get_config(args):
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config


