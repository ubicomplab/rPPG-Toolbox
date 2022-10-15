# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']
# -----------------------------------------------------------------------------
# Train settings
# -----------------------------------------------------------------------------\
_C.TOOLBOX_MODE = ""
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
# Train.Data settings
_C.TRAIN.DATA = CN()
_C.TRAIN.DATA.FS = 0
_C.TRAIN.DATA.DATA_PATH = ''
_C.TRAIN.DATA.EXP_DATA_NAME = ''
_C.TRAIN.DATA.CACHED_PATH = 'PreprocessedData'
_C.TRAIN.DATA.FILE_LIST_PATH = os.path.join(_C.TRAIN.DATA.CACHED_PATH, 'DataFileLists')
_C.TRAIN.DATA.DATASET = ''
_C.TRAIN.DATA.DO_PREPROCESS = False
_C.TRAIN.DATA.DATA_FORMAT = 'NDCHW'
_C.TRAIN.DATA.BEGIN = 0.0
_C.TRAIN.DATA.END = 1.0
# Train Data preprocessing
_C.TRAIN.DATA.PREPROCESS = CN()
_C.TRAIN.DATA.PREPROCESS.DO_CHUNK = True
_C.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH = 180
_C.TRAIN.DATA.PREPROCESS.DYNAMIC_DETECTION = True
_C.TRAIN.DATA.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY  = 180
_C.TRAIN.DATA.PREPROCESS.CROP_FACE = True
_C.TRAIN.DATA.PREPROCESS.LARGE_FACE_BOX = True
_C.TRAIN.DATA.PREPROCESS.LARGE_BOX_COEF = 1.5
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
_C.VALID.DATA.DATA_PATH = ''
_C.VALID.DATA.EXP_DATA_NAME = ''
_C.VALID.DATA.CACHED_PATH = 'PreprocessedData'
_C.VALID.DATA.FILE_LIST_PATH = os.path.join(_C.VALID.DATA.CACHED_PATH, 'DataFileLists')
_C.VALID.DATA.DATASET = ''
_C.VALID.DATA.DO_PREPROCESS = False
_C.VALID.DATA.DATA_FORMAT = 'NDCHW'
_C.VALID.DATA.BEGIN = 0.0
_C.VALID.DATA.END = 1.0
# Valid Data preprocessing
_C.VALID.DATA.PREPROCESS = CN()
_C.VALID.DATA.PREPROCESS.DO_CHUNK = True
_C.VALID.DATA.PREPROCESS.CHUNK_LENGTH = 180
_C.VALID.DATA.PREPROCESS.DYNAMIC_DETECTION = True
_C.VALID.DATA.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY  = 180
_C.VALID.DATA.PREPROCESS.CROP_FACE = True
_C.VALID.DATA.PREPROCESS.LARGE_FACE_BOX = True
_C.VALID.DATA.PREPROCESS.LARGE_BOX_COEF = 1.5
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
_C.TEST.DATA.DATA_PATH = ''
_C.TEST.DATA.EXP_DATA_NAME = ''
_C.TEST.DATA.CACHED_PATH = 'PreprocessedData'
_C.TEST.DATA.FILE_LIST_PATH = os.path.join(_C.TEST.DATA.CACHED_PATH, 'DataFileLists')
_C.TEST.DATA.DATASET = ''
_C.TEST.DATA.DO_PREPROCESS = False
_C.TEST.DATA.DATA_FORMAT = 'NDCHW'
_C.TEST.DATA.BEGIN = 0.0
_C.TEST.DATA.END = 1.0
# Test Data preprocessing
_C.TEST.DATA.PREPROCESS = CN()
_C.TEST.DATA.PREPROCESS.DO_CHUNK = True
_C.TEST.DATA.PREPROCESS.CHUNK_LENGTH = 180
_C.TEST.DATA.PREPROCESS.DYNAMIC_DETECTION = True
_C.TEST.DATA.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY  = 180
_C.TEST.DATA.PREPROCESS.CROP_FACE = True
_C.TEST.DATA.PREPROCESS.LARGE_FACE_BOX = True
_C.TEST.DATA.PREPROCESS.LARGE_BOX_COEF = 1.5
_C.TEST.DATA.PREPROCESS.W = 128
_C.TEST.DATA.PREPROCESS.H = 128
_C.TEST.DATA.PREPROCESS.DATA_TYPE = ['']
_C.TEST.DATA.PREPROCESS.LABEL_TYPE = ''

# -----------------------------------------------------------------------------
# Signal method settings
# -----------------------------------------------------------------------------\
_C.SIGNAL = CN()
_C.SIGNAL.METHOD = []
_C.SIGNAL.METRICS = []
# Signal.Data settings
_C.SIGNAL.DATA = CN()
_C.SIGNAL.DATA.FS = 0
_C.SIGNAL.DATA.DATA_PATH = ''
_C.SIGNAL.DATA.EXP_DATA_NAME = ''
_C.SIGNAL.DATA.CACHED_PATH = 'PreprocessedData'
_C.SIGNAL.DATA.FILE_LIST_PATH = os.path.join(_C.SIGNAL.DATA.CACHED_PATH, 'DataFileLists')
_C.SIGNAL.DATA.DATASET = ''
_C.SIGNAL.DATA.DO_PREPROCESS = False
_C.SIGNAL.DATA.DATA_FORMAT = 'NDCHW'
_C.SIGNAL.DATA.BEGIN = 0.0
_C.SIGNAL.DATA.END = 1.0
# Signal Data preprocessing
_C.SIGNAL.DATA.PREPROCESS = CN()
_C.SIGNAL.DATA.PREPROCESS.DO_CHUNK = True
_C.SIGNAL.DATA.PREPROCESS.CHUNK_LENGTH = 180
_C.SIGNAL.DATA.PREPROCESS.DYNAMIC_DETECTION = True
_C.SIGNAL.DATA.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY  = 180
_C.SIGNAL.DATA.PREPROCESS.CROP_FACE = True
_C.SIGNAL.DATA.PREPROCESS.LARGE_FACE_BOX = True
_C.SIGNAL.DATA.PREPROCESS.LARGE_BOX_COEF = 1.5
_C.SIGNAL.DATA.PREPROCESS.W = 128
_C.SIGNAL.DATA.PREPROCESS.H = 128
_C.SIGNAL.DATA.PREPROCESS.DATA_TYPE = ['']
_C.SIGNAL.DATA.PREPROCESS.LABEL_TYPE = ''

### -----------------------------------------------------------------------------
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
_C.INFERENCE.EVALUATION_METHOD = 'FFT'
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

    # store default file list path for checking against later
    default_TRAIN_FILE_LIST_PATH = config.TRAIN.DATA.FILE_LIST_PATH
    default_VALID_FILE_LIST_PATH = config.VALID.DATA.FILE_LIST_PATH
    default_TEST_FILE_LIST_PATH = config.TEST.DATA.FILE_LIST_PATH
    default_SIGNAL_FILE_LIST_PATH = config.SIGNAL.DATA.FILE_LIST_PATH

    # update flag from config file
    _update_config_from_file(config, args.config_file)
    config.defrost()
    
    # UPDATE TRAIN PATHS
    if config.TRAIN.DATA.FILE_LIST_PATH == default_TRAIN_FILE_LIST_PATH:
        config.TRAIN.DATA.FILE_LIST_PATH = os.path.join(config.TRAIN.DATA.CACHED_PATH, 'DataFileLists')

    if config.TRAIN.DATA.EXP_DATA_NAME == '':
        config.TRAIN.DATA.EXP_DATA_NAME = "_".join([config.TRAIN.DATA.DATASET, "SizeW{0}".format(
            str(config.TRAIN.DATA.PREPROCESS.W)), "SizeH{0}".format(str(config.TRAIN.DATA.PREPROCESS.W)), "ClipLength{0}".format(
            str(config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH)), "DataType{0}".format("_".join(config.TRAIN.DATA.PREPROCESS.DATA_TYPE)),
                                      "LabelType{0}".format(config.TRAIN.DATA.PREPROCESS.LABEL_TYPE),
                                      "Large_box{0}".format(config.TRAIN.DATA.PREPROCESS.LARGE_FACE_BOX),
                                      "Large_size{0}".format(config.TRAIN.DATA.PREPROCESS.LARGE_BOX_COEF),
                                      "Dyamic_Det{0}".format(config.TRAIN.DATA.PREPROCESS.DYNAMIC_DETECTION),
                                        "det_len{0}".format(config.TRAIN.DATA.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY )
                                              ])
    config.TRAIN.DATA.CACHED_PATH = os.path.join(config.TRAIN.DATA.CACHED_PATH, config.TRAIN.DATA.EXP_DATA_NAME)

    name, ext = os.path.splitext(config.TRAIN.DATA.FILE_LIST_PATH)
    if not ext: # no file extension
        config.TRAIN.DATA.FILE_LIST_PATH = os.path.join(config.TRAIN.DATA.FILE_LIST_PATH, \
                                                        config.TRAIN.DATA.EXP_DATA_NAME + '_' + \
                                                        str(config.TRAIN.DATA.BEGIN) + '_' + \
                                                        str(config.TRAIN.DATA.END) + '.csv')
    elif ext != '.csv':
        raise ValueError(self.name, 'FILE_LIST_PATH must either be a directory path or a .csv file name')
    
    if ext == '.csv' and config.TRAIN.DATA.DO_PREPROCESS:
        raise ValueError(self.name, 'User specified FILE_LIST_PATH .csv file already exists. \
                         Please turn DO_PREPROCESS to False or delete existing FILE_LIST_PATH .csv file.')

    # UPDATE VALID PATHS
    if config.VALID.DATA.FILE_LIST_PATH == default_VALID_FILE_LIST_PATH:
        config.VALID.DATA.FILE_LIST_PATH = os.path.join(config.VALID.DATA.CACHED_PATH, 'DataFileLists')

    if config.VALID.DATA.EXP_DATA_NAME == '':
        config.VALID.DATA.EXP_DATA_NAME = "_".join([config.VALID.DATA.DATASET, "SizeW{0}".format(
            str(config.VALID.DATA.PREPROCESS.W)), "SizeH{0}".format(str(config.VALID.DATA.PREPROCESS.W)), "ClipLength{0}".format(
            str(config.VALID.DATA.PREPROCESS.CHUNK_LENGTH)), "DataType{0}".format("_".join(config.VALID.DATA.PREPROCESS.DATA_TYPE)),
                                      "LabelType{0}".format(config.VALID.DATA.PREPROCESS.LABEL_TYPE),
                                      "Large_box{0}".format(config.VALID.DATA.PREPROCESS.LARGE_FACE_BOX),
                                      "Large_size{0}".format(config.VALID.DATA.PREPROCESS.LARGE_BOX_COEF),
                                      "Dyamic_Det{0}".format(config.VALID.DATA.PREPROCESS.DYNAMIC_DETECTION),
                                        "det_len{0}".format(config.VALID.DATA.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY )
                                              ])
    config.VALID.DATA.CACHED_PATH = os.path.join(config.VALID.DATA.CACHED_PATH, config.VALID.DATA.EXP_DATA_NAME)

    name, ext = os.path.splitext(config.VALID.DATA.FILE_LIST_PATH)
    if not ext:  # no file extension
        config.VALID.DATA.FILE_LIST_PATH = os.path.join(config.VALID.DATA.FILE_LIST_PATH, \
                                                        config.VALID.DATA.EXP_DATA_NAME + '_' + \
                                                        str(config.VALID.DATA.BEGIN) + '_' + \
                                                        str(config.VALID.DATA.END) + '.csv')
    elif ext != '.csv':
        raise ValueError(self.name, 'FILE_LIST_PATH must either be a directory path or a .csv file name')

    if ext == '.csv' and config.VALID.DATA.DO_PREPROCESS:
        raise ValueError(self.name, 'User specified FILE_LIST_PATH .csv file already exists. \
                         Please turn DO_PREPROCESS to False or delete existing FILE_LIST_PATH .csv file.')

    # UPDATE TEST PATHS
    if config.TEST.DATA.FILE_LIST_PATH == default_TEST_FILE_LIST_PATH:
        config.TEST.DATA.FILE_LIST_PATH = os.path.join(config.TEST.DATA.CACHED_PATH, 'DataFileLists')

    if config.TEST.DATA.EXP_DATA_NAME == '':
        config.TEST.DATA.EXP_DATA_NAME = "_".join([config.TEST.DATA.DATASET, "SizeW{0}".format(
            str(config.TEST.DATA.PREPROCESS.W)), "SizeH{0}".format(str(config.TEST.DATA.PREPROCESS.W)), "ClipLength{0}".format(
            str(config.TEST.DATA.PREPROCESS.CHUNK_LENGTH)), "DataType{0}".format("_".join(config.TEST.DATA.PREPROCESS.DATA_TYPE)),
                                      "LabelType{0}".format(config.TEST.DATA.PREPROCESS.LABEL_TYPE),
                                      "Large_box{0}".format(config.TEST.DATA.PREPROCESS.LARGE_FACE_BOX),
                                      "Large_size{0}".format(config.TEST.DATA.PREPROCESS.LARGE_BOX_COEF),
                                      "Dyamic_Det{0}".format(config.TEST.DATA.PREPROCESS.DYNAMIC_DETECTION),
                                        "det_len{0}".format(config.TEST.DATA.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY )
                                              ])
    config.TEST.DATA.CACHED_PATH = os.path.join(config.TEST.DATA.CACHED_PATH, config.TEST.DATA.EXP_DATA_NAME)

    name, ext = os.path.splitext(config.TEST.DATA.FILE_LIST_PATH)
    if not ext: # no file extension
        config.TEST.DATA.FILE_LIST_PATH = os.path.join(config.TEST.DATA.FILE_LIST_PATH, \
                                                       config.TEST.DATA.EXP_DATA_NAME + '_' + \
                                                       str(config.TEST.DATA.BEGIN) + '_' + \
                                                       str(config.TEST.DATA.END) + '.csv')
    elif ext != '.csv':
        raise ValueError(self.name, 'FILE_LIST_PATH must either be a directory path or a .csv file name')

    if ext == '.csv' and config.TEST.DATA.DO_PREPROCESS:
        raise ValueError(self.name, 'User specified FILE_LIST_PATH .csv file already exists. \
                         Please turn DO_PREPROCESS to False or delete existing FILE_LIST_PATH .csv file.')
    

    # UPDATE SIGNAL PATHS
    if config.SIGNAL.DATA.FILE_LIST_PATH == default_SIGNAL_FILE_LIST_PATH:
        config.SIGNAL.DATA.FILE_LIST_PATH = os.path.join(config.SIGNAL.DATA.CACHED_PATH, 'DataFileLists')

    if config.SIGNAL.DATA.EXP_DATA_NAME == '':
        config.SIGNAL.DATA.EXP_DATA_NAME = "_".join([config.SIGNAL.DATA.DATASET, "SizeW{0}".format(
            str(config.SIGNAL.DATA.PREPROCESS.W)), "SizeH{0}".format(str(config.SIGNAL.DATA.PREPROCESS.W)), "ClipLength{0}".format(
            str(config.SIGNAL.DATA.PREPROCESS.CHUNK_LENGTH)), "DataType{0}".format("_".join(config.SIGNAL.DATA.PREPROCESS.DATA_TYPE)),
                                      "LabelType{0}".format(config.SIGNAL.DATA.PREPROCESS.LABEL_TYPE),
                                      "Large_box{0}".format(config.SIGNAL.DATA.PREPROCESS.LARGE_FACE_BOX),
                                      "Large_size{0}".format(config.SIGNAL.DATA.PREPROCESS.LARGE_BOX_COEF),
                                      "Dyamic_Det{0}".format(config.SIGNAL.DATA.PREPROCESS.DYNAMIC_DETECTION),
                                        "det_len{0}".format(config.SIGNAL.DATA.PREPROCESS.DYNAMIC_DETECTION_FREQUENCY),
                                        "signal"
                                              ])
    config.SIGNAL.DATA.CACHED_PATH = os.path.join(config.SIGNAL.DATA.CACHED_PATH, config.SIGNAL.DATA.EXP_DATA_NAME)

    name, ext = os.path.splitext(config.SIGNAL.DATA.FILE_LIST_PATH)
    if not ext: # no file extension
        config.SIGNAL.DATA.FILE_LIST_PATH = os.path.join(config.SIGNAL.DATA.FILE_LIST_PATH, \
                                                         config.SIGNAL.DATA.EXP_DATA_NAME + '_' + \
                                                         str(config.SIGNAL.DATA.BEGIN) + '_' + \
                                                         str(config.SIGNAL.DATA.END) + '.csv')
    elif ext != '.csv':
        raise ValueError(self.name, 'FILE_LIST_PATH must either be a directory path or a .csv file name')

    if ext == '.csv' and config.SIGNAL.DATA.DO_PREPROCESS:
        raise ValueError(self.name, 'User specified FILE_LIST_PATH .csv file already exists. \
                         Please turn DO_PREPROCESS to False or delete existing FILE_LIST_PATH .csv file.')


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


