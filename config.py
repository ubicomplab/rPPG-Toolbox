# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os, re
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
_C.TRAIN.PLOT_LOSSES_AND_LR = True
# Train.Data settings
_C.TRAIN.DATA = CN()
_C.TRAIN.DATA.INFO = CN()
_C.TRAIN.DATA.INFO.LIGHT = ['']
_C.TRAIN.DATA.INFO.MOTION = ['']
_C.TRAIN.DATA.INFO.EXERCISE = [True]
_C.TRAIN.DATA.INFO.SKIN_COLOR = [1]
_C.TRAIN.DATA.INFO.GENDER = ['']
_C.TRAIN.DATA.INFO.GLASSER = [True]
_C.TRAIN.DATA.INFO.HAIR_COVER = [True]
_C.TRAIN.DATA.INFO.MAKEUP = [True]
_C.TRAIN.DATA.FILTERING = CN()
_C.TRAIN.DATA.FILTERING.USE_EXCLUSION_LIST = False
_C.TRAIN.DATA.FILTERING.EXCLUSION_LIST = ['']
_C.TRAIN.DATA.FILTERING.SELECT_TASKS = False
_C.TRAIN.DATA.FILTERING.TASK_LIST = ['']
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
_C.TRAIN.DATA.FOLD = CN()
_C.TRAIN.DATA.FOLD.FOLD_NAME = ''
_C.TRAIN.DATA.FOLD.FOLD_PATH = ''
# Train Data preprocessing
_C.TRAIN.DATA.PREPROCESS = CN()
_C.TRAIN.DATA.PREPROCESS.USE_PSUEDO_PPG_LABEL = False
_C.TRAIN.DATA.PREPROCESS.DATA_TYPE = ['']
_C.TRAIN.DATA.PREPROCESS.DATA_AUG = ['None']
_C.TRAIN.DATA.PREPROCESS.LABEL_TYPE = ''
_C.TRAIN.DATA.PREPROCESS.DO_CHUNK = True
_C.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH = 180
_C.TRAIN.DATA.PREPROCESS.CROP_FACE = CN()
_C.TRAIN.DATA.PREPROCESS.CROP_FACE.DO_CROP_FACE = True
_C.TRAIN.DATA.PREPROCESS.CROP_FACE.BACKEND = 'HC'
_C.TRAIN.DATA.PREPROCESS.CROP_FACE.USE_LARGE_FACE_BOX = True
_C.TRAIN.DATA.PREPROCESS.CROP_FACE.LARGE_BOX_COEF = 1.5
_C.TRAIN.DATA.PREPROCESS.CROP_FACE.DETECTION = CN()
_C.TRAIN.DATA.PREPROCESS.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION = False
_C.TRAIN.DATA.PREPROCESS.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY = 30
_C.TRAIN.DATA.PREPROCESS.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX = False
_C.TRAIN.DATA.PREPROCESS.ROI = CN()
_C.TRAIN.DATA.PREPROCESS.ROI.EXTRACT_ROI = False
_C.TRAIN.DATA.PREPROCESS.ROI.LANDMARKS = []
_C.TRAIN.DATA.PREPROCESS.RESIZE = CN()
_C.TRAIN.DATA.PREPROCESS.RESIZE.W = 128
_C.TRAIN.DATA.PREPROCESS.RESIZE.H = 128
_C.TRAIN.DATA.PREPROCESS.BIGSMALL = CN()
_C.TRAIN.DATA.PREPROCESS.BIGSMALL.BIG_DATA_TYPE = ['']
_C.TRAIN.DATA.PREPROCESS.BIGSMALL.SMALL_DATA_TYPE = ['']
_C.TRAIN.DATA.PREPROCESS.BIGSMALL.RESIZE = CN()
_C.TRAIN.DATA.PREPROCESS.BIGSMALL.RESIZE.BIG_W = 144
_C.TRAIN.DATA.PREPROCESS.BIGSMALL.RESIZE.BIG_H = 144
_C.TRAIN.DATA.PREPROCESS.BIGSMALL.RESIZE.SMALL_W = 9
_C.TRAIN.DATA.PREPROCESS.BIGSMALL.RESIZE.SMALL_H = 9
# -----------------------------------------------------------------------------
# Valid settings
# -----------------------------------------------------------------------------\
_C.VALID = CN()
# Valid.Data settings
_C.VALID.DATA = CN()
_C.VALID.DATA.INFO = CN()
_C.VALID.DATA.INFO.LIGHT = ['']
_C.VALID.DATA.INFO.MOTION = ['']
_C.VALID.DATA.INFO.EXERCISE = [True]
_C.VALID.DATA.INFO.SKIN_COLOR = [1]
_C.VALID.DATA.INFO.GENDER = ['']
_C.VALID.DATA.INFO.GLASSER = [True]
_C.VALID.DATA.INFO.HAIR_COVER = [True]
_C.VALID.DATA.INFO.MAKEUP = [True]
_C.VALID.DATA.FILTERING = CN()
_C.VALID.DATA.FILTERING.USE_EXCLUSION_LIST = False
_C.VALID.DATA.FILTERING.EXCLUSION_LIST = ['']
_C.VALID.DATA.FILTERING.SELECT_TASKS = False
_C.VALID.DATA.FILTERING.TASK_LIST = ['']
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
_C.VALID.DATA.FOLD = CN()
_C.VALID.DATA.FOLD.FOLD_NAME = ''
_C.VALID.DATA.FOLD.FOLD_PATH = ''
# Valid Data preprocessing
_C.VALID.DATA.PREPROCESS = CN()
_C.VALID.DATA.PREPROCESS.USE_PSUEDO_PPG_LABEL = False
_C.VALID.DATA.PREPROCESS.DATA_TYPE = ['']
_C.VALID.DATA.PREPROCESS.DATA_AUG = ['None']
_C.VALID.DATA.PREPROCESS.LABEL_TYPE = ''
_C.VALID.DATA.PREPROCESS.DO_CHUNK = True
_C.VALID.DATA.PREPROCESS.CHUNK_LENGTH = 180
_C.VALID.DATA.PREPROCESS.ROI = CN()
_C.VALID.DATA.PREPROCESS.ROI.EXTRACT_ROI = False
_C.VALID.DATA.PREPROCESS.ROI.LANDMARKS = []
_C.VALID.DATA.PREPROCESS.CROP_FACE = CN()
_C.VALID.DATA.PREPROCESS.CROP_FACE.DO_CROP_FACE = True
_C.VALID.DATA.PREPROCESS.CROP_FACE.BACKEND = 'HC'
_C.VALID.DATA.PREPROCESS.CROP_FACE.USE_LARGE_FACE_BOX = True
_C.VALID.DATA.PREPROCESS.CROP_FACE.LARGE_BOX_COEF = 1.5
_C.VALID.DATA.PREPROCESS.CROP_FACE.DETECTION = CN()
_C.VALID.DATA.PREPROCESS.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION = False
_C.VALID.DATA.PREPROCESS.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY = 30
_C.VALID.DATA.PREPROCESS.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX = False
_C.VALID.DATA.PREPROCESS.RESIZE = CN()
_C.VALID.DATA.PREPROCESS.RESIZE.W = 128
_C.VALID.DATA.PREPROCESS.RESIZE.H = 128
_C.VALID.DATA.PREPROCESS.BIGSMALL = CN()
_C.VALID.DATA.PREPROCESS.BIGSMALL.BIG_DATA_TYPE = ['']
_C.VALID.DATA.PREPROCESS.BIGSMALL.SMALL_DATA_TYPE = ['']
_C.VALID.DATA.PREPROCESS.BIGSMALL.RESIZE = CN()
_C.VALID.DATA.PREPROCESS.BIGSMALL.RESIZE.BIG_W = 144
_C.VALID.DATA.PREPROCESS.BIGSMALL.RESIZE.BIG_H = 144
_C.VALID.DATA.PREPROCESS.BIGSMALL.RESIZE.SMALL_W = 9
_C.VALID.DATA.PREPROCESS.BIGSMALL.RESIZE.SMALL_H = 9

# -----------------------------------------------------------------------------
# Test settings
# -----------------------------------------------------------------------------\
_C.TEST = CN()
_C.TEST.OUTPUT_SAVE_DIR = ''
_C.TEST.METRICS = []
_C.TEST.USE_LAST_EPOCH = True
# Test.Data settings
_C.TEST.DATA = CN()
_C.TEST.DATA.INFO = CN()
_C.TEST.DATA.INFO.LIGHT = ['']
_C.TEST.DATA.INFO.MOTION = ['']
_C.TEST.DATA.INFO.EXERCISE = [True]
_C.TEST.DATA.INFO.SKIN_COLOR = [1]
_C.TEST.DATA.INFO.GENDER = ['']
_C.TEST.DATA.INFO.GLASSER = [True]
_C.TEST.DATA.INFO.HAIR_COVER = [True]
_C.TEST.DATA.INFO.MAKEUP = [True]
_C.TEST.DATA.FILTERING = CN()
_C.TEST.DATA.FILTERING.USE_EXCLUSION_LIST = False
_C.TEST.DATA.FILTERING.EXCLUSION_LIST = ['']
_C.TEST.DATA.FILTERING.SELECT_TASKS = False
_C.TEST.DATA.FILTERING.TASK_LIST = ['']
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
_C.TEST.DATA.FOLD = CN()
_C.TEST.DATA.FOLD.FOLD_NAME = ''
_C.TEST.DATA.FOLD.FOLD_PATH = ''
# Test Data preprocessing
_C.TEST.DATA.PREPROCESS = CN()
_C.TEST.DATA.PREPROCESS.USE_PSUEDO_PPG_LABEL = False
_C.TEST.DATA.PREPROCESS.DATA_TYPE = ['']
_C.TEST.DATA.PREPROCESS.DATA_AUG = ['None']
_C.TEST.DATA.PREPROCESS.LABEL_TYPE = ''
_C.TEST.DATA.PREPROCESS.DO_CHUNK = True
_C.TEST.DATA.PREPROCESS.CHUNK_LENGTH = 180
_C.TEST.DATA.PREPROCESS.ROI = CN()
_C.TEST.DATA.PREPROCESS.ROI.EXTRACT_ROI = False
_C.TEST.DATA.PREPROCESS.ROI.LANDMARKS = []
_C.TEST.DATA.PREPROCESS.CROP_FACE = CN()
_C.TEST.DATA.PREPROCESS.CROP_FACE.DO_CROP_FACE = True
_C.TEST.DATA.PREPROCESS.CROP_FACE.BACKEND = 'HC'
_C.TEST.DATA.PREPROCESS.CROP_FACE.USE_LARGE_FACE_BOX = True
_C.TEST.DATA.PREPROCESS.CROP_FACE.LARGE_BOX_COEF = 1.5
_C.TEST.DATA.PREPROCESS.CROP_FACE.DETECTION = CN()
_C.TEST.DATA.PREPROCESS.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION = False
_C.TEST.DATA.PREPROCESS.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY = 30
_C.TEST.DATA.PREPROCESS.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX = False
_C.TEST.DATA.PREPROCESS.RESIZE = CN()
_C.TEST.DATA.PREPROCESS.RESIZE.W = 128
_C.TEST.DATA.PREPROCESS.RESIZE.H = 128
_C.TEST.DATA.PREPROCESS.BIGSMALL = CN()
_C.TEST.DATA.PREPROCESS.BIGSMALL.BIG_DATA_TYPE = ['']
_C.TEST.DATA.PREPROCESS.BIGSMALL.SMALL_DATA_TYPE = ['']
_C.TEST.DATA.PREPROCESS.BIGSMALL.RESIZE = CN()
_C.TEST.DATA.PREPROCESS.BIGSMALL.RESIZE.BIG_W = 144
_C.TEST.DATA.PREPROCESS.BIGSMALL.RESIZE.BIG_H = 144
_C.TEST.DATA.PREPROCESS.BIGSMALL.RESIZE.SMALL_W = 9
_C.TEST.DATA.PREPROCESS.BIGSMALL.RESIZE.SMALL_H = 9

# -----------------------------------------------------------------------------
# Unsupervised method settings
# -----------------------------------------------------------------------------\
_C.UNSUPERVISED = CN()
_C.UNSUPERVISED.METHOD = []
_C.UNSUPERVISED.OUTPUT_SAVE_DIR = ''
_C.UNSUPERVISED.METRICS = []
# Unsupervised.Data settings
_C.UNSUPERVISED.DATA = CN()
_C.UNSUPERVISED.DATA.INFO = CN()
_C.UNSUPERVISED.DATA.INFO.LIGHT = ['']
_C.UNSUPERVISED.DATA.INFO.MOTION = ['']
_C.UNSUPERVISED.DATA.INFO.EXERCISE = [True]
_C.UNSUPERVISED.DATA.INFO.SKIN_COLOR = [1]
_C.UNSUPERVISED.DATA.INFO.GENDER = ['']
_C.UNSUPERVISED.DATA.INFO.GLASSER = [True]
_C.UNSUPERVISED.DATA.INFO.HAIR_COVER = [True]
_C.UNSUPERVISED.DATA.INFO.MAKEUP = [True]
_C.UNSUPERVISED.DATA.FILTERING = CN()
_C.UNSUPERVISED.DATA.FILTERING.USE_EXCLUSION_LIST = False
_C.UNSUPERVISED.DATA.FILTERING.EXCLUSION_LIST = ['']
_C.UNSUPERVISED.DATA.FILTERING.SELECT_TASKS = False
_C.UNSUPERVISED.DATA.FILTERING.TASK_LIST = ['']
_C.UNSUPERVISED.DATA.FS = 0
_C.UNSUPERVISED.DATA.DATA_PATH = ''
_C.UNSUPERVISED.DATA.EXP_DATA_NAME = ''
_C.UNSUPERVISED.DATA.CACHED_PATH = 'PreprocessedData'
_C.UNSUPERVISED.DATA.FILE_LIST_PATH = os.path.join(_C.UNSUPERVISED.DATA.CACHED_PATH, 'DataFileLists')
_C.UNSUPERVISED.DATA.DATASET = ''
_C.UNSUPERVISED.DATA.DO_PREPROCESS = False
_C.UNSUPERVISED.DATA.DATA_FORMAT = 'NDCHW'
_C.UNSUPERVISED.DATA.BEGIN = 0.0
_C.UNSUPERVISED.DATA.END = 1.0
_C.UNSUPERVISED.DATA.FOLD = CN()
_C.UNSUPERVISED.DATA.FOLD.FOLD_NAME = ''
_C.UNSUPERVISED.DATA.FOLD.FOLD_PATH = ''
# Unsupervised Data preprocessing
_C.UNSUPERVISED.DATA.PREPROCESS = CN()
_C.UNSUPERVISED.DATA.PREPROCESS.USE_PSUEDO_PPG_LABEL = False
_C.UNSUPERVISED.DATA.PREPROCESS.DATA_TYPE = ['']
_C.UNSUPERVISED.DATA.PREPROCESS.DATA_AUG = ['None']
_C.UNSUPERVISED.DATA.PREPROCESS.LABEL_TYPE = ''
_C.UNSUPERVISED.DATA.PREPROCESS.DO_CHUNK = True
_C.UNSUPERVISED.DATA.PREPROCESS.CHUNK_LENGTH = 180
_C.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE = CN()
_C.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.DO_CROP_FACE = True
_C.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.BACKEND = 'HC'
_C.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.USE_LARGE_FACE_BOX = True
_C.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.LARGE_BOX_COEF = 1.5
_C.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.DETECTION = CN()
_C.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION = False
_C.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY = 30
_C.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX = False
_C.UNSUPERVISED.DATA.PREPROCESS.ROI = CN()
_C.UNSUPERVISED.DATA.PREPROCESS.ROI.EXTRACT_ROI = False
_C.UNSUPERVISED.DATA.PREPROCESS.ROI.LANDMARKS = []
_C.UNSUPERVISED.DATA.PREPROCESS.RESIZE = CN()
_C.UNSUPERVISED.DATA.PREPROCESS.RESIZE.W = 128
_C.UNSUPERVISED.DATA.PREPROCESS.RESIZE.H = 128

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
# Model Settings for BigSmall
# -----------------------------------------------------------------------------
_C.MODEL.BIGSMALL = CN()
_C.MODEL.BIGSMALL.FRAME_DEPTH = 3

# -----------------------------------------------------------------------------
# Model Settings for PhysFormer
# -----------------------------------------------------------------------------
_C.MODEL.PHYSFORMER = CN()
_C.MODEL.PHYSFORMER.PATCH_SIZE = 4
_C.MODEL.PHYSFORMER.DIM = 96
_C.MODEL.PHYSFORMER.FF_DIM = 144
_C.MODEL.PHYSFORMER.NUM_HEADS = 4
_C.MODEL.PHYSFORMER.NUM_LAYERS = 12
_C.MODEL.PHYSFORMER.THETA = 0.7

# -----------------------------------------------------------------------------
# Inference settings
# -----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.BATCH_SIZE = 4
_C.INFERENCE.EVALUATION_METHOD = 'FFT'
_C.INFERENCE.EVALUATION_WINDOW = CN()
_C.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW = False
_C.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE = 10
_C.INFERENCE.EVALUATION_WINDOW.WINDOW_OVERLAP = 0
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
    # print('=> Merging a config file from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):

    # store default file list path for checking against later
    default_TRAIN_FILE_LIST_PATH = config.TRAIN.DATA.FILE_LIST_PATH
    default_VALID_FILE_LIST_PATH = config.VALID.DATA.FILE_LIST_PATH
    default_TEST_FILE_LIST_PATH = config.TEST.DATA.FILE_LIST_PATH
    default_UNSUPERVISED_FILE_LIST_PATH = config.UNSUPERVISED.DATA.FILE_LIST_PATH

    # update flag from config file
    _update_config_from_file(config, args.config_file)
    config.defrost()
    
    # UPDATE TRAIN PATHS
    if config.TRAIN.DATA.FILE_LIST_PATH == default_TRAIN_FILE_LIST_PATH:
        config.TRAIN.DATA.FILE_LIST_PATH = os.path.join(config.TRAIN.DATA.CACHED_PATH, 'DataFileLists')

    if config.TRAIN.DATA.EXP_DATA_NAME == '':
        config.TRAIN.DATA.EXP_DATA_NAME = "_".join([config.TRAIN.DATA.DATASET, "SizeW{0}".format(
            str(config.TRAIN.DATA.PREPROCESS.RESIZE.W)), "SizeH{0}".format(str(config.TRAIN.DATA.PREPROCESS.RESIZE.W)), "ClipLength{0}".format(
            str(config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH)), "DataType{0}".format("_".join(config.TRAIN.DATA.PREPROCESS.DATA_TYPE)),
                                      "DataAug{0}".format("_".join(config.TRAIN.DATA.PREPROCESS.DATA_AUG)),
                                      "LabelType{0}".format(config.TRAIN.DATA.PREPROCESS.LABEL_TYPE),
                                      "Crop_face{0}".format(config.TRAIN.DATA.PREPROCESS.CROP_FACE.DO_CROP_FACE),
                                      "Backend{0}".format(config.TRAIN.DATA.PREPROCESS.CROP_FACE.BACKEND),
                                      "Large_box{0}".format(config.TRAIN.DATA.PREPROCESS.CROP_FACE.USE_LARGE_FACE_BOX),
                                      "Large_size{0}".format(config.TRAIN.DATA.PREPROCESS.CROP_FACE.LARGE_BOX_COEF),
                                      "Dyamic_Det{0}".format(config.TRAIN.DATA.PREPROCESS.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION),
                                        "det_len{0}".format(config.TRAIN.DATA.PREPROCESS.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY),
                                        "Median_face_box{0}".format(config.TRAIN.DATA.PREPROCESS.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX)
                                              ])
    config.TRAIN.DATA.CACHED_PATH = os.path.join(config.TRAIN.DATA.CACHED_PATH, config.TRAIN.DATA.EXP_DATA_NAME)

    name, ext = os.path.splitext(config.TRAIN.DATA.FILE_LIST_PATH)
    if not ext: # no file extension
        FOLD_STR = '_' + config.TRAIN.DATA.FOLD.FOLD_NAME if config.TRAIN.DATA.FOLD.FOLD_NAME else ''
        config.TRAIN.DATA.FILE_LIST_PATH = os.path.join(config.TRAIN.DATA.FILE_LIST_PATH, \
                                                        config.TRAIN.DATA.EXP_DATA_NAME + '_' + \
                                                        str(config.TRAIN.DATA.BEGIN) + '_' + \
                                                        str(config.TRAIN.DATA.END) + \
                                                        FOLD_STR + '.csv')
    elif ext != '.csv':
        raise ValueError('TRAIN dataset FILE_LIST_PATH must either be a directory path or a .csv file name')
    
    if ext == '.csv' and config.TRAIN.DATA.DO_PREPROCESS:
        raise ValueError('User specified TRAIN dataset FILE_LIST_PATH .csv file already exists. \
                         Please turn DO_PREPROCESS to False or delete existing TRAIN dataset FILE_LIST_PATH .csv file.')

    if not config.TEST.USE_LAST_EPOCH and config.VALID.DATA.DATASET is not None:
        # UPDATE VALID PATHS
        if config.VALID.DATA.FILE_LIST_PATH == default_VALID_FILE_LIST_PATH:
            config.VALID.DATA.FILE_LIST_PATH = os.path.join(config.VALID.DATA.CACHED_PATH, 'DataFileLists')

        if config.VALID.DATA.EXP_DATA_NAME == '':
            config.VALID.DATA.EXP_DATA_NAME = "_".join([config.VALID.DATA.DATASET, "SizeW{0}".format(
                str(config.VALID.DATA.PREPROCESS.RESIZE.W)), "SizeH{0}".format(str(config.VALID.DATA.PREPROCESS.RESIZE.W)), "ClipLength{0}".format(
                str(config.VALID.DATA.PREPROCESS.CHUNK_LENGTH)), "DataType{0}".format("_".join(config.VALID.DATA.PREPROCESS.DATA_TYPE)),
                                        "DataAug{0}".format("_".join(config.VALID.DATA.PREPROCESS.DATA_AUG)),
                                        "LabelType{0}".format(config.VALID.DATA.PREPROCESS.LABEL_TYPE),
                                        "Crop_face{0}".format(config.VALID.DATA.PREPROCESS.CROP_FACE.DO_CROP_FACE),
                                        "Backend{0}".format(config.VALID.DATA.PREPROCESS.CROP_FACE.BACKEND),
                                        "Large_box{0}".format(config.VALID.DATA.PREPROCESS.CROP_FACE.USE_LARGE_FACE_BOX),
                                        "Large_size{0}".format(config.VALID.DATA.PREPROCESS.CROP_FACE.LARGE_BOX_COEF),
                                        "Dyamic_Det{0}".format(config.VALID.DATA.PREPROCESS.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION),
                                          "det_len{0}".format(config.VALID.DATA.PREPROCESS.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY),
                                          "Median_face_box{0}".format(config.VALID.DATA.PREPROCESS.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX)
                                                ])
        config.VALID.DATA.CACHED_PATH = os.path.join(config.VALID.DATA.CACHED_PATH, config.VALID.DATA.EXP_DATA_NAME)

        name, ext = os.path.splitext(config.VALID.DATA.FILE_LIST_PATH)
        if not ext:  # no file extension
            FOLD_STR = '_' + config.VALID.DATA.FOLD.FOLD_NAME if config.VALID.DATA.FOLD.FOLD_NAME else ''
            config.VALID.DATA.FILE_LIST_PATH = os.path.join(config.VALID.DATA.FILE_LIST_PATH, \
                                                            config.VALID.DATA.EXP_DATA_NAME + '_' + \
                                                            str(config.VALID.DATA.BEGIN) + '_' + \
                                                            str(config.VALID.DATA.END) + \
                                                            FOLD_STR + '.csv')
        elif ext != '.csv':
            raise ValueError('VALIDATION dataset FILE_LIST_PATH must either be a directory path or a .csv file name')

        if ext == '.csv' and config.VALID.DATA.DO_PREPROCESS:
            raise ValueError('User specified VALIDATION dataset FILE_LIST_PATH .csv file already exists. \
                            Please turn DO_PREPROCESS to False or delete existing VALIDATION dataset FILE_LIST_PATH .csv file.')
    elif not config.TEST.USE_LAST_EPOCH and config.VALID.DATA.DATASET is None:
        raise ValueError('VALIDATION dataset is not provided despite USE_LAST_EPOCH being False!')

    # UPDATE TEST PATHS
    if config.TEST.DATA.FILE_LIST_PATH == default_TEST_FILE_LIST_PATH:
        config.TEST.DATA.FILE_LIST_PATH = os.path.join(config.TEST.DATA.CACHED_PATH, 'DataFileLists')

    if config.TEST.DATA.EXP_DATA_NAME == '':
        config.TEST.DATA.EXP_DATA_NAME = "_".join([config.TEST.DATA.DATASET, "SizeW{0}".format(
            str(config.TEST.DATA.PREPROCESS.RESIZE.W)), "SizeH{0}".format(str(config.TEST.DATA.PREPROCESS.RESIZE.H)), "ClipLength{0}".format(
            str(config.TEST.DATA.PREPROCESS.CHUNK_LENGTH)), "DataType{0}".format("_".join(config.TEST.DATA.PREPROCESS.DATA_TYPE)),
                                      "DataAug{0}".format("_".join(config.TEST.DATA.PREPROCESS.DATA_AUG)),
                                      "LabelType{0}".format(config.TEST.DATA.PREPROCESS.LABEL_TYPE),
                                      "Crop_face{0}".format(config.TEST.DATA.PREPROCESS.CROP_FACE.DO_CROP_FACE),
                                      "Backend{0}".format(config.TEST.DATA.PREPROCESS.CROP_FACE.BACKEND),
                                      "Large_box{0}".format(config.TEST.DATA.PREPROCESS.CROP_FACE.USE_LARGE_FACE_BOX),
                                      "Large_size{0}".format(config.TEST.DATA.PREPROCESS.CROP_FACE.LARGE_BOX_COEF),
                                      "Dyamic_Det{0}".format(config.TEST.DATA.PREPROCESS.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION),
                                        "det_len{0}".format(config.TEST.DATA.PREPROCESS.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY),
                                        "Median_face_box{0}".format(config.TEST.DATA.PREPROCESS.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX)
                                              ])
    config.TEST.DATA.CACHED_PATH = os.path.join(config.TEST.DATA.CACHED_PATH, config.TEST.DATA.EXP_DATA_NAME)

    name, ext = os.path.splitext(config.TEST.DATA.FILE_LIST_PATH)
    if not ext: # no file extension
        FOLD_STR = '_' + config.TEST.DATA.FOLD.FOLD_NAME if config.TEST.DATA.FOLD.FOLD_NAME else ''
        config.TEST.DATA.FILE_LIST_PATH = os.path.join(config.TEST.DATA.FILE_LIST_PATH, \
                                                       config.TEST.DATA.EXP_DATA_NAME + '_' + \
                                                       str(config.TEST.DATA.BEGIN) + '_' + \
                                                       str(config.TEST.DATA.END) + \
                                                       FOLD_STR + '.csv')
    elif ext != '.csv':
        raise ValueError('TEST dataset FILE_LIST_PATH must either be a directory path or a .csv file name')

    if ext == '.csv' and config.TEST.DATA.DO_PREPROCESS:
        raise ValueError('User specified TEST dataset FILE_LIST_PATH .csv file already exists. \
                         Please turn DO_PREPROCESS to False or delete existing TEST dataset FILE_LIST_PATH .csv file.')
    

    # UPDATE MODEL_FILE_NAME IF NEEDED
    if any(aug != 'None' for aug in config.TRAIN.DATA.PREPROCESS.DATA_AUG + config.VALID.DATA.PREPROCESS.DATA_AUG + config.TEST.DATA.PREPROCESS.DATA_AUG):
        # Check if the initial MODEL_FILE_NAME follows the expected pattern
        if re.match(r'^[^_]+(_[^_]+)?(_[^_]+)?_[^_]+$', config.TRAIN.MODEL_FILE_NAME):
            model_file_name_parts = config.TRAIN.MODEL_FILE_NAME.split('_')
            if model_file_name_parts[2] == config.TEST.DATA.DATASET:
                train_name_idx = 0
                valid_name_idx = 1
                test_name_idx = 2
            else:
                train_name_idx = 0
                valid_name_idx = None
                test_name_idx = 1
            if 'Motion' in config.TRAIN.DATA.PREPROCESS.DATA_AUG:
                model_file_name_parts = config.TRAIN.MODEL_FILE_NAME.split('_')
                model_file_name_parts[train_name_idx] = 'MA-' + model_file_name_parts[train_name_idx]
                config.TRAIN.MODEL_FILE_NAME = '_'.join(model_file_name_parts)
            if 'Motion' in config.VALID.DATA.PREPROCESS.DATA_AUG:
                model_file_name_parts = config.TRAIN.MODEL_FILE_NAME.split('_')
                model_file_name_parts[valid_name_idx] = 'MA-' + model_file_name_parts[valid_name_idx]
                config.TRAIN.MODEL_FILE_NAME = '_'.join(model_file_name_parts)
            if 'Motion' in config.TEST.DATA.PREPROCESS.DATA_AUG:
                model_file_name_parts = config.TRAIN.MODEL_FILE_NAME.split('_')
                model_file_name_parts[test_name_idx] = 'MA-' + model_file_name_parts[test_name_idx]
                config.TRAIN.MODEL_FILE_NAME = '_'.join(model_file_name_parts)
        else:
            raise ValueError(f'MODEL_FILE_NAME does not follow expected naming pattern of [TRAIN_SET]_[VALID_SET]_[TEST_SET]! \
                             \nReceived {config.TRAIN.MODEL_FILE_NAME}.')

    # ENSURE USE_PSEUDO_LABELS IS NOT TRUE FOR UNSUPERVISED METHODS
    if config.TOOLBOX_MODE == 'unsupervised_method' and config.UNSUPERVISED.DATA.PREPROCESS.USE_PSUEDO_PPG_LABEL == True:
        raise ValueError('Pseudo PPG labels are NOT supported for unsupervised methods.')

    # UPDATE UNSUPERVISED PATHS
    if config.UNSUPERVISED.DATA.FILE_LIST_PATH == default_UNSUPERVISED_FILE_LIST_PATH:
        config.UNSUPERVISED.DATA.FILE_LIST_PATH = os.path.join(config.UNSUPERVISED.DATA.CACHED_PATH, 'DataFileLists')

    if config.UNSUPERVISED.DATA.EXP_DATA_NAME == '':
        config.UNSUPERVISED.DATA.EXP_DATA_NAME = "_".join([config.UNSUPERVISED.DATA.DATASET, "SizeW{0}".format(
            str(config.UNSUPERVISED.DATA.PREPROCESS.RESIZE.W)), "SizeH{0}".format(str(config.UNSUPERVISED.DATA.PREPROCESS.RESIZE.W)), "ClipLength{0}".format(
            str(config.UNSUPERVISED.DATA.PREPROCESS.CHUNK_LENGTH)), "DataType{0}".format("_".join(config.UNSUPERVISED.DATA.PREPROCESS.DATA_TYPE)),
                                      "DataAug{0}".format("_".join(config.UNSUPERVISED.DATA.PREPROCESS.DATA_AUG)),
                                      "LabelType{0}".format(config.UNSUPERVISED.DATA.PREPROCESS.LABEL_TYPE),
                                      "Crop_face{0}".format(config.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.DO_CROP_FACE),
                                      "Backend{0}".format(config.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.BACKEND),
                                      "Large_box{0}".format(config.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.USE_LARGE_FACE_BOX),
                                      "Large_size{0}".format(config.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.LARGE_BOX_COEF),
                                      "Dyamic_Det{0}".format(config.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION),
                                        "det_len{0}".format(config.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY),
                                        "Median_face_box{0}".format(config.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX),
                                        "unsupervised"
                                              ])
    config.UNSUPERVISED.DATA.CACHED_PATH = os.path.join(config.UNSUPERVISED.DATA.CACHED_PATH, config.UNSUPERVISED.DATA.EXP_DATA_NAME)

    name, ext = os.path.splitext(config.UNSUPERVISED.DATA.FILE_LIST_PATH)
    if not ext: # no file extension
        FOLD_STR = '_' + config.UNSUPERVISED.DATA.FOLD.FOLD_NAME if config.UNSUPERVISED.DATA.FOLD.FOLD_NAME else ''
        config.UNSUPERVISED.DATA.FILE_LIST_PATH = os.path.join(config.UNSUPERVISED.DATA.FILE_LIST_PATH, \
                                                        config.UNSUPERVISED.DATA.EXP_DATA_NAME + '_' + \
                                                        str(config.UNSUPERVISED.DATA.BEGIN) + '_' + \
                                                        str(config.UNSUPERVISED.DATA.END) + \
                                                        FOLD_STR + '.csv')
    elif ext != '.csv':
        raise ValueError('UNSUPERVISED dataset FILE_LIST_PATH must either be a directory path or a .csv file name')

    if ext == '.csv' and config.UNSUPERVISED.DATA.DO_PREPROCESS:
        raise ValueError('User specified UNSUPERVISED dataset FILE_LIST_PATH .csv file already exists. \
                         Please turn DO_PREPROCESS to False or delete existing UNSUPERVISED dataset FILE_LIST_PATH .csv file.')

    # Establish the directory to hold pre-trained models from a given experiment inside 
    # the configured log directory (runs/exp by default)
    config.MODEL.MODEL_DIR = os.path.join(config.LOG.PATH, config.TRAIN.DATA.EXP_DATA_NAME, config.MODEL.MODEL_DIR)

    # Establish the directory to hold outputs saved during testing inside the
    # configured log directory (runs/exp by default)
    if config.TOOLBOX_MODE == 'train_and_test' or config.TOOLBOX_MODE == 'only_test':
        config.TEST.OUTPUT_SAVE_DIR = os.path.join(config.LOG.PATH, config.TEST.DATA.EXP_DATA_NAME, 'saved_test_outputs')
    elif config.TOOLBOX_MODE == 'unsupervised_method':
        config.UNSUPERVISED.OUTPUT_SAVE_DIR = os.path.join(config.LOG.PATH, config.UNSUPERVISED.DATA.EXP_DATA_NAME, 'saved_outputs')
    else:
        raise ValueError('TOOLBOX_MODE only supports train_and_test, only_test, or unsupervised_method!')

    config.freeze()
    return

def get_config(args):
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config


