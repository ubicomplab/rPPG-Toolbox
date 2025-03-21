# General utils

import glob
import logging
import math
import os
import random
import re
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import yaml

from dataset.data_loader.face_detector.utils.torch_utils import init_torch_seeds

# Settings
torch.set_printoptions(linewidth=320, precision=5, profile='long')
# format short g, %precision=5
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})
# prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
cv2.setNumThreads(0)
os.environ['NUMEXPR_MAX_THREADS'] = str(
    min(os.cpu_count(), 8))  # NumExpr max threads


def set_logging(rank=-1):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def check_file(file):
    # Search for file if not found
    if os.path.isfile(file) or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        assert len(files) == 1, "Multiple files match '%s', specify exact path: %s" % (
            file, files)  # assert unique
        return files[0]  # return file
