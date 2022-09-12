import numpy as np
import math
from scipy import signal
from scipy import linalg
from signal_methods import utils


def GREEN(frames):
    precessed_data = utils.process_video(frames)
    BVP = precessed_data[:, 1, :]
    BVP = BVP.reshape(-1)
    return BVP
