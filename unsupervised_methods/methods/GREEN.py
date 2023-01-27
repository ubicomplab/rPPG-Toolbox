""" GREEN
Verkruysse, W., Svaasand, L. O. & Nelson, J. S.
Remote plethysmographic imaging using ambient light.
Optical. Express 16, 21434–21445 (2008).
"""

import numpy as np
import math
from scipy import signal
from scipy import linalg
from unsupervised_methods import utils


def GREEN(frames):
    precessed_data = utils.process_video(frames)
    BVP = precessed_data[:, 1, :]
    BVP = BVP.reshape(-1)
    return BVP
