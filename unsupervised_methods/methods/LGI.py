"""LGI
Local group invariance for heart rate estimation from face videos.
Pilz, C. S., Zaunseder, S., Krajewski, J. & Blazek, V.
In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, 1254–1262
(2018).
"""

import math

import numpy as np
from scipy import linalg
from scipy import signal
from unsupervised_methods import utils


def LGI(frames):
    precessed_data = utils.process_video(frames)
    u, _, _ = np.linalg.svd(precessed_data)
    s = u[:, :, 0]
    s = np.expand_dims(s, 2)
    sst = np.matmul(s, np.swapaxes(s, 1, 2))
    p = np.tile(np.identity(3), (s.shape[0], 1, 1))
    p = p - sst
    y = np.matmul(p, precessed_data)
    bvp = y[:, 1, :]
    bvp = bvp.reshape(-1)
    return bvp
