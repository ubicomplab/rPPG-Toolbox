"""POS
Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). 
Algorithmic principles of remote PPG. 
IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491. 
"""

import math

import numpy as np
from scipy import signal
from unsupervised_methods import utils


def _process_video(frames):
    """Calculates the average value of each frame."""
    rgb = []
    for frame in frames:
        summation = np.sum(np.sum(frame, axis=0), axis=0)
        rgb.append(summation / (frame.shape[0] * frame.shape[1]))
    return np.asarray(rgb)


def Pos(frames, FS):
    WinSec = 1.6
    
    rgb = _process_video(frames)
    n = rgb.shape[0]
    h = np.zeros((1, n))
    l = math.ceil(WinSec * FS)

    for n in range(n):
        m = n - l
        if m >= 0:
            c_n = np.true_divide(rgb[m:n, :], np.mean(rgb[m:n, :], axis=0))
            c_n = np.mat(c_n).H
            s = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), c_n)
            h = s[0, :] + (np.std(s[0, :]) / np.std(s[1, :])) * s[1, :]
            mean_h = np.mean(h)
            for temp in range(h.shape[1]):
                h[0, temp] = h[0, temp] - mean_h
            h[0, m:n] = h[0, m:n] + (h[0])

    bvp = h
    bvp = utils.detrend(np.mat(bvp).H, 100)
    bvp = np.asarray(np.transpose(bvp))[0]
    b, a = signal.butter(1, [0.75 / FS * 2, 3 / FS * 2], btype='bandpass')
    bvp = signal.filtfilt(b, a, bvp.astype(np.double))
    return bvp


