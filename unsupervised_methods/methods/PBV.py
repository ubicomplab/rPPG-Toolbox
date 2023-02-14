"""PBV
Improved motion robustness of remote-ppg by using the blood volume pulse signature.
De Haan, G. & Van Leest, A.
Physiol. measurement 35, 1913 (2014)
"""

import math

import numpy as np
from scipy import linalg
from scipy import signal
from unsupervised_methods import utils


def PBV(frames):
    precessed_data = utils.process_video(frames)
    sig_mean = np.mean(precessed_data, axis=2)

    signal_norm_r = precessed_data[:, 0, :] / np.expand_dims(sig_mean[:, 0], axis=1)
    signal_norm_g = precessed_data[:, 1, :] / np.expand_dims(sig_mean[:, 1], axis=1)
    signal_norm_b = precessed_data[:, 2, :] / np.expand_dims(sig_mean[:, 2], axis=1)

    pbv_n = np.array([np.std(signal_norm_r, axis=1), np.std(signal_norm_g, axis=1), np.std(signal_norm_b, axis=1)])
    pbv_d = np.sqrt(np.var(signal_norm_r, axis=1) + np.var(signal_norm_g, axis=1) + np.var(signal_norm_b, axis=1))
    pbv = pbv_n / pbv_d

    c = np.swapaxes(np.array([signal_norm_r, signal_norm_g, signal_norm_b]), 0, 1)
    ct = np.swapaxes(np.swapaxes(np.transpose(c), 0, 2), 1, 2)
    q = np.matmul(c, ct)
    w = np.linalg.solve(q, np.swapaxes(pbv, 0, 1))

    a = np.matmul(ct, np.expand_dims(w, axis=2))
    b = np.matmul(np.swapaxes(np.expand_dims(pbv.T, axis=2), 1, 2), np.expand_dims(w, axis=2))
    bvp = a / b
    return bvp.squeeze(axis=2).reshape(-1)


def PBV2(frames):
    precessed_data = utils.process_video(frames)
    data_mean = np.mean(precessed_data, axis=2)
    r_norm = precessed_data[:, 0, :] / np.expand_dims(data_mean[:, 0], axis=1)
    g_norm = precessed_data[:, 1, :] / np.expand_dims(data_mean[:, 1], axis=1)
    b_norm = precessed_data[:, 2, :] / np.expand_dims(data_mean[:, 2], axis=1)
    rgb_array = np.array([r_norm, g_norm, b_norm])

    pbv_n = np.array([np.std(r_norm, axis=1), np.std(g_norm, axis=1), np.std(b_norm, axis=1)])
    pbv_d = np.sqrt(np.var(r_norm, axis=1) + np.var(g_norm, axis=1) + np.var(b_norm, axis=1))
    pbv = pbv_n / pbv_d
    c = np.transpose(rgb_array, (1, 0, 2))
    ct = np.transpose(rgb_array, (1, 2, 0))

    q = np.matmul(c, ct)
    w = np.linalg.solve(q, np.swapaxes(pbv, 0, 1))

    numerator = np.matmul(ct, np.expand_dims(w, axis=2))
    denominator = np.matmul(np.swapaxes(np.expand_dims(pbv.T, axis=2), 1, 2), np.expand_dims(w, axis=2))
    bvp = numerator / denominator
    bvp = bvp.squeeze(axis=2).reshape(-1)
    return bvp
