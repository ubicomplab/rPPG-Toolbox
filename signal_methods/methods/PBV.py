import numpy as np
import math
from scipy import signal
from scipy import linalg
from signal_methods import utils


def PBV(frames):
    precessed_data = utils.process_video(frames)
    data_mean = np.mean(precessed_data, axis=2)
    R_norm = precessed_data[:, 0, :] / np.expand_dims(data_mean[:, 0], axis=1)
    G_norm = precessed_data[:, 1, :] / np.expand_dims(data_mean[:, 1], axis=1)
    B_norm = precessed_data[:, 2, :] / np.expand_dims(data_mean[:, 2], axis=1)
    RGB_array = np.array([R_norm, G_norm, B_norm])

    PBV_n = np.array([np.std(R_norm, axis=1), np.std(G_norm, axis=1), np.std(B_norm, axis=1)])
    PBV_d = np.sqrt(np.var(R_norm, axis=1) + np.var(G_norm, axis=1) + np.var(B_norm, axis=1))
    PBV = PBV_n / PBV_d
    C = np.transpose(RGB_array, (1, 0, 2))
    Ct = np.transpose(RGB_array, (1, 2, 0))

    Q = np.matmul(C, Ct)
    W = np.linalg.solve(Q, np.swapaxes(PBV, 0, 1))
        
    Numerator = np.matmul(Ct, np.expand_dims(W, axis=2))
    Denominator  = np.matmul(np.swapaxes(np.expand_dims(PBV.T, axis=2), 1, 2), np.expand_dims(W, axis=2))
    BVP = Numerator / Denominator
    BVP = BVP.squeeze(axis=2).reshape(-1)
    return BVP
