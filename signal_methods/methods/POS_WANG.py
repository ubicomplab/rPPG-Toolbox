import numpy as np
import math
from scipy import signal
from signal_methods import utils
from metrics.metrics import *


def POS_WANG(frames,fs):
    LPF = 0.7
    HPF = 2.5
    WinSec = 1.6
    RGB = process_video(frames)
    N = RGB.shape[0]
    H = np.zeros((1, N))
    l = math.ceil(WinSec*fs)
    C = np.zeros((1, 3))

    for n in range(N):
        m = n-l
        if(m >= 0):
            Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
            Cn = np.mat(Cn).H
            S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
            h = S[0, :]+(np.std(S[0, :])/np.std(S[1, :]))*S[1, :]
            mean_h = np.mean(h)
            for temp in range(h.shape[1]):
                h[0, temp] = h[0, temp]-mean_h
            H[0, m:n] = H[0, m:n]+(h[0])

    BVP = H
    BVP = utils.detrend(np.mat(BVP).H, 100)
    BVP = np.asarray(np.transpose(BVP))[0]
    B, A = signal.butter(1, [0.75/fs*2, 3/fs*2], 'bandpass')
    BVP = signal.filtfilt(B, A, BVP.astype(np.double))
    return BVP

def process_video(frames):
    RGB = []
    for frame in frames:
        sum = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(sum/(frame.shape[0]*frame.shape[1]))
    return np.asarray(RGB)