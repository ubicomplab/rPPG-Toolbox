# The Chrominance Method from: De Haan, G., & Jeanne, V. (2013). 
# Robust pulse rate from chrominance-based rPPG. IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886. 
# DOI: 10.1109/TBME.2013.2266196
import numpy as np
import math
from scipy import signal

import unsupervised_methods.utils as utils


def Chrome(frames,FS):
    LPF = 0.7
    HPF = 2.5
    WinSec = 1.6
    NyquistF = 1/2*FS

    rgb = _process_video(frames)
    fn = rgb.shape[0]
    b, a = signal.butter(3, [LPF/NyquistF, HPF/NyquistF], 'bandpass')
    win_l = math.ceil(WinSec*FS)
    if(win_l % 2):
        win_l = win_l+1
    n_win = math.floor((fn-win_l//2)/(win_l//2))
    s = np.zeros((n_win, 1))
    win_s = 0
    win_m = int(win_s+win_l//2)
    win_e = win_s+win_l
    total_len = (win_l//2)*(n_win+1)
    s = np.zeros(total_len)

    for i in range(n_win):
        rgb_base = np.mean(rgb[win_s:win_e, :], axis=0)
        rgb_norm = np.zeros((win_e-win_s, 3))
        for temp in range(win_s, win_e):
            rgb_norm[temp-win_s] = np.true_divide(rgb[temp], rgb_base)-1
        xs = np.squeeze(3*rgb_norm[:, 0]-2*rgb_norm[:, 1])
        ys = np.squeeze(1.5*rgb_norm[:, 0]+rgb_norm[:, 1]-1.5*rgb_norm[:, 2])
        xf = signal.filtfilt(b, a, xs, axis=0)
        yf = signal.filtfilt(b, a, ys)
        alpha = np.std(xf) / np.std(yf)
        s_win = xf-alpha*yf
        s_win = np.multiply(s_win, signal.hanning(win_l))
        if(i == -1):
            s = s_win
        else:
            temp = s_win[:int(win_l//2)]
            s[win_s:win_m] = s[win_s:win_m] + s_win[:int(win_l//2)]
            s[win_m:win_e] = s_win[int(win_l//2):]
        win_s = win_m
        win_m = win_s+win_l//2
        win_e = win_s+win_l
    bvp = s
    return bvp

def _process_video(frames):
    "Calculates the average value of each frame."
    rgb = []
    for frame in frames:
        sum = np.sum(np.sum(frame, axis=0), axis=0)
        rgb.append(sum/(frame.shape[0]*frame.shape[1]))
    return np.asarray(rgb)
