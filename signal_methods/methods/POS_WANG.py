# The Plane Orthogonal to Skin-Tone (POS) Method from: Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). 
# Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491. 
# DOI: 10.1109/TBME.2016.2609282
import numpy as np
import math
from scipy import signal
from signal_methods import utils
from metrics.metrics import *

def POS_WANG2(frames,fs):
    eps = 10 ** -9
    X = process_video2(frames)
    print(X.shape)
    e, c, f = X.shape  # e = #estimators, c = 3 rgb ch., f = #frames
    w = int(1.6 * fs)  # window length

    # stack e times fixed mat P
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    Q = np.stack([P for _ in range(e)], axis=0)

    # Initialize (1)
    H = np.zeros((e, f))
    for n in np.arange(w, f):
        # Start index of sliding window (4)
        m = n - w + 1
        # Temporal normalization (5)
        Cn = X[:, :, m:(n + 1)]
        M = 1.0 / (np.mean(Cn, axis=2) + eps)
        M = np.expand_dims(M, axis=2)  # shape [e, c, w]
        Cn = np.multiply(M, Cn)

        # Projection (6)
        S = np.dot(Q, Cn)
        S = S[0, :, :, :]
        S = np.swapaxes(S, 0, 1)  # remove 3-th dim

        # Tuning (7)
        S1 = S[:, 0, :]
        S2 = S[:, 1, :]
        alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
        alpha = np.expand_dims(alpha, axis=1)
        Hn = np.add(S1, alpha * S2)
        Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)
        # Overlap-adding (8)
        H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)

    return H.reshape(-1)



def POS_WANG(frames,fs):
    WinSec = 1.6
    RGB = process_video(frames)
    print(RGB.shape)
    N = RGB.shape[0]
    H = np.zeros((1, N))
    l = math.ceil(WinSec*fs)

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
    "Calculates the average value of each frame."
    RGB = []
    for frame in frames:
        sum = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(sum/(frame.shape[0]*frame.shape[1]))
    return np.asarray(RGB)

def process_video2(frames):
    # Standard:
    RGB = []
    for frame in frames:
        sum = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(sum/(frame.shape[0]*frame.shape[1]))
    RGB = np.asarray(RGB)
    RGB = RGB.transpose(1,0).reshape(1,3,-1)
    return np.asarray(RGB)