import cv2
import numpy as np
import math
from scipy import signal
from scipy import sparse
from scipy import linalg
from scipy import io as scio
from skimage.util import img_as_float
from sklearn.metrics import mean_squared_error

import signal_methods.utils as utils


def CHROME_DEHAAN(frames, bvps, FS, PlotTF):
    # Parameters
    SkinSegementTF = False
    LPF = 0.7
    HPF = 2.5
    WinSec = 1.6

    # Add backup function
    # Plot Control
    if (PlotTF):
        PlotPRPSD = True
        PlotSNR = True
    else:
        PlotPRPSD = False
        PlotSNR = False

    RGB = process_video(frames)
    FN = RGB.shape[0]
    NyquistF = 1/2*FS
    B, A = signal.butter(3, [LPF/NyquistF, HPF/NyquistF], 'bandpass')

    WinL = math.ceil(WinSec*FS)
    if(WinL % 2):
        WinL = WinL+1
    NWin = math.floor((FN-WinL//2)/(WinL//2))
    S = np.zeros((NWin, 1))
    WinS = 0
    WinM = int(WinS+WinL//2)
    WinE = WinS+WinL
    # todo:the total len
    totallen = (WinL//2)*(NWin+1)
    S = np.zeros(totallen)

    for i in range(NWin):
        RGBBase = np.mean(RGB[WinS:WinE, :], axis=0)
        # TODO bsxfun
        RGB_w = np.true_divide(1, RGBBase)
        RGBNorm = np.zeros((WinE-WinS, 3))
        for temp in range(WinS, WinE):
            RGBNorm[temp-WinS] = np.true_divide(RGB[temp], RGBBase)-1

        # RGBNorm =bsxfun(@times,RGB(WinS:WinE,:),1./RGBBase)-1;

        # CHROM
        Xs = np.squeeze(3*RGBNorm[:, 0]-2*RGBNorm[:, 1])
        Ys = np.squeeze(1.5*RGBNorm[:, 0]+RGBNorm[:, 1]-1.5*RGBNorm[:, 2])
        Xf = signal.filtfilt(B, A, Xs, axis=0)
        Yf = signal.filtfilt(B, A, Ys)

        Alpha = np.std(Xf) / np.std(Yf)
        SWin = Xf-Alpha*Yf
        SWin = np.multiply(SWin, signal.hanning(WinL))

        if(i == -1):
            S = SWin
        else:
            temp = SWin[:int(WinL//2)]
            S[WinS:WinM] = S[WinS:WinM] + SWin[:int(WinL//2)]
            S[WinM:WinE] = SWin[int(WinL//2):]
        WinS = WinM
        WinM = WinS+WinL//2
        WinE = WinS+WinL

    BVP = S
    # Evaluate
    # BVP_mat = scio.loadmat("BVP_ch.mat")["S"]
    # print(np.sqrt(mean_squared_error(BVP_mat,BVP)))

    # PR = utils.prpsd(BVP,FS,40,240,PlotPRPSD)
    # PR_0 = utils.prpsd(bvps,FS,40,240,PlotPRPSD)
    return BVP, 0, 0


def process_video(frames):
    RGB = []
    for frame in frames:
        sum = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(sum/(frame.shape[0]*frame.shape[1]))
    return np.asarray(RGB)
