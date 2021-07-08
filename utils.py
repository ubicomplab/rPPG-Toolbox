import cv2
import numpy as np
import math
from scipy import signal
from scipy import sparse
from scipy import linalg
from scipy import io as scio
from skimage.util import img_as_float
from sklearn.metrics import mean_squared_error

def detrend(signal, Lambda):
    """detrend(signal, Lambda) -> filtered_signal
    This function applies a detrending filter.
    This code is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
    *Parameters*
      ``signal`` (1d numpy array):
        The signal where you want to remove the trend.
      ``Lambda`` (int):
        The smoothing parameter.
    *Returns*
      ``filtered_signal`` (1d numpy array):
        The detrended signal.
    """
    signal_length = signal.shape[0]

    # observation matrix
    H = np.identity(signal_length)

    # second-order difference matrix

    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal


def bvpsnr(BVP, FS, HR, PlotTF):
    '''

    :param BVP:
    :param FS:
    :param HR:
    :param PlotTF:
    :return:
    '''
    HR_F = HR / 60
    NyquistF = FS / 2
    FResBPM = 0.5
    N = (60 * 2 * NyquistF) / FResBPM

    ##Construct Periodogram
    F, Pxx = signal.periodogram(BVP, FS, nfft=N, window="hamming")
    GTMask1 = (F >= HR_F - 0.1) & (F <= HR_F + 0.1)
    GTMask2 = (F >= (HR_F * 2 - 0.2)) & (F <= (HR_F * 2 + 0.2))
    temp = GTMask1 | GTMask2
    SPower = np.sum(Pxx[temp])
    FMask2 = (F >= 0.5) & (F <= 4)
    AllPower = np.sum(Pxx[FMask2])
    SNR = 10 * math.log10(SPower / (AllPower - SPower))
    print("SignalNoiseRatio", SNR)
    return SNR

def prpsd(BVP,FS,LL_PR,UL_PR,PlotTF):
    """

    :param BVP:
    :param FS:
    :param LL_PR:
    :param UL_PR:
    :param PlotTF:
    :return:
    """

    Nyquist = FS/2
    FResBPM = 0.5
    N = (60*2*Nyquist)/FResBPM
    #TODO: here is different too:https://stackoverflow.com/questions/17130795/periodogram-in-octave-matlab-vs-scipy
    F,Pxx = signal.periodogram(BVP,FS,nfft=N,window="hamming")
    FMask = ((F >= (LL_PR/60)) & (F <= (UL_PR/60)))
    #Calculate predicted HR
    FRange = F[FMask]
    PRange = Pxx[FMask]
    MaxInd = np.argmax(Pxx[FMask],0)
    PR_F = FRange[MaxInd]
    PR = PR_F*60
    print("PulseRate:",PR)

    return PR

def parse_ECG(ECGFile,StartTime,Duration):
    ECG = scio.loadmat(ECGFile)["ECG"]
    ECG_data = ECG["data"][0,0]
    ECG_fs = ECG["fs"][0,0]
    ECG_peaks = ECG["peaks"][0,0]

    ECG_time = (np.arange(0,ECG_data.shape[0]))/ECG_fs
    ECGMask = (ECG_time >= StartTime) & (ECG_data <= (StartTime+Duration))
    ECGPeakMask = ((ECG_peaks / ECG_fs) >= StartTime) & ((ECG_peaks / ECG_fs) <= (StartTime+Duration))
    diff = np.diff(ECG_peaks[ECGPeakMask]/ECG_fs)
    HR_ECG = (1/np.mean(diff))*60
    print("HR_ECG",HR_ECG)
    return HR_ECG

def parse_PPG(PPGFile,StartTime,Duration):
    try:
        PPG = scio.loadmat(PPGFile)["PPG"]
    except:
        print("Error: Can't load PPG File.")
        return -1
    PPG_data = PPG["data"][0,0]
    PPG_fs = PPG["fs"][0,0]
    try:
        PPG_peaks = PPG["peaks"][0,0]
    except:
        print("Error: Can't find peaks field in PPG File.")
        return -1
    PPG_time = (np.arange(0,PPG_data.shape[0]))/PPG_fs
    PPGMask = (PPG_time >= StartTime) & (PPG_data <= (StartTime+Duration))

    PPGPeakMask = ((PPG_peaks / PPG_fs) >= StartTime) & ((PPG_peaks / PPG_fs) <= (StartTime+Duration))
    diff = np.diff(PPG_peaks[PPGPeakMask]/PPG_fs)
    PR_PPG = (1/np.mean(diff))*60
    print("PR_PPG",PR_PPG)
    return PR_PPG