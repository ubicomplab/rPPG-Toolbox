# You can simply run this file to get the hrs for subject1
# TODO:(overall)
# Get 3 average hr on all subjects. Report when 3 results are quite different.
# You can modify the code as you wish, for example by wrapping it in a
# function, so that it can automatically measure all the data and save the
# results in the way you define.

# UBFC datasets:https://drive.google.com/drive/folders/1Iu7mdh9oTbsX5YbQjHfzrFOAjn-AdnZG?usp=sharing


import os
from matplotlib import pyplot as plt
from scipy.sparse import spdiags
from scipy.signal import butter
import scipy.io
import scipy
import numpy as np

# TODO:
# you can config the file imformation here
SUBJECT = "subject1"
FILE_NAME = "ground_truth.txt"
# TODO ENDS

# 对于UBFC
# %% Import


# %% Helper Function
def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def read_bvp():
    with open(SUBJECT + os.sep + FILE_NAME, "r") as f:
        str1 = f.read()
        str1 = str1.split("\n")
        bvp = str1[0]
        bvp = [float(x) for x in bvp.split()]
        hr = str1[1]
        hr = [float(x) for x in hr.split()]
        times = str1[2]
        times = [float(x) for x in times.split()]
        # str1 =  [ float(x) for x in str1 if x != ""]
        fs = len(times) / times[-1]
        # TODO: if fs isn't around 30, please report
    return bvp, hr, times,fs


def detrend(signal, Lambda):
    signal_length = signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(
        diags_data,
        diags_index,
        (signal_length - 2),
        signal_length).toarray()
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal


def mag2db(mag):
    return 20. * np.log10(mag)


def calculate_HR(
        pxx_pred,
        frange_pred,
        fmask_pred,
        pxx_label,
        frange_label,
        fmask_label):
    pred_HR = np.take(
        frange_pred,
        np.argmax(
            np.take(
                pxx_pred,
                fmask_pred),
            0))[0] * 60
    ground_truth_HR = np.take(
        frange_label,
        np.argmax(
            np.take(
                pxx_label,
                fmask_label),
            0))[0] * 60
    return pred_HR, ground_truth_HR


def calculate_SNR(pxx_pred, f_pred, currHR, signal):
    currHR = currHR / 60
    f = f_pred
    pxx = pxx_pred
    gtmask1 = (f >= currHR - 0.1) & (f <= currHR + 0.1)
    gtmask2 = (f >= currHR * 2 - 0.1) & (f <= currHR * 2 + 0.1)
    sPower = np.sum(np.take(pxx, np.where(gtmask1 | gtmask2)))
    if signal == 'pulse':
        fmask2 = (f >= 0.75) & (f <= 4)
    else:
        fmask2 = (f >= 0.08) & (f <= 0.5)
    allPower = np.sum(np.take(pxx, np.where(fmask2)))
    SNR_temp = mag2db(sPower / (allPower - sPower))
    return SNR_temp


# %%  Processing


def calculate_metric_peak_per_video(predictions, labels,signal='pulse',
                                    fs=30, bpFlag=True):
    if signal == 'pulse':
        [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2],
                        btype='bandpass')  # 2.5 -> 1.7
    else:
        [b, a] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')

    data_len = len(predictions)
    HR_pred = []
    HR0_pred = []
    all_peaks = []
    all_peaks0 = []

    if signal == 'pulse':
        pred_window = detrend(np.cumsum(predictions), 100)
        label_window = detrend(np.cumsum(labels), 100)

    if bpFlag:
        pred_window = scipy.signal.filtfilt(b, a, np.double(pred_window))
        label_window = scipy.signal.filtfilt(b, a, np.double(label_window))

    # Peak detection
    labels_peaks, _ = scipy.signal.find_peaks(label_window)
    preds_peaks, _ = scipy.signal.find_peaks(pred_window)


    temp_HR_0 = 60 / (np.mean(np.diff(labels_peaks)) / fs)
    temp_HR = 60 / (np.mean(np.diff(preds_peaks)) / fs)

    HR_pred.append(temp_HR)
    HR0_pred.append(temp_HR_0)
    all_peaks.extend(preds_peaks)
    all_peaks0.extend(labels_peaks)

    HR = np.mean(np.array(HR_pred))
    HR0 = np.mean(np.array(HR0_pred))
    return HR0, HR,labels_peaks,label_window


def calculate_metric_FFT_per_video(predictions, labels, signal='pulse',
                                   fs=30, bpFlag=True):
    if signal == 'pulse':
        [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2],
                        btype='bandpass')  # 2.5 -> 1.7
    else:
        [b, a] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')

    if signal == 'pulse':
        pred_window = detrend(np.cumsum(predictions), 100)
        label_window = detrend(np.cumsum(labels), 100)
    else:
        pred_window = np.cumsum(predictions)

    if bpFlag:
        pred_window = scipy.signal.filtfilt(b, a, np.double(pred_window))
        label_window = scipy.signal.filtfilt(b, a, np.double(label_window))
    filt_signal = pred_window
    pred_window = np.expand_dims(pred_window, 0)
    label_window = np.expand_dims(label_window, 0)
    # Predictions FFT
    N = next_power_of_2(pred_window.shape[1])
    f_prd, pxx_pred = scipy.signal.periodogram(
        pred_window, fs=fs, nfft=N, detrend=False)
    if signal == 'pulse':
        # regular Heart beat are 0.75*60 and 2.5*60
        fmask_pred = np.argwhere((f_prd >= 0.75) & (f_prd <= 2.5))
    else:
        # regular Heart beat are 0.75*60 and 2.5*60
        fmask_pred = np.argwhere((f_prd >= 0.08) & (f_prd <= 0.5))
    pred_window = np.take(f_prd, fmask_pred)
    # Labels FFT
    f_label, pxx_label = scipy.signal.periodogram(
        label_window, fs=fs, nfft=N, detrend=False)
    if signal == 'pulse':
        # regular Heart beat are 0.75*60 and 2.5*60
        fmask_label = np.argwhere((f_label >= 0.75) & (f_label <= 2.5))
    else:
        # regular Heart beat are 0.75*60 and 2.5*60
        fmask_label = np.argwhere((f_label >= 0.08) & (f_label <= 0.5))
    label_window,pxx = np.take(f_label, fmask_label),np.take(pxx_label, fmask_label)
    # MAE
    temp_HR, temp_HR_0 = calculate_HR(
        pxx_pred, pred_window, fmask_pred, pxx_label, label_window, fmask_label)

    return temp_HR_0, temp_HR,label_window,pxx



def plot_signal(label_signal,peaks0,HR,label_window,pxx,HR_FFT):
    '''
    HR0: Ground Truth Heart Rate using FFT
    HR: Predicted Heart Rate using nfft
    HR0_peak: Ground Truth Heart Rate using Peak Detection
    HR_peak: Predicted Heart Rate using Peak Detection
    peaks: Peak locations from predicted signal
    peaks0: Peak locations from ground-truth signal
    '''
    plt.subplot(211)
    plt.plot(label_window, np.reshape(pxx, [-1, 1]))
    plt.title("FFT:"+str(HR_FFT))
    label_signal = np.array(label_signal)
    plt.subplot(212)
    plt.plot(label_signal)
    plt.plot(peaks0, label_signal[peaks0], "x")
    plt.title("PEAK:"+str(HR))
    plt.show()

def plot_fft(signal,HR):
    label_signal = np.array(signal)
    plt.subplot(211)
    plt.plot(label_signal)
    plt.title(HR)
    plt.show()

# 先读UBFC的wave

fig = plt.gcf()
fig.set_size_inches(20, 12)
# TODO: you can give the figure a name here:
fig.suptitle("Figure Name", fontsize=15)
bvp, hr, times,fs  = read_bvp()
print("fs:",fs)
print("average hr from ground truth label:", np.mean(np.array(hr)))
HR0, HR,peaks,signal = calculate_metric_peak_per_video(bvp, bvp)
print("average hr from peak detection:", HR0)
HR0_FFT, HR_FFT,label_window,pxx = calculate_metric_FFT_per_video(bvp, bvp)
print("average hr from fft:", HR0)
plot_signal(signal,peaks,HR0,label_window,pxx,HR_FFT)
