# %% Import
import numpy as np
import scipy
import scipy.io
from scipy.signal import butter
from scipy.sparse import spdiags


# %% Helper Function
def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()


def detrend(signal, Lambda):
    signal_length = signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal

def mag2db(mag):

    return 20. * np.log10(mag)


def calculate_HR(pxx_pred, frange_pred, fmask_pred, pxx_label, frange_label, fmask_label):
    pred_HR = np.take(frange_pred, np.argmax(np.take(pxx_pred, fmask_pred), 0))[0] * 60
    ground_truth_HR = np.take(frange_label, np.argmax(np.take(pxx_label, fmask_label), 0))[0] * 60
    return pred_HR, ground_truth_HR


def calculate_SNR(pxx_pred, f_pred, currHR, signal):
    currHR = currHR/60
    f = f_pred
    pxx = pxx_pred
    gtmask1 = (f >= currHR - 0.1) & (f <= currHR + 0.1)
    gtmask2 = (f >= currHR * 2 - 0.1) & (f <= currHR * 2 + 0.1)
    sPower = np.sum(np.take(pxx, np.where(gtmask1 | gtmask2)))
    if signal == 'pulse':
        fmask2 = (f >= 0.75) & (f <= 4)
    else:
        fmask2 = (f >= 0.08) & (f <= 0.5)
    allPower = np.sum(np.take(pxx, np.where(fmask2 == True)))
    SNR_temp = mag2db(sPower / (allPower - sPower))
    return SNR_temp
# %%  Processing


def calculate_metric_peak_per_video(predictions, labels, signal='pulse',
                                    fs=30, bpFlag=True):
    if signal == 'pulse':
        [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass') # 2.5 -> 1.7
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
    else:
        pred_window = np.cumsum(pred_window)

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
    all_peaks.extend(preds_peaks + j)
    all_peaks0.extend(labels_peaks + j)

    HR = np.mean(np.array(HR_pred))
    HR0 = np.mean(np.array(HR0_pred))

    return HR0, HR, all_peaks, all_peaks0, pred_signal, label_signal


def calculate_metric_FFT_per_video(predictions, labels, signal='pulse',
                                    fs=30, bpFlag=True):
    if signal == 'pulse':
        [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')  # 2.5 -> 1.7
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

    pred_window = np.expand_dims(pred_window, 0)
    label_window = np.expand_dims(label_window, 0)
    # Predictions FFT
    N = next_power_of_2(pred_window.shape[1])
    f_prd, pxx_pred = scipy.signal.periodogram(pred_window, fs=fs, nfft=N, detrend=False)
    if signal == 'pulse':
        fmask_pred = np.argwhere((f_prd >= 0.75) & (f_prd <= 2.5))  # regular Heart beat are 0.75*60 and 2.5*60
    else:
        fmask_pred = np.argwhere((f_prd >= 0.08) & (f_prd <= 0.5))  # regular Heart beat are 0.75*60 and 2.5*60
    pred_window = np.take(f_prd, fmask_pred)
    # Labels FFT
    f_label, pxx_label = scipy.signal.periodogram(label_window, fs=fs, nfft=N, detrend=False)
    if signal == 'pulse':
        fmask_label = np.argwhere((f_label >= 0.75) & (f_label <= 2.5))  # regular Heart beat are 0.75*60 and 2.5*60
    else:
        fmask_label = np.argwhere((f_label >= 0.08) & (f_label <= 0.5))  # regular Heart beat are 0.75*60 and 2.5*60
    label_window = np.take(f_label, fmask_label)

    # MAE
    temp_HR, temp_HR_0 = calculate_HR(pxx_pred, pred_window, fmask_pred, pxx_label, label_window, fmask_label)

    return temp_HR_0, temp_HR


def plot_signal(pred_signal, label_signal, video_filename, save_path, HR0, HR,
                    HR0_peak, HR_peak, peaks, peaks0):
    '''
    HR0: Ground Truth Heart Rate using FFT
    HR: Predicted Heart Rate using nfft
    HR0_peak: Ground Truth Heart Rate using Peak Detection
    HR_peak: Predicted Heart Rate using Peak Detection
    peaks: Peak locations from predicted signal
    peaks0: Peak locations from ground-truth signal
    '''
    pred_signal = np.array(pred_signal)
    label_signal = np.array(label_signal)
    fig = plt.gcf()
    fig.set_size_inches(20, 12)
    figure_title = video_filename + ' | ' + ' | HR_FFT: ' + str(HR) + ' | HR0_FFT: ' + str(HR0) + ' | HR_peak: ' + str(HR_peak) \
                        + ' | HR0_peak: ' + str(HR0_peak) + ' | MAE-FFT: ' + str(abs(HR0-HR)) + ' | MAE-Peak: ' \
                        + str(abs(HR0_peak-HR_peak))
    fig.suptitle(figure_title, fontsize=15)
    plt.subplot(211)
    plt.plot(pred_signal)
    plt.plot(peaks, signal[peaks], "x")
    plt.title('Prediction')
    plt.show()
    plt.subplot(212)
    plt.plot(label_signal)
    plt.plot(peaks0, label_signal[peaks0], "x")
    plt.title('Label')
    plt.show()
    plt.savefig(os.path.join(save_path, video_filename + '.png'))
    plt.close()

