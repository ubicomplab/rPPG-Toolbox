import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from evaluation.metrics import calculate_metrics, _reform_data_from_dict
from evaluation.post_process import _detrend, _next_power_of_2

import scipy
import scipy.io
from scipy.signal import butter
from scipy.sparse import spdiags




# PPG Metrics
def calculate_bvp_metrics(predictions, labels, config):
    print('=====================')
    print('==== PPG Metrics ====')
    print('=====================')
    calculate_metrics(predictions, labels, config)
    print('')

# Resp Metrics
def _calculate_fft_rr(resp_signal, fs=30, low_pass=0.08, high_pass=0.5):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    resp_signal = np.expand_dims(resp_signal, 0)
    N = _next_power_of_2(resp_signal.shape[1])
    f_resp, pxx_resp = scipy.signal.periodogram(resp_signal, fs=fs, nfft=N, detrend=False)
    fmask_resp = np.argwhere((f_resp >= low_pass) & (f_resp <= high_pass))
    mask_resp = np.take(f_resp, fmask_resp)
    mask_pxx = np.take(pxx_resp, fmask_resp)
    fft_rr = np.take(mask_resp, np.argmax(mask_pxx, 0))[0] * 60
    return fft_rr

def _calculate_peak_rr(resp_signal, fs):
    """Calculate heart rate based on PPG using peak detection."""
    resp_peaks, _ = scipy.signal.find_peaks(resp_signal)
    rr_peak = 60 / (np.mean(np.diff(resp_peaks)) / fs)
    return rr_peak


def calculate_resp_metrics_per_video(predictions, labels, fs=30, diff_flag=True, use_bandpass=True, rr_method='FFT'):
    """Calculate video-level RR"""
    if diff_flag:  # if the predictions and labels are 1st derivative of Resp signal.
        predictions = _detrend(np.cumsum(predictions), 100)
        labels = _detrend(np.cumsum(labels), 100)
    else:
        predictions = _detrend(predictions, 100)
        labels = _detrend(labels, 100)
    if use_bandpass:
        # bandpass filter between [0.08, 0.5] Hz
        # equals [5, 30] breaths per min
        [b, a] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')
        predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
        labels = scipy.signal.filtfilt(b, a, np.double(labels))
    if rr_method == 'FFT':
        rr_pred = _calculate_fft_rr(predictions, fs=fs)
        rr_label = _calculate_fft_rr(labels, fs=fs)
    elif rr_method == 'Peak':
        rr_pred = _calculate_peak_rr(predictions, fs=fs)
        rr_label = _calculate_peak_rr(labels, fs=fs)
    else:
        raise ValueError('Please use FFT or Peak to calculate your RR.')
    return rr_label, rr_pred

def calculate_resp_metrics(predictions, labels, config):
    """Calculate Respiration Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""

    print('=====================')
    print('==== Resp Metrics ===')
    print('=====================')

    predict_rr_fft_all = list()
    gt_rr_fft_all = list()
    predict_rr_peak_all = list()
    gt_rr_peak_all = list()
    for index in predictions.keys():
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        if config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Raw":
            diff_flag_test = False
        elif config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            diff_flag_test = True
        else:
            raise ValueError("Not supported label type in testing!")
        gt_rr_fft, pred_rr_fft = calculate_resp_metrics_per_video(
            prediction, label, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, rr_method='FFT')
        gt_rr_peak, pred_rr_peak = calculate_resp_metrics_per_video(
            prediction, label, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, rr_method='Peak')
        gt_rr_fft_all.append(gt_rr_fft)
        predict_rr_fft_all.append(pred_rr_fft)
        predict_rr_peak_all.append(pred_rr_peak)
        gt_rr_peak_all.append(gt_rr_peak)
    predict_rr_peak_all = np.array(predict_rr_peak_all)
    predict_rr_fft_all = np.array(predict_rr_fft_all)
    gt_rr_peak_all = np.array(gt_rr_peak_all)
    gt_rr_fft_all = np.array(gt_rr_fft_all)
    for metric in config.TEST.METRICS:
        if metric == "MAE":
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                MAE_FFT = np.mean(np.abs(predict_rr_fft_all - gt_rr_fft_all))
                print("FFT MAE (FFT Label):{0}".format(MAE_FFT))
            elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                MAE_PEAK = np.mean(np.abs(predict_rr_peak_all - gt_rr_peak_all))
                print("Peak MAE (Peak Label):{0}".format(MAE_PEAK))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "RMSE":
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                RMSE_FFT = np.sqrt(np.mean(np.square(predict_rr_fft_all - gt_rr_fft_all)))
                print("FFT RMSE (FFT Label):{0}".format(RMSE_FFT))
            elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                RMSE_PEAK = np.sqrt(np.mean(np.square(predict_rr_peak_all - gt_rr_peak_all)))
                print("PEAK RMSE (Peak Label):{0}".format(RMSE_PEAK))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "MAPE":
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                MAPE_FFT = np.mean(np.abs((predict_rr_fft_all - gt_rr_fft_all) / gt_rr_fft_all)) * 100
                print("FFT MAPE (FFT Label):{0}".format(MAPE_FFT))
            elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                MAPE_PEAK = np.mean(np.abs((predict_rr_peak_all - gt_rr_peak_all) / gt_rr_peak_all)) * 100
                print("PEAK MAPE (Peak Label):{0}".format(MAPE_PEAK))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "Pearson":
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                Pearson_FFT = np.corrcoef(predict_rr_fft_all, gt_rr_fft_all)
                print("FFT Pearson (FFT Label):{0}".format(Pearson_FFT[0][1]))
            elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                Pearson_PEAK = np.corrcoef(predict_rr_peak_all, gt_rr_peak_all)
                print("PEAK Pearson  (Peak Label):{0}".format(Pearson_PEAK[0][1]))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        else:
            # raise ValueError("Wrong Test Metric Type")
            pass

# AU Metrics
def _reform_au_data_from_dict(predictions, labels, flatten=True):
    for index in predictions.keys():
        predictions[index] = _reform_data_from_dict(predictions[index], flatten=flatten)
        labels[index] = _reform_data_from_dict(labels[index], flatten=flatten)

    return predictions, labels


def calculate_au_metrics(preds, labels, config):

    for index in preds.keys():
        preds[index] = _reform_data_from_dict(preds[index], flatten=False)
        labels[index] = _reform_data_from_dict(labels[index], flatten=False)

    metrics_dict = dict()
    all_trial_preds = []
    all_trial_labels = []

    for T in labels.keys():
        all_trial_preds.append(preds[T])
        all_trial_labels.append(labels[T])

    all_trial_preds = np.concatenate(all_trial_preds, axis=0)
    all_trial_labels = np.concatenate(all_trial_labels, axis=0)

    for metric in config.TEST.METRICS:

        if metric == '12AUF1':

            named_AU = ['AU01', 'AU02', 'AU04', 'AU06', 'AU07', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU23', 'AU24']
            AU_data = dict()
            AU_data['labels'] = dict()
            AU_data['preds'] = dict()

            for i in range(len(named_AU)):
                AU_data['labels'][named_AU[i]] = all_trial_labels[:, i, 0]
                AU_data['preds'][named_AU[i]] = all_trial_preds[:, i]

            # Calculate F1
            f1_dict = dict()  
            avg_f1 = 0
            acc_dict = dict()
            avg_acc = 0  
            print('')
            print('=====================')
            print('======= AU F1 =======')
            print('=====================')
            print('AU | F1')
            print('AU | F1 | Avg Val | Avg Label Val')
            for au in named_AU:
                preds = np.array(AU_data['preds'][au])
                preds[preds < 0.5] = 0
                preds[preds >= 0.5] = 1
                labels = np.array(AU_data['labels'][au])
                f1 = f1_score(labels, preds)*100
                f1_dict[au] = (f1, np.sum(preds)/len(preds), np.sum(labels)/len(labels))
                avg_f1 += f1
                print(au, f1, np.sum(preds)/len(preds), np.sum(labels)/len(labels))

                # get AU accuracy
                acc = sum(1 for x,y in zip(preds,labels) if x == y) / len(labels)
                acc_dict[au] = acc
                avg_acc += acc

            # Save Dictionary
            metrics_dict['12AUF1'] = f1_dict
            metrics_dict['12AUAcc'] = acc_dict

            print('Average F1:', avg_f1/len(named_AU))
            print('Average Acc:', avg_acc/len(named_AU))

        else:
            pass
            # print('This AU metric does not exit')