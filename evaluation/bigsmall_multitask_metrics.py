import numpy as np
import scipy
import scipy.io
from scipy.signal import butter
from sklearn.metrics import f1_score, precision_recall_fscore_support
from evaluation.metrics import calculate_metrics, _reform_data_from_dict
from evaluation.post_process import _detrend, _next_power_of_2, _calculate_SNR
from tqdm import tqdm
from evaluation.BlandAltmanPy import BlandAltman

# PPG Metrics
def calculate_bvp_metrics(predictions, labels, config):
    """Calculate PPG Metrics (MAE, RMSE, MAPE, Pearson Coef., SNR)."""

    print('=====================')
    print('==== PPG Metrics ====')
    print('=====================')
    calculate_metrics(predictions, labels, config)
    print('')


# Resp Metrics
def _calculate_fft_rr(resp_signal, fs=30, low_pass=0.13, high_pass=0.5):
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
        # bandpass filter between [0.13, 0.5] Hz
        # equals [8, 30] breaths per min
        [b, a] = butter(1, [0.13 / fs * 2, 0.5 / fs * 2], btype='bandpass')
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
    SNR = _calculate_SNR(predictions, rr_label, fs=fs, low_pass=0.13, high_pass=0.5)
    return rr_label, rr_pred, SNR


def calculate_resp_metrics(predictions, labels, config):
    """Calculate Respiration Metrics (MAE, RMSE, MAPE, Pearson Coef., SNR)."""

    print('=====================')
    print('==== Resp Metrics ===')
    print('=====================')

    predict_rr_fft_all = list()
    gt_rr_fft_all = list()
    predict_rr_peak_all = list()
    gt_rr_peak_all = list()
    SNR_all = list()
    for index in tqdm(predictions.keys(), ncols=80):
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        video_frame_size = prediction.shape[0]
        if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
            window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.TEST.DATA.FS
            if window_frame_size > video_frame_size:
                window_frame_size = video_frame_size
        else:
            window_frame_size = video_frame_size

        for i in range(0, len(prediction), window_frame_size):
            pred_window = prediction[i:i+window_frame_size]
            label_window = label[i:i+window_frame_size]

            if len(pred_window) < 9:
                print(f"Window frame size of {len(pred_window)} is smaller than minimum pad length of 9. Window ignored!")
                continue

            if config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                    config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Raw":
                diff_flag_test = False
            elif config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
                diff_flag_test = True
            else:
                raise ValueError("Unsupported label type in testing!")
            
            if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                gt_rr_peak, pred_rr_peak, SNR = calculate_resp_metrics_per_video(
                    prediction, label, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, rr_method='Peak')
                gt_rr_peak_all.append(gt_rr_peak)
                predict_rr_peak_all.append(pred_rr_peak)
                SNR_all.append(SNR)
            elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                gt_rr_fft, pred_rr_fft, SNR = calculate_resp_metrics_per_video(
                    prediction, label, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, rr_method='FFT')
                gt_rr_fft_all.append(gt_rr_fft)
                predict_rr_fft_all.append(pred_rr_fft)
                SNR_all.append(SNR)
            else:
                raise ValueError("Inference evaluation method name wrong!")

    if config.INFERENCE.EVALUATION_METHOD == "FFT":
        gt_rr_fft_all = np.array(gt_rr_fft_all)
        predict_rr_fft_all = np.array(predict_rr_fft_all)
        SNR_all = np.array(SNR_all)
        num_test_samples = len(predict_rr_fft_all)
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_FFT = np.mean(np.abs(predict_rr_fft_all - gt_rr_fft_all))
                standard_error = np.std(np.abs(predict_rr_fft_all - gt_rr_fft_all)) / np.sqrt(num_test_samples)
                print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, standard_error))
            elif metric == "RMSE":
                RMSE_FFT = np.sqrt(np.mean(np.square(predict_rr_fft_all - gt_rr_fft_all)))
                standard_error = np.std(np.square(predict_rr_fft_all - gt_rr_fft_all)) / np.sqrt(num_test_samples)
                print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, standard_error))
            elif metric == "MAPE":
                MAPE_FFT = np.mean(np.abs((predict_rr_fft_all - gt_rr_fft_all) / gt_rr_fft_all)) * 100
                standard_error = np.std(np.abs((predict_rr_fft_all - gt_rr_fft_all) / gt_rr_fft_all)) / np.sqrt(num_test_samples) * 100
                print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, standard_error))
            elif metric == "Pearson":
                Pearson_FFT = np.corrcoef(predict_rr_fft_all, gt_rr_fft_all)
                correlation_coefficient = Pearson_FFT[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_FFT = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1}".format(SNR_FFT, standard_error))
            elif "AU" in metric:
                pass
            elif "BA" in metric:
                compare = BlandAltman(gt_rr_fft_all, predict_rr_fft_all, config, averaged=True)
                compare.scatter_plot(
                    x_label='GT RR [bpm]',
                    y_label='Predicted RR [bpm]', 
                    show_legend=True, figure_size=(5, 5), 
                    file_name=f'FFT_BlandAltman_ScatterPlot.pdf', 
                    measure_lower_lim=10, 
                    measure_upper_lim=60)
                compare.difference_plot(
                    x_label='Difference between Predicted RR and GT RR [bpm]', 
                    y_label='Average of Predicted RR and GT RR [bpm]', 
                    show_legend=True, figure_size=(5, 5), file_name=f'FFT_BlandAltman_DifferencePlot.pdf')
            else:
                raise ValueError("Wrong Test Metric Type")
    elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
        gt_rr_peak_all = np.array(gt_rr_peak_all)
        predict_rr_peak_all = np.array(predict_rr_peak_all)
        SNR_all = np.array(SNR_all)
        num_test_samples = len(predict_rr_peak_all)
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_PEAK = np.mean(np.abs(predict_rr_peak_all - gt_rr_peak_all))
                standard_error = np.std(np.abs(predict_rr_peak_all - gt_rr_peak_all)) / np.sqrt(num_test_samples)
                print("Peak MAE (Peak Label): {0} +/- {1}".format(MAE_PEAK, standard_error))
            elif metric == "RMSE":
                RMSE_PEAK = np.sqrt(np.mean(np.square(predict_rr_peak_all - gt_rr_peak_all)))
                standard_error = np.std(np.square(predict_rr_peak_all - gt_rr_peak_all)) / np.sqrt(num_test_samples)
                print("PEAK RMSE (Peak Label): {0} +/- {1}".format(RMSE_PEAK, standard_error))
            elif metric == "MAPE":
                MAPE_PEAK = np.mean(np.abs((predict_rr_peak_all - gt_rr_peak_all) / gt_rr_peak_all)) * 100
                standard_error = np.std(np.abs((predict_rr_peak_all - gt_rr_peak_all) / gt_rr_peak_all)) / np.sqrt(num_test_samples) * 100
                print("PEAK MAPE (Peak Label): {0} +/- {1}".format(MAPE_PEAK, standard_error))
            elif metric == "Pearson":
                Pearson_PEAK = np.corrcoef(predict_rr_peak_all, gt_rr_peak_all)
                correlation_coefficient = Pearson_PEAK[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("PEAK Pearson (Peak Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_PEAK = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1}".format(SNR_PEAK, standard_error))
            elif "AU" in metric:
                pass
            elif "BA" in metric:
                compare = BlandAltman(gt_rr_peak_all, predict_rr_peak_all, config, averaged=True)
                compare.scatter_plot(
                    x_label='GT RR [bpm]',
                    y_label='Predicted RR [bpm]', 
                    show_legend=True, figure_size=(5, 5), 
                    file_name=f'Peak_BlandAltman_ScatterPlot.pdf', 
                    measure_lower_lim=10, 
                    measure_upper_lim=60)
                compare.difference_plot(
                    x_label='Difference between Predicted RR and GT RR [bpm]', 
                    y_label='Average of Predicted RR and GT RR [bpm]', 
                    show_legend=True, figure_size=(5, 5), file_name=f'Peak_BlandAltman_DifferencePlot.pdf')
            else:
                raise ValueError("Wrong Test Metric Type")
    else:
        raise ValueError("Inference evaluation method name wrong!")


# AU Metrics
def _reform_au_data_from_dict(predictions, labels, flatten=True):
    for index in predictions.keys():
        predictions[index] = _reform_data_from_dict(predictions[index], flatten=flatten)
        labels[index] = _reform_data_from_dict(labels[index], flatten=flatten)

    return predictions, labels


def calculate_bp4d_au_metrics(preds, labels, config):
    """Calculate AU Metrics (12 AU F1, Precision, Mean F1, Mean Acc, Mean Precision)."""

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

        if metric == 'AU_METRICS':

            named_AU = ['AU01', 'AU02', 'AU04', 'AU06', 'AU07', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU23', 'AU24']
            AU_data = dict()
            AU_data['labels'] = dict()
            AU_data['preds'] = dict()

            for i in range(len(named_AU)):
                AU_data['labels'][named_AU[i]] = all_trial_labels[:, i, 0]
                AU_data['preds'][named_AU[i]] = all_trial_preds[:, i]

            # Calculate F1
            metric_dict = dict()  
            avg_f1 = 0
            avg_prec = 0 
            avg_acc = 0   

            print('')
            print('=====================')
            print('===== AU METRICS ====')
            print('=====================')
            print('AU / F1 / Precision')
            for au in named_AU:
                preds = np.array(AU_data['preds'][au])
                preds[preds < 0.5] = 0
                preds[preds >= 0.5] = 1
                labels = np.array(AU_data['labels'][au])

                precision, recall, f1, support = precision_recall_fscore_support(labels, preds, beta=1.0)

                f1 = f1[1]
                precision = precision[1]
                recall = recall[1]

                f1 = f1*100
                precision = precision*100
                recall = recall*100
                acc = sum(1 for x,y in zip(preds,labels) if x == y) / len(labels) * 100

                # save to dict
                metric_dict[au] = (f1, precision, recall, acc)

                # update avgs
                avg_f1 += f1
                avg_prec += precision
                avg_acc += acc
                
                # Print
                print(au, f1, precision)

            # Save Dictionary
            avg_f1 = avg_f1/len(named_AU)
            avg_acc = avg_acc/len(named_AU)
            avg_prec = avg_prec/len(named_AU)

            metric_dict['12AU_AvgF1'] = avg_f1
            metric_dict['12AU_AvgPrec'] = avg_prec
            metric_dict['12AU_AvgAcc'] = avg_acc

            print('')
            print('Mean 12 AU F1:', avg_f1)
            print('Mean 12 AU Prec.:', avg_prec)
            print('Mean 12 AU Acc.:', avg_acc)
            print('')

        else:
            pass
            # print('This AU metric does not exit')