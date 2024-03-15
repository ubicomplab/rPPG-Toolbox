import numpy as np
import pandas as pd
import torch
from evaluation.post_process import *
from tqdm import tqdm
from evaluation.BlandAltmanPy import BlandAltman

def read_label(dataset):
    """Read manually corrected labels."""
    df = pd.read_csv("label/{0}_Comparison.csv".format(dataset))
    out_dict = df.to_dict(orient='index')
    out_dict = {str(value['VideoID']): value for key, value in out_dict.items()}
    return out_dict


def read_hr_label(feed_dict, index):
    """Read manually corrected UBFC labels."""
    # For UBFC only
    if index[:7] == 'subject':
        index = index[7:]
    video_dict = feed_dict[index]
    if video_dict['Preferred'] == 'Peak Detection':
        hr = video_dict['Peak Detection']
    elif video_dict['Preferred'] == 'FFT':
        hr = video_dict['FFT']
    else:
        hr = video_dict['Peak Detection']
    return index, hr


def _reform_data_from_dict(data, flatten=True):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)

    if flatten:
        sort_data = np.reshape(sort_data.cpu(), (-1))
    else:
        sort_data = np.array(sort_data.cpu())

    return sort_data


def calculate_metrics(predictions, labels, config):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    predict_hr_all = []
    gt_hr_all = []
    SNR_all = []
    
    print("Calculating metrics!")
    for index in tqdm(predictions.keys(), ncols=80):
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        video_frame_size = prediction.shape[0]
        if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
            window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.TEST.DATA.FS
            overlap = config.INFERENCE.EVALUATION_WINDOW.WINDOW_OVERLAP
            
            if window_frame_size > video_frame_size:
                window_frame_size = video_frame_size
                overlap = 0
        else:
            window_frame_size = video_frame_size
            overlap = 0
            
        temp_gt = []
        temp_pred = []
            
        for i in range(0, len(prediction), window_frame_size - overlap):
            pred_window = prediction[i:i+window_frame_size]
            label_window = label[i:i+window_frame_size]

            if len(pred_window) <= 9:
                print(f"Window frame size of {len(pred_window)} is smaller than window size of {window_frame_size}. Window ignored!")
                continue

            if config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                    config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Raw":
                diff_flag_test = False
            elif config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
                diff_flag_test = True
            else:
                raise ValueError("Unsupported label type in testing!")
            
            gt_hr, pred_hr, SNR = calculate_metric_per_video(pred_window, label_window, diff_flag=diff_flag_test,
                                                            fs=config.TEST.DATA.FS, hr_method=config.INFERENCE.EVALUATION_METHOD)

            gt_hr_all.append(gt_hr)
            predict_hr_all.append(pred_hr)
            SNR_all.append(SNR)
            
        #     temp_gt.append(gt_hr)
        #     temp_pred.append(pred_hr)
            
        # temp_gt = np.array(temp_gt)
        # temp_pred = np.array(temp_pred)
        # print('GT HR: ', temp_gt)
        # print('Predicted HR: ', temp_pred)
        
        # num_test_samples = len(temp_pred)
        # RMSE = np.sqrt(np.mean(np.square(temp_pred - temp_gt)))
        # standard_error = np.std(np.square(temp_pred - temp_gt)) / np.sqrt(num_test_samples)
        # print("RMSE: {0} +/- {1}".format(RMSE, standard_error))
    
    # Filename ID to be used in any results files (e.g., Bland-Altman plots) that get saved
    if config.TOOLBOX_MODE == 'train_and_test':
        filename_id = config.TRAIN.MODEL_FILE_NAME
    elif config.TOOLBOX_MODE == 'only_test':
        model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
        filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
    else:
        raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')

    predict_hr_all = np.array(predict_hr_all)
    gt_hr_all = np.array(gt_hr_all)
    SNR_all = np.array(SNR_all)
    num_test_samples = len(predict_hr_all)
    
    # print('GT HR: ', gt_hr_all)
    # print('Predicted HR: ', predict_hr_all)

    for metric in config.TEST.METRICS:
        if metric == "MAE":
            MAE = np.mean(np.abs(predict_hr_all - gt_hr_all))
            standard_error = np.std(np.abs(predict_hr_all - gt_hr_all)) / np.sqrt(num_test_samples)
            print("MAE: {0} +/- {1}".format(MAE, standard_error))
            # print(MAE)
        elif metric == "RMSE":
            RMSE = np.sqrt(np.mean(np.square(predict_hr_all - gt_hr_all)))
            standard_error = np.std(np.square(predict_hr_all - gt_hr_all)) / np.sqrt(num_test_samples)
            print("RMSE: {0} +/- {1}".format(RMSE, standard_error))
            # print(RMSE)
        elif metric == "MAPE":
            MAPE = np.mean(np.abs((predict_hr_all - gt_hr_all) / gt_hr_all)) * 100
            standard_error = np.std(np.abs((predict_hr_all - gt_hr_all) / gt_hr_all)) / np.sqrt(num_test_samples) * 100
            print("MAPE: {0} +/- {1}".format(MAPE, standard_error))
            # print(MAPE)
        elif metric == "Pearson":
            Pearson = np.corrcoef(predict_hr_all, gt_hr_all)
            correlation_coefficient = Pearson[0][1]
            standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
            print("Pearson: {0} +/- {1}".format(correlation_coefficient, standard_error))
            # print(correlation_coefficient)
        elif metric == "SNR":
            SNR = np.mean(SNR_all)
            standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
            print("SNR: {0} +/- {1} (dB)".format(SNR, standard_error))
            # print(SNR)
        elif "BA" in metric:
            compare = BlandAltman(gt_hr_all, predict_hr_all, config, averaged=True)
            compare.scatter_plot(
                x_label='GT PPG HR [bpm]',
                y_label='rPPG HR [bpm]',
                show_legend=True, figure_size=(5, 5),
                the_title=f'{filename_id}_BlandAltman_ScatterPlot',
                file_name=f'{filename_id}_BlandAltman_ScatterPlot.pdf')
            compare.difference_plot(
                x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                y_label='Average of rPPG HR and GT PPG HR [bpm]',
                show_legend=True, figure_size=(5, 5),
                the_title=f'{filename_id}_BlandAltman_DifferencePlot',
                file_name=f'{filename_id}_BlandAltman_DifferencePlot.pdf')
        else:
            raise ValueError("Wrong Test Metric Type")