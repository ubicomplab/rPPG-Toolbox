"""Unsupervised learning methods including POS, GREEN, CHROME, ICA, LGI and PBV."""
import numpy as np
from evaluation.post_process import *
from unsupervised_methods.methods.CHROME_DEHAAN import *
from unsupervised_methods.methods.GREEN import *
from unsupervised_methods.methods.ICA_POH import *
from unsupervised_methods.methods.LGI import *
from unsupervised_methods.methods.PBV import *
from unsupervised_methods.methods.POS_WANG import *
from tqdm import tqdm
from evaluation.BlandAltmanPy import BlandAltman

import os
import pickle
import torch


def save_test_outputs(predictions, labels, config, method_name):

    output_dir = config.UNSUPERVISED.OUTPUT_SAVE_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    filename_id = method_name + "_" + config.UNSUPERVISED.DATA.DATASET

    output_path = os.path.join(output_dir, filename_id + '_outputs.pickle')

    data = dict()
    data['predictions'] = predictions
    data['labels'] = labels
    data['label_type'] = config.UNSUPERVISED.DATA.PREPROCESS.LABEL_TYPE
    data['fs'] = config.UNSUPERVISED.DATA.FS

    with open(output_path, 'wb') as handle: # save out frame dict pickle file
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Saving outputs to:', output_path)
    print('')


def unsupervised_predict(config, data_loader, method_name):
    """ Model evaluation on the testing dataset."""
    if data_loader["unsupervised"] is None:
        raise ValueError("No data for unsupervised method predicting")
    print("\n===Unsupervised Method ( " + method_name + " ) Predicting ===")

    predict_hr_all = []
    gt_hr_all = []
    SNR_all = []
    
    predictions = dict()
    labels = dict()
    
    sbar = tqdm(data_loader["unsupervised"], ncols=80)
    for _, test_batch in enumerate(sbar):
        batch_size = test_batch[0].shape[0]
        for idx in range(batch_size):
            data_input, labels_input = test_batch[0][idx].cpu().numpy(), test_batch[1][idx].cpu().numpy()
                        
            if method_name == "POS":
                BVP = POS_WANG(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "CHROM":
                BVP = CHROME_DEHAAN(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "ICA":
                BVP = ICA_POH(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "GREEN":
                BVP = GREEN(data_input)
            elif method_name == "LGI":
                BVP = LGI(data_input)
            elif method_name == "PBV":
                BVP = PBV(data_input)
            else:
                raise ValueError("unsupervised method name wrong!")
            
            subj_index = test_batch[2][idx]
            sort_index = int(test_batch[3][idx])
            if subj_index not in predictions.keys():
                predictions[subj_index] = dict()
                labels[subj_index] = dict()
                
            if config.UNSUPERVISED.DATA.PREPROCESS.DO_CHUNK:
                chunk_len = config.UNSUPERVISED.DATA.PREPROCESS.CHUNK_LENGTH
                predictions[subj_index][sort_index] = torch.from_numpy(BVP[idx * chunk_len:(idx + 1) * chunk_len].copy())
                labels[subj_index][sort_index] = torch.from_numpy(labels_input[idx * chunk_len:(idx + 1) * chunk_len].copy())
            else:
                predictions[subj_index][sort_index] = torch.from_numpy(BVP.copy())
                labels[subj_index][sort_index] = torch.from_numpy(labels_input.copy())
            
            video_frame_size = test_batch[0].shape[1]
            if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
                window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.UNSUPERVISED.DATA.FS
                overlap = config.INFERENCE.EVALUATION_WINDOW.WINDOW_OVERLAP
                
                if window_frame_size > video_frame_size:
                    window_frame_size = video_frame_size
                    overlap = 0
                elif overlap > window_frame_size:
                    overlap = 0
            else:
                window_frame_size = video_frame_size
                overlap = 0

            temp_gt = []
            temp_pred = []

            for i in range(0, len(BVP), window_frame_size - overlap):
                BVP_window = BVP[i:i+window_frame_size]
                label_window = labels_input[i:i+window_frame_size]

                if len(BVP_window) < window_frame_size:
                    # print(f"Window frame size of {len(BVP_window)} is smaller than window size of {window_frame_size}. Window ignored!")
                    continue

                gt_hr, pred_hr, SNR = calculate_metric_per_video(BVP_window, label_window, diff_flag=False,
                                                                fs=config.UNSUPERVISED.DATA.FS, hr_method=config.INFERENCE.EVALUATION_METHOD)
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
    
    print("Used Unsupervised Method: " + method_name)
    
    # print("GT HR: ", gt_hr_all)
    # print("Predict HR: ", predict_hr_all)

    # Filename ID to be used in any results files (e.g., Bland-Altman plots) that get saved
    if config.TOOLBOX_MODE == 'unsupervised_method':
        filename_id = method_name + "_" + config.UNSUPERVISED.DATA.DATASET
    else:
        raise ValueError('unsupervised_predictor.py evaluation only supports unsupervised_method!')

    predict_hr_all = np.array(predict_hr_all)
    gt_hr_all = np.array(gt_hr_all)
    SNR_all = np.array(SNR_all)
    num_test_samples = len(predict_hr_all)
    
    for metric in config.UNSUPERVISED.METRICS:
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
        
    save_test_outputs(predictions, labels, config, method_name)
    