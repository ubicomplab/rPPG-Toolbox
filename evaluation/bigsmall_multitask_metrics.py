import numpy as np
import scipy
import scipy.io
from scipy.signal import butter
from sklearn.metrics import f1_score, precision_recall_fscore_support
from evaluation.metrics import calculate_metrics, _reform_data_from_dict, calculate_resp_metrics
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