import numpy as np
from matplotlib import pyplot as plt
import scipy


def normalization(data):
    '''normalize the data into [-1,1]'''
    _range = np.max(abs(data))
    return data / _range


def corr_relate_align(signal, gt, maxshift=-1):
    '''do align'''
    signal = normalization(signal)
    gt = normalization(gt)
    if(maxshift <= 0):
        peaks, _ = scipy.signal.find_peaks(gt)
        maxshift = peaks[1]-peaks[0]
    print(np.correlate(gt, signal) / np.shape(gt))
    shift = 1
    maxV = -100
    final_shift = 1
    while (True):
        if(shift > maxshift):
            break
        # plt.plot(gt[shift:])
        # plt.plot(signal[:-shift])
        # plt.show()
        if np.correlate(gt[shift:], signal[:-shift]) / np.shape(gt[shift:])[0] > maxV:
            maxV = np.correlate(gt[shift:], signal[:-shift]
                                ) / np.shape(gt[shift:])[0]
            final_shift = shift
        # else:
        #     found = True
        shift = shift + 1
    print(maxshift)
    print(final_shift)
    return final_shift
