import numpy as np
from matplotlib import pyplot as plt
import scipy
import util

def corr_relate_align(signal,gt,maxshift = 30):
    signal = util.normalization(signal)
    gt = util.normalization(gt)
    plt.figure()
    plt.plot(gt)
    plt.plot(signal)
    plt.show()

    print(np.correlate(gt, signal) / np.shape(gt))
    found = False
    plt.figure()
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
            maxV = np.correlate(gt[shift:], signal[:-shift]) / np.shape(gt[shift:])[0]
            final_shift = shift
        # else:
        #     found = True
        shift = shift + 1

    plt.plot(gt[final_shift:])
    plt.plot(signal[:-final_shift])
    plt.show()
    print(maxV)
    return signal[:-final_shift],gt[final_shift:]