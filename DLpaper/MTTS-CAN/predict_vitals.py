import tensorflow as tf
import numpy as np
import scipy.io
import os
import sys
import argparse
sys.path.append('../')
from model import Attention_mask, MTTS_CAN
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter
from inference_preprocess import preprocess_raw_video, detrend

def predict_vitals(args):
    img_rows = 36
    img_cols = 36
    frame_depth = 10
    model_checkpoint = './mtts_can.hdf5'
    batch_size = args.batch_size
    fs = args.sampling_rate
    sample_data_path = args.video_path

    dXsub = preprocess_raw_video(sample_data_path, dim=36)

    dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
    dXsub = dXsub[:dXsub_len, :, :, :]
    # print(dXsub.shape)
    # raise
    np.save('parker_normalized',(dXsub[:, :, :, :3]))
    np.save('parker_raw',(dXsub[:, :, :, -3:]))


    model = MTTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    model.load_weights(model_checkpoint)

    yptest = model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)

    pulse_pred = yptest[0]
    pulse_pred = detrend(np.cumsum(pulse_pred), 100)
    [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))

    resp_pred = yptest[1]
    resp_pred = detrend(np.cumsum(resp_pred), 100)
    [b_resp, a_resp] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')
    resp_pred = scipy.signal.filtfilt(b_resp, a_resp, np.double(resp_pred))

    ## Calculating HR
    N = 30 * fs
    pulse_fft = np.expand_dims(pulse_pred, 0)
    f, pxx = scipy.signal.periodogram(pulse_fft, fs=fs, nfft=4 * N, detrend=False)
    fmask = np.argwhere((f >= 0.75) & (f <= 2.5))  # regular Heart beat are 0.75*60 and 2.5*60
    frange = np.take(f, fmask)
    HR = np.take(frange, np.argmax(np.take(pxx, fmask), 0))[0] * 60

    ## Calculating RR
    resp_fft = np.expand_dims(resp_pred, 0)
    f, pxx = scipy.signal.periodogram(resp_fft, fs=fs, nfft=4 * N, detrend=False)
    fmask = np.argwhere((f >= 0.08) & (f <= 0.5))  # regular RR are 0.08*60 and 0.5*60
    frange = np.take(f, fmask)
    RR = np.take(frange, np.argmax(np.take(pxx, fmask), 0))[0] * 60

    np.save('pulse_pred.npy', pulse_pred)
    np.save('resp_pred.npy', resp_pred)
    ########## Plot ##################
    plt.subplot(211)
    plt.plot(pulse_pred)
    plt.title('Pulse Prediction - HR: {} BPM'.format(str(HR)))
    plt.subplot(212)
    plt.plot(resp_pred)
    plt.title('Resp Prediction - RR: {} BPM'.format(str(RR)))
    plt.show()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='processed video path')
    parser.add_argument('--sampling_rate', type=int, default = 30, help='sampling rate of your video')
    parser.add_argument('--batch_size', type=int, default = 100, help='batch size (multiplier of 10)')
    args = parser.parse_args()

    predict_vitals(args)

