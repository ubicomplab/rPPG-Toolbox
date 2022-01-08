import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy


def down_sample(a, len):
    '''downsample'''
    return np.interp(
        np.linspace(
            1, a.shape[0], len), np.linspace(
            1, a.shape[0], a.shape[0]), a)


def next_power_of_2(x):
    '''get the signal with the length of 2^n'''
    return 1 if x == 0 else 2**(x - 1).bit_length()


def green_channel(video, bvp, face_region=None, plot_flag=False):
    '''get green channel signals'''
    imCrop = video[0]
    # r = cv2.selectROI("face", sample_image)  # Press enter after selecting box
    # print('Coordiantes: ', r)
    # imCrop = sample_image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    # cv2.imshow("Image", imCrop)
    # cv2.imwrite("save.bmp",imCrop)
    # cv2.waitKey(0)  # Press enter again to close both windows
    # cv2.destroyWindow("face")
    # cv2.destroyWindow("Image")

    # Region of interest

    print('Start processing ROI')
    length = video.shape[0]
    roi_data = []
    for i in range(length):
        im = video[i]
        if (face_region):
            imCrop = im[int(face_region[1]):int(face_region[1] + face_region[3]),
                        int(face_region[0]):int(face_region[0] + face_region[2]), :]
        roi_data.append(imCrop)
    roi_data = np.array(roi_data)
    print('roi shape', roi_data.shape)

    green_channel = roi_data[:, :, :, 1]
    green_channel_mean = green_channel.mean(axis=(1, 2))
    fs = 30
    b, a = signal.butter(1, [1.5 / fs, 5 / fs], 'bandpass')
    green_channel_mean = signal.filtfilt(b, a, green_channel_mean)
    N = next_power_of_2(green_channel_mean.shape[0])
    f, pxx = scipy.signal.periodogram(
        green_channel_mean, fs=fs, nfft=N, detrend=False)
    fmask = np.argwhere((f >= 0.75) & (f <= 2.5))
    f, pxx = np.take(f, fmask), np.take(pxx, fmask)
    if(plot_flag):
        plt.subplot(411)
        plt.plot(bvp)
        plt.title('BVP')
        plt.subplot(412)
        plt.plot(green_channel_mean)
        plt.title('Green Channel')
        plt.subplot(413)
        plt.plot(f, np.reshape(pxx, [-1, 1]))
        plt.title('FFT')
        plt.subplot(414)
        # plt.plot(filted_signal)
        # plt.title('Filted')
        plt.savefig('green_channel.png')
    return green_channel_mean, bvp

# ecg = (util.ECG_from_mat("IPhys_data/ECG.mat"))
# t,rgb = (util.process_video("Iphys_data/video_example.mp4",0,5))
# cv2.imwrite("save.bmp",rgb[0])

#
# # read the video
# videodata = skvideo.io.vread("video001.avi")
# # read the ground-true ecg
# ecg = util.read_wave("wave.csv")
# # get green channel signals


def gc_and_align(frames, bvps):
    ''' get green channel signals and do align'''
    greenchannel, bvp = green_channel(frames, bvps)
    # # downsample
    greenchannel_ds = down_sample(greenchannel, bvp.shape[0])
    # # do align
    final_shift = align.corr_relate_align(greenchannel_ds, bvp)
    frames = frames[:-final_shift]
    bvps = bvps[final_shift:]
    green_channel(frames, bvps)
    return frames, bvps
