import scipy
import matplotlib.pyplot as plt
import scipy.signal
from utils.utils import *

def fft(frames,fs):
    N = next_power_of_2(frames.shape[0])
    f, pxx = scipy.signal.periodogram(
        frames, fs=fs, nfft=N, detrend=False)
    fmask = np.argwhere((f >= 0.75) & (f <= 2.5))
    f, pxx = np.take(f, fmask), np.take(pxx, fmask)
    return f, pxx

def green_channel(video, bvp, plot_flag=True):
    """
    Args:
        video: T*W*H*3. T-Time, W-Frame Width, H-Frame Height
    """
    green_channel = video[:, :, :, 1]
    green_channel_mean = green_channel.mean(axis=(1, 2))
    fs = 20
    b, a = scipy.signal.butter(1, [1.5 / fs, 5 / fs], 'bandpass')
    # green_channel_mean = scipy.signal.filtfilt(b, a, green_channel_mean)

    f,pxx = fft(green_channel,fs)
    bvp_f,bvp_pxx = fft(bvp,fs)

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
        plt.plot(bvp_f,np.reshape(bvp_pxx, [-1, 1]))
        plt.title('Filted')
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


# def gc_and_align(frames, bvps):
#
#     greenchannel, bvp = green_channel(frames, bvps)
# # # downsample
#     greenchannel_ds = down_sample(greenchannel, bvp.shape[0])
# # # do align
#     final_shift = align.corr_relate_align(greenchannel_ds, bvp)
#     frames = frames[:-final_shift]
#     bvps = bvps[final_shift:]
#     green_channel(frames, bvps)
#     return frames, bvps
