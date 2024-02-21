"""The post processing files for caluclating heart rate.
The file also  includes helper funcs such as detrend, mag2db etc.
"""

import numpy as np
import scipy
import scipy.io
from scipy.signal import butter
from scipy.sparse import spdiags

from scipy.interpolate import Akima1DInterpolator


def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def _detrend(input_signal, lambda_value):
    """Detrend PPG signal."""
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal

def compute_power_spectrum(signal, fs, zero_pad=None):
    if zero_pad is not None:
        L = len(signal)
        signal = np.pad(signal, (int(zero_pad/2*L), int(zero_pad/2*L)), 'constant')
    freqs = np.fft.fftfreq(len(signal), 1 / fs) * 60  # in bpm
    ps = np.abs(np.fft.fft(signal))**2
    cutoff = len(freqs)//2
    freqs = freqs[:cutoff]
    ps = ps[:cutoff]
    return freqs, ps


def _calculate_hr(signal, fs, min_hr=40., max_hr=180., method='fast_ideal'):
    if method == 'Ideal':
        """ Zero-pad in time domain for ideal interp in freq domain
        """
        signal = signal - np.mean(signal)
        freqs, ps = compute_power_spectrum(signal, fs, zero_pad=100)
        cs = Akima1DInterpolator(freqs, ps)
        max_val = -np.Inf
        interval = 0.1
        min_bound = max(min(freqs), min_hr)
        max_bound = min(max(freqs), max_hr) + interval
        for bpm in np.arange(min_bound, max_bound, interval):
            cur_val = cs(bpm)
            if cur_val > max_val:
                max_val = cur_val
                max_bpm = bpm
        return max_bpm
    elif method == 'FastIdeal':
        """ Zero-pad in time domain for ideal interp in freq domain
        """
        signal = signal - np.mean(signal)
        freqs, ps = compute_power_spectrum(signal, fs, zero_pad=100)
        freqs_valid = np.logical_and(freqs >= min_hr, freqs <= max_hr)
        freqs = freqs[freqs_valid]
        ps = ps[freqs_valid]
        max_ind = np.argmax(ps)
        if 0 < max_ind < len(ps)-1:
            inds = [-1, 0, 1] + max_ind
            x = ps[inds]
            f = freqs[inds]
            d1 = x[1]-x[0]
            d2 = x[1]-x[2]
            offset = (1 - min(d1,d2)/max(d1,d2)) * (f[1]-f[0])
            if d2 > d1:
                offset *= -1
            max_bpm = f[1] + offset
        elif max_ind == 0:
            x0, x1 = ps[0], ps[1]
            f0, f1 = freqs[0], freqs[1]
            max_bpm = f0 + (x1 / (x0 + x1)) * (f1 - f0)
        elif max_ind == len(ps) - 1:
            x0, x1 = ps[-2], ps[-1]
            f0, f1 = freqs[-2], freqs[-1]
            max_bpm = f0 + (x1 / (x0 + x1)) * (f1 - f0)
        return max_bpm
    elif method == 'FastIdealBimodalFilter':
        """ Same as above but check for secondary peak around 1/2 of first
        (to break the tie in case of occasional bimodal PS)
        Note - this may make metrics worse if the power spectrum is relatively flat
        """
        signal = signal - np.mean(signal)
        freqs, ps = compute_power_spectrum(signal, fs, zero_pad=100)
        freqs_valid = np.logical_and(freqs >= min_hr, freqs <= max_hr)
        freqs = freqs[freqs_valid]
        ps = ps[freqs_valid]
        max_ind = np.argmax(ps)
        max_freq = freqs[max_ind]
        max_ps = ps[max_ind]

        # check for a second lower peak at 0.45-0.55f and >50% power
        freqs_valid = np.logical_and(freqs >= max_freq * 0.45, freqs <= max_freq * 0.55)
        freqs = freqs[freqs_valid]
        ps = ps[freqs_valid]
        if len(freqs) > 0:
            max_ind_lower = np.argmax(ps)
            max_freq_lower = freqs[max_ind_lower]
            max_ps_lower = ps[max_ind_lower]
        else:
            max_ps_lower = 0

        if max_ps_lower / max_ps > 0.50:
            return max_freq_lower
        else:
            return max_freq
    elif method == 'FFT':
        """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
        low_pass = min_hr / 60
        high_pass = max_hr / 60
        
        signal = np.expand_dims(signal, 0)
        N = _next_power_of_2(signal.shape[1])
        f_ppg, pxx_ppg = scipy.signal.periodogram(signal, fs=fs, nfft=N, detrend=False)
        fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
        mask_ppg = np.take(f_ppg, fmask_ppg)
        mask_pxx = np.take(pxx_ppg, fmask_ppg)
        fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
        return fft_hr
    elif method == 'Peak':
        """Calculate heart rate based on PPG using peak detection."""
        ppg_peaks, _ = scipy.signal.find_peaks(signal)
        hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
        return hr_peak
    else:
        raise ValueError('Unsupported method for HR calculation.')


def mag2db(mag):
    """Convert magnitude to db."""
    return 20. * np.log10(mag)

def _calculate_SNR(pred_ppg_signal, hr_label, fs=30, low_pass=0.75, high_pass=2.5):
    """Calculate SNR as the ratio of the area under the curve of the frequency spectrum around the first and second harmonics 
        of the ground truth HR frequency to the area under the curve of the remainder of the frequency spectrum, from 0.75 Hz
        to 2.5 Hz. 

        Args:
            pred_ppg_signal(np.array): predicted PPG signal 
            label_ppg_signal(np.array): ground truth, label PPG signal
            fs(int or float): sampling rate of the video
        Returns:
            SNR(float): Signal-to-Noise Ratio
    """
    # Get the first and second harmonics of the ground truth HR in Hz
    first_harmonic_freq = hr_label / 60
    second_harmonic_freq = 2 * first_harmonic_freq
    deviation = 6 / 60  # 6 beats/min converted to Hz (1 Hz = 60 beats/min)

    # Calculate FFT
    pred_ppg_signal = np.expand_dims(pred_ppg_signal, 0)
    N = _next_power_of_2(pred_ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(pred_ppg_signal, fs=fs, nfft=N, detrend=False)

    # Calculate the indices corresponding to the frequency ranges
    idx_harmonic1 = np.argwhere((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation)))
    idx_harmonic2 = np.argwhere((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation)))
    idx_remainder = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass) \
     & ~((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation))) \
     & ~((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation))))

    # Select the corresponding values from the periodogram
    pxx_ppg = np.squeeze(pxx_ppg)
    pxx_harmonic1 = pxx_ppg[idx_harmonic1]
    pxx_harmonic2 = pxx_ppg[idx_harmonic2]
    pxx_remainder = pxx_ppg[idx_remainder]

    # Calculate the signal power
    signal_power_hm1 = np.sum(pxx_harmonic1)
    signal_power_hm2 = np.sum(pxx_harmonic2)
    signal_power_rem = np.sum(pxx_remainder)

    # Calculate the SNR as the ratio of the areas
    if not signal_power_rem == 0: # catches divide by 0 runtime warning 
        SNR = mag2db((signal_power_hm1 + signal_power_hm2) / signal_power_rem)
    else:
        SNR = 0
        
    return SNR


def calculate_metric_per_video(predictions, labels, fs=30, diff_flag=True, use_bandpass=True, hr_method='FFT'):
    """Calculate video-level HR and SNR"""
    if diff_flag:  # if the predictions and labels are 1st derivative of PPG signal.
        predictions = _detrend(np.cumsum(predictions), 100)
        labels = _detrend(np.cumsum(labels), 100)
    else:
        predictions = _detrend(predictions, 100)
        labels = _detrend(labels, 100)

    # bandpass filter between [0.75, 2.5] Hz
    # equals [45, 150] beats per min
    low_pass = 0.75
    high_pass = 2.5
    
    if use_bandpass:
        [b, a] = butter(1, [low_pass / fs * 2, high_pass / fs * 2], btype='bandpass')
        predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
        labels = scipy.signal.filtfilt(b, a, np.double(labels))
        
    hr_pred = _calculate_hr(predictions, fs=fs, method=hr_method)
    hr_label = _calculate_hr(labels, fs=fs, method=hr_method)
    SNR = _calculate_SNR(predictions, hr_label, fs=fs, low_pass=low_pass, high_pass=high_pass)
    
    return hr_label, hr_pred, SNR