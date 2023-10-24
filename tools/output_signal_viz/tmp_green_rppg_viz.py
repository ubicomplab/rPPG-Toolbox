import numpy as np
import cv2
import os

import math
from scipy import signal
from scipy import linalg

import matplotlib.pyplot as plt


import math

import cv2
import numpy as np
from scipy import io as scio
from scipy import linalg
from scipy import signal
from scipy import sparse
from skimage.util import img_as_float
from sklearn.metrics import mean_squared_error


def detrend(input_signal, lambda_value):
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return filtered_signal


# Process Video
def process_video(frames):
    RGB = []
    for frame in frames:
        summation = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(summation / (frame.shape[0] * frame.shape[1]))
    RGB = np.asarray(RGB)
    RGB = RGB.transpose(1, 0).reshape(1, 3, -1)
    return np.asarray(RGB)


# Green Channel rPPG
def GREEN(frames):
    precessed_data = process_video(frames)
    BVP = precessed_data[:, 1, :]
    BVP = BVP.reshape(-1)
    return BVP


# Face Detection
def face_detection(frame, use_larger_box=False, larger_box_coef=1.0):
    """Face detection on a single frame.

    Args:
        frame(np.array): a single frame.
        use_larger_box(bool): whether to use a larger bounding box on face detection.
        larger_box_coef(float): Coef. of larger box.
    Returns:
        face_box_coor(List[int]): coordinates of face bouding box.
    """

    detector = cv2.CascadeClassifier(
        '../../dataset/haarcascade_frontalface_default.xml')
    face_zone = detector.detectMultiScale(frame)
    if len(face_zone) < 1:
        print("ERROR: No Face Detected")
        face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
    elif len(face_zone) >= 2:
        face_box_coor = np.argmax(face_zone, axis=0)
        face_box_coor = face_zone[face_box_coor[2]]
        print("Warning: More than one faces are detected(Only cropping the biggest one.)")
    else:
        face_box_coor = face_zone[0]
    if use_larger_box:
        face_box_coor[0] = max(0, face_box_coor[0] - (larger_box_coef - 1.0) / 2 * face_box_coor[2])
        face_box_coor[1] = max(0, face_box_coor[1] - (larger_box_coef - 1.0) / 2 * face_box_coor[3])
        face_box_coor[2] = larger_box_coef * face_box_coor[2]
        face_box_coor[3] = larger_box_coef * face_box_coor[3]
    return face_box_coor


# Face Crop
def crop_face_resize(frames, use_face_detection, use_larger_box, larger_box_coef, use_dynamic_detection, 
                     detection_freq, use_median_box, width, height):
    """Crop face and resize frames.

    Args:
        frames(np.array): Video frames.
        use_dynamic_detection(bool): If False, all the frames use the first frame's bouding box to crop the faces
                                        and resizing.
                                        If True, it performs face detection every "detection_freq" frames.
        detection_freq(int): The frequency of dynamic face detection e.g., every detection_freq frames.
        width(int): Target width for resizing.
        height(int): Target height for resizing.
        use_larger_box(bool): Whether enlarge the detected bouding box from face detection.
        use_face_detection(bool):  Whether crop the face.
        larger_box_coef(float): the coefficient of the larger region(height and weight),
                            the middle point of the detected region will stay still during the process of enlarging.
    Returns:
        resized_frames(list[np.array(float)]): Resized and cropped frames
    """
    # Face Cropping
    if use_dynamic_detection:
        num_dynamic_det = math.ceil(frames.shape[0] / detection_freq)
    else:
        num_dynamic_det = 1
    face_region_all = []
    # Perform face detection by num_dynamic_det" times.
    for idx in range(num_dynamic_det):
        if use_face_detection:
            face_region_all.append(face_detection(frames[detection_freq * idx], use_larger_box, larger_box_coef))
        else:
            face_region_all.append([0, 0, frames.shape[1], frames.shape[2]])
    face_region_all = np.asarray(face_region_all, dtype='int')
    if use_median_box:
        # Generate a median bounding box based on all detected face regions
        face_region_median = np.median(face_region_all, axis=0).astype('int')


    # Frame Resizing
    resized_frames = np.zeros((frames.shape[0], height, width, 3))
    for i in range(0, frames.shape[0]):
        frame = frames[i]
        if use_dynamic_detection:  # use the (i // detection_freq)-th facial region.
            reference_index = i // detection_freq
        else:  # use the first region obtrained from the first frame.
            reference_index = 0
        if use_face_detection:
            if use_median_box:
                face_region = face_region_median
            else:
                face_region = face_region_all[reference_index]
            frame = frame[max(face_region[1], 0):min(face_region[1] + face_region[3], frame.shape[0]),
                    max(face_region[0], 0):min(face_region[0] + face_region[2], frame.shape[1])]
        resized_frames[i] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return resized_frames


# Read Video
def read_video(video_file):
    """Reads a video file, returns frames(T,H,W,3) """
    VidObj = cv2.VideoCapture(video_file)
    VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
    success, frame = VidObj.read()

    # Check if the video file has been opened successfully
    if not VidObj.isOpened():
        print("Error: Could not open video.")
        exit()

    frames = list()
    while success:
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
        frame = np.asarray(frame)
        frames.append(frame)
        success, frame = VidObj.read()
    return np.asarray(frames)

# Apply Face Cropping

data_dir = '/gscratch/ubicomp/girishvn/rppg/rppg_datasets/TestData'
fname = 'girish1.avi'  # girish1.avi  ham1.avi

data_path = os.path.join(data_dir, fname)
print('Video Path:', data_path)

# Read in data
print('Reading video...')
frames = read_video(data_path)

print('Done Reading Video')
print(frames.shape)

frames = crop_face_resize(
    frames,
    True,
    True,
    1.5,
    True,
    180,
    True,
    72,
    72)

green_ppg = GREEN(frames)
green_ppg = detrend(green_ppg, 100)

fs=24
min_freq = 0.70
max_freq = 3
b, a = signal.butter(2, [(min_freq) / fs * 2, (max_freq) / fs * 2], btype='bandpass')
green_ppg_filtered = signal.filtfilt(b, a, green_ppg.astype(np.double))

print('PPG Shape', green_ppg.shape)

fs = 30

plt.figure()
plt.plot(green_ppg[30*60:30*66])
plt.plot(green_ppg_filtered[30*60:30*66])
plt.savefig('./girish_green_ppg.png')


