""" Short functions for data-preprocessing and data-loading. """

import numpy as np
import cv2


def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()


def sample(a, len):
    """Samples a sequence into specific length."""
    return np.interp(
        np.linspace(
            1, a.shape[0], len), np.linspace(
            1, a.shape[0], a.shape[0]), a)


def facial_detection(frame, larger_box=False, larger_box_size=1.0):
    """Conducts face detection on a single frame.
    Sets larger_box=True for larger bounding box, e.g. moving trials."""
    detector = cv2.CascadeClassifier(
        './dataset/haarcascade_frontalface_default.xml')
    face_zone = detector.detectMultiScale(frame)
    if (len(face_zone) < 1):
        print("ERROR:No Face Detected")
        result = [0, 0, frame.shape[0], frame.shape[1]]
    elif (len(face_zone) >= 2):
        result = np.argmax(face_zone, axis=0)
        result = face_zone[result[2]]
        print("WARN:More than one faces are detected(Only cropping the biggest one.)")
    else:
        result = face_zone[0]
    if larger_box:
        print("Larger Bounding Box")
        result[0] = max(0, result[0] - (larger_box_size-1.0) / 2 * result[2])
        result[1] = max(0, result[1] - (larger_box_size-1.0) / 2 * result[3])
        result[2] = larger_box_size * result[2]
        result[3] = larger_box_size * result[3]
    return result


def resize(frames):
    face_region = facial_detection(frames[0], True, 1.6)
    print(face_region)
    frame = frames[0]
    temp_frame = frame[max(face_region[1], 0):min(face_region[1] + face_region[3], frame.shape[0]),
                       max(face_region[0], 0):min(face_region[0] + face_region[2], frame.shape[1])]
    print(temp_frame.shape)
    resize_frames = np.zeros(
        (frames.shape[0], temp_frame.shape[0], temp_frame.shape[1], 3))
    for i in range(0, frames.shape[0]):
        frame = frames[i]
        frame = frame[max(face_region[1], 0):min(face_region[1] + face_region[3], frame.shape[0]),
                      max(face_region[0], 0):min(face_region[0] + face_region[2], frame.shape[1])]
        resize_frames[i] = frame
    return resize_frames
