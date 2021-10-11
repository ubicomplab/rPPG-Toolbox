import numpy as np
import cv2
from skimage.util import img_as_float
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import scipy.io
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import dlib
from scipy.sparse import spdiags

def preprocess_raw_video(videoFilePath, dim=36):

    # Set timer to calcuate time (ms)
    start_time = time.time()
    #########################################################################
    # set up
    t = []
    i = 0
    vidObj = cv2.VideoCapture(videoFilePath);
    totalFrames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT)) # get total frame size
    print('Total Frame: ', totalFrames)
    Xsub = np.zeros((totalFrames, dim, dim, 3), dtype = np.float32)
    height = vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vidObj.get(cv2.CAP_PROP_FRAME_WIDTH)
    success, img = vidObj.read()
    detector = dlib.get_frontal_face_detector()
    CROPPED_BORDER = 150
    cropped_cor = False
    #########################################################################
    # Crop each frame size into dim x dim
    while success:
        t.append(vidObj.get(cv2.CAP_PROP_POS_MSEC))# current timestamp in milisecond
        # vidLxL = cv2.resize(img_as_float(img[:, int(width/2)-int(height/2 + 1):int(height/2)+int(width/2), :]), (dim, dim), interpolation = cv2.INTER_AREA)
        # vidLxL = cv2.rotate(vidLxL, cv2.ROTATE_90_COUNTERCLOCKWISE) # rotate 90 degree
        # vidLxL = cv2.cvtColor(vidLxL.astype('float32'), cv2.COLOR_BGR2RGB)
        vidLxL = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) # rotate 90 degree
        vidLxL = cv2.cvtColor(vidLxL, cv2.COLOR_BGR2RGB)
        try:
            if cropped_cor == False:
                rect = detector(vidLxL, 1)
                (x, y, w, h) = rect_to_bb(rect[0])
                # cropped_cor = True
            cropped_img = vidLxL[y - CROPPED_BORDER:y + h + CROPPED_BORDER, x - CROPPED_BORDER:x + w + CROPPED_BORDER]
            vidLxL = cv2.resize(img_as_float(cropped_img), (dim, dim), interpolation = cv2.INTER_AREA).astype('float32')
        except Exception as e:
            vidLxL = cv2.resize(img_as_float(vidLxL[:, int(width/2)-int(height/2 + 1):int(height/2)+int(width/2), :]), (dim, dim), interpolation = cv2.INTER_AREA).astype('float32')
        vidLxL[vidLxL > 1] = 1
        vidLxL[vidLxL < (1/255)] = 1/255
        Xsub[i, :, :, :] = vidLxL
        success, img = vidObj.read() # read the next one
        i = i + 1
        print(i)
    #########################################################################
    # Normalized Frames in the motion branch
    normalized_len = len(t) - 1
    dXsub = np.zeros((normalized_len, dim, dim, 3), dtype = np.float32)
    for j in range(normalized_len - 1):
        dXsub[j, :, :, :] = (Xsub[j+1, :, :, :] - Xsub[j, :, :, :]) / (Xsub[j+1, :, :, :] + Xsub[j, :, :, :])
    dXsub = dXsub / np.std(dXsub)
    #########################################################################
    # Normalize raw frames in the apperance branch
    plt.imshow(Xsub[500])
    plt.title('Sample Frame')
    plt.show()
    # raise
    Xsub = Xsub - np.mean(Xsub)
    Xsub = Xsub  / np.std(Xsub) # / 2*std
    Xsub = Xsub[:totalFrames-1, :, :, :]
    # dXsub = dXsub[500:1500, :, :, :]
    # Xsub = Xsub[500:1500, :, :, :]
    #########################################################################
    # Plot an example of data after preprocess
    dXsub = np.concatenate((dXsub, Xsub), axis = 3);
    return dXsub


def detrend(signal, Lambda):
    """detrend(signal, Lambda) -> filtered_signal
    This function applies a detrending filter.
    This code is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
    *Parameters*
      ``signal`` (1d numpy array):
        The signal where you want to remove the trend.
      ``Lambda`` (int):
        The smoothing parameter.
    *Returns*
      ``filtered_signal`` (1d numpy array):
        The detrended signal.
    """
    signal_length = signal.shape[0]

    # observation matrix
    H = np.identity(signal_length)

    # second-order difference matrix

    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal
