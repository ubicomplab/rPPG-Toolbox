"""Ops for data preprocessing.

Provide a serires of implmentation for data preprocessing such as resize, cropping, normalization etc.
"""
import numpy as np
import os
import cv2


def crop(data):
    pass


def resize(data):
    pass


def diff_data(data, axis=0):
    """Differentiate in data"""
    _, c, h, w = data.shape
    pad_tensor = np.zeros([1, c, h, w])
    data = np.concatenate((data, pad_tensor), axis=axis)  # keep the same dimension
    data = np.diff(data, axis=axis)
    return data


def diff_label(label, axis=0):
    """Differentiate in label"""
    pad_tensor = np.zeros([1, 1])
    label = np.concatenate((label, pad_tensor), axis=axis)  # keep the same dimension
    label = np.diff(label, axis=axis)
    return label


def normalize_data(data):
    pass


def normalize_label(label):
    pass
