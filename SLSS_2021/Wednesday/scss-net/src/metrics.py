import numpy as np
from tensorflow.keras import backend as K

"""
Metrics used to validate model. 
"""


def dice_np(y_true, y_pred, smooth=1.0):
    intersection = np.sum(y_true * y_pred)
    return 2 * (intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)


def iou_np(y_true, y_pred, smooth=1.0):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true + y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


def iou(y_true, y_pred, smooth=1.0):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true + y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


def dice(y_true, y_pred, smooth=1.0):
    intersection = K.sum(y_true * y_pred)
    return (2 * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
