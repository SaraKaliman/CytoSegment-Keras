# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:27:58 2023

@author: skalima
"""
from keras import backend as K
from tensorflow import where, cast, float32

def recall(y_true, y_pred):
    #flatten label and prediction tensors
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    # Treshold Tensor output on value 0.5
    y_pred = where(y_pred>0.5, 1, 0)
    # Cast output Tensor from int32 to float32
    y_pred = cast(y_pred, dtype=float32)

    #True Positives, False Positives & False Negatives
    TP = K.sum((y_true * y_pred))
    FN = K.sum(((1-y_pred) * y_true))

    recall = TP/ (TP + FN + K.epsilon())
    return recall

def precision(y_true, y_pred):
    #flatten label and prediction tensors
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    # Treshold Tensor output on value 0.5
    y_pred = where(y_pred>0.5, 1, 0)
    # Cast output Tensor from int32 to float32
    y_pred = cast(y_pred, dtype=float32)

    #True Positives, False Positives & False Negatives
    TP = K.sum((y_true * y_pred))
    FP = K.sum((y_pred * (1-y_true)))

    precision = TP/ (TP + FP + K.epsilon())
    return precision

def dice(y_true, y_pred):
    #flatten label and prediction tensors
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    # Treshold Tensor output on value 0.5
    y_pred = where(y_pred>0.5, 1, 0)
    # Cast output Tensor from int32 to float32
    y_pred = cast(y_pred, dtype=float32)

    #True Positives, False Positives & False Negatives
    TP = K.sum((y_true * y_pred))
    FN = K.sum(((1-y_pred) * y_true))
    FP = K.sum((y_pred * (1-y_true)))

    Dice = 2*TP/ (2*TP + FP + FN + K.epsilon())
    return Dice

def IoU(y_true, y_pred):
    #flatten label and prediction tensors
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    # Treshold Tensor output on value 0.5
    y_pred = where(y_pred>0.5, 1, 0)
    # Cast output Tensor from int32 to float32
    y_pred = cast(y_pred, dtype=float32)

    #True Positives, False Positives & False Negatives
    TP = K.sum((y_true * y_pred))
    FN = K.sum(((1-y_pred) * y_true))
    FP = K.sum((y_pred * (1-y_true)))

    IoU = TP/ (TP + FP + FN + K.epsilon())
    return IoU

# Raghava: alpha: 0.3, beta: 0.7, gamma: 0.75
def FocalTverskyLoss(y_true, y_pred):
    alpha = 0.5
    gamma = 1/1.5

    #flatten label and prediction tensors
    y_true = K.flatten(y_true)
    y_true = cast(y_true, dtype=float32)
    y_pred = K.flatten(y_pred)

    #True Positives, False Positives & False Negatives
    TP = K.sum((y_true * y_pred))
    FN = K.sum(((1-y_pred) * y_true))
    FP = K.sum((y_pred * (1-y_true)))

    Tversky = (TP + K.epsilon()) / (TP + alpha*FP + (1-alpha)*FN + K.epsilon())
    FocalTversky = K.pow((1 - Tversky), gamma)

    return FocalTversky
