#!/usr/bin/env python

from __future__ import division, print_function

from keras import backend as K


def soft_sorensen_dice(y_true, y_pred, axis=None, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=axis)
    area_true = K.sum(y_true, axis=axis)
    area_pred = K.sum(y_pred, axis=axis)
    return (2 * intersection + smooth) / (area_true + area_pred + smooth)
    
def hard_sorensen_dice(y_true, y_pred, axis=None, smooth=1):
    y_true_int = K.round(y_true)
    y_pred_int = K.round(y_pred)
    return soft_sorensen_dice(y_true_int, y_pred_int, axis, smooth)

sorensen_dice = hard_sorensen_dice

def sorensen_dice_loss(y_true, y_pred, weights):
    # Input tensors have shape (batch_size, height, width, classes)
    # User must input list of weights with length equal to number of classes
    #
    # Ex: for simple binary classification, with the 0th mask
    # corresponding to the background and the 1st mask corresponding
    # to the object of interest, we set weights = [0, 1]
    batch_dice_coefs = soft_sorensen_dice(y_true, y_pred, axis=[1, 2])
    dice_coefs = K.mean(batch_dice_coefs, axis=0)
    w = K.constant(weights)
    return -K.sum(w * dice_coefs)

def soft_jaccard(y_true, y_pred, axis=None, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=axis)
    area_true = K.sum(y_true, axis=axis)
    area_pred = K.sum(y_pred, axis=axis)
    union = area_true + area_pred - intersection
    return (intersection + smooth) / (union + smooth)

def hard_jaccard(y_true, y_pred, axis=None, smooth=1):
    y_true_int = K.round(y_true)
    y_pred_int = K.round(y_pred)
    return soft_jaccard(y_true_int, y_pred_int, axis, smooth)

jaccard = hard_jaccard

def jaccard_loss(y_true, y_pred, weights):
    batch_jaccard_coefs = soft_jaccard(y_true, y_pred, axis=[1, 2])
    jaccard_coefs = K.mean(batch_jaccard_coefs, axis=0)
    w = K.constant(weights)
    return -K.sum(w * jaccard_coefs)
