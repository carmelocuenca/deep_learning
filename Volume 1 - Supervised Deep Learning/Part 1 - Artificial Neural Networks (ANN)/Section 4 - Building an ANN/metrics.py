#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import keras.backend as K
import numpy as np


###############################################################################
def precision(y_true, y_pred):
    #y = K.round(y_pred)
    y = y_pred
    TP = K.sum(y_true*y)
    FP = K.sum((1-y_true)*y)
    return K.switch(TP+FP>0., TP/(TP+FP), 1.)

def recall(y_true, y_pred):
    #y = K.round(y_pred)
    y = y_pred
    TP = K.sum(y_true*y)
    FN = K.sum(y_true*(1.-y))
    return K.switch(TP+FN>0., TP/(TP+FN), 1.)
###############################################################################
def DL2(y_true, y_pred):
    y = y_pred
    TP = K.sum(y_true*y)
    TN = K.sum((1-y_true)*(1-y_pred))
    NPy = K.sum(y); NNy = K.sum(1.-y)
    NPy_true = K.sum(y_true); NNy_true = K.sum(1.-y_true)
    dl2 = 1. - (TP + K.epsilon())/(NPy + NPy_true + K.epsilon()) - (TN + K.epsilon())/( NNy+NNy_true+K.epsilon())
    return dl2
###############################################################################
def dsc(y_true, y_pred):
    #y = K.round(y_pred)
    y = y_pred
    TP = K.sum(y_true*y)
    FN = K.sum(y_true*(1.-y))
    FP = K.sum((1-y_true)*y)
    return K.switch(FN+TP+FP>0., TP/(FN+TP+FP), 1.)

def dsc2(y_true, y_pred):
    return 1. - dsc(y_true, y_pred)
###############################################################################

###############################################################################
# weighted_binary_crossentropy & weighted_binary_accuracy DEFINITIONS
###############################################################################

def compute_binary_class_weight(y):
     count_params = K.shape(y)[1]*K.shape(y)[2]
     nb_ones = K.sum(K.reshape(y, (-1, count_params)), axis=1)
     
     count_params = K.cast(count_params, 'float32')
     nb_zeros = count_params - nb_ones
     
     # To manage empty classes
     m = K.cast(K.equal(nb_ones, 0), dtype=K.floatx()) 
     nb_ones = m*count_params/2. + (1.-m)*nb_ones
     nb_zeros = m*count_params/2. + (1.-m)*nb_zeros
     
     one_weight = count_params/2.0/nb_ones
     zero_weight = count_params/2.0/nb_zeros
     return zero_weight, one_weight
     
def create_weighted_binary_crossentropy(wc=64.):
     def weighted_binary_crossentropy(y_true, y_pred):
          # Original binary crossentropy (see losses.py):
          # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

          # Calculate the binary crossentropy
          b_ce = K.binary_crossentropy(y_true, y_pred)

          filter = np.array([-1., -1., -1, -1., 8., -1, -1., -1., -1.]).reshape(3, 3, 1, 1)
          filter *= wc;
          filter = K.constant(filter)

          conv2d = K.conv2d(y_true, filter, padding='same')
          clip = K.clip(conv2d, 0., wc)
          clip_1 = -1.*K.clip(conv2d, -wc, 0)

          zero_weight, one_weight = compute_binary_class_weight(y_true)
          zero_weight = K.reshape(zero_weight, (-1, 1, 1, 1))
          one_weight = K.reshape(one_weight, (-1, 1, 1, 1))
          
          weight_vector = one_weight*y_true + (1. - y_true) * zero_weight
          weighted_b_ce = weight_vector * b_ce

          return K.mean(weighted_b_ce  + y_pred*clip_1 + (1.-y_pred)*clip)
     return weighted_binary_crossentropy


def create_weighted_binary_accuracy():
    def weighted_binary_accuracy(y_true, y_pred):
        # Apply the weights
         zero_weight, one_weight = compute_binary_class_weight(y_true)
         zero_weight = K.reshape(zero_weight, (-1, 1, 1, 1))
         one_weight = K.reshape(one_weight, (-1, 1, 1, 1))
          
         weight_vector = one_weight*y_true + (1. - y_true) * zero_weight
         return K.mean(weight_vector*K.cast(K.equal(y_true, K.round(y_pred)), dtype=K.floatx()))
    return weighted_binary_accuracy

