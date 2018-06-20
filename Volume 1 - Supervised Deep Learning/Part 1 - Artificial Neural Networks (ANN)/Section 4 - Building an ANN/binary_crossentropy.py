#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 11:09:30 2018

@author: carmelo.cuenca
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras


y_true = np.array([
# open 2, close 0
                     [[0., 0., 0., 0., 0., 0., 0.],
                     [0., 1., 0., 0., 0., 1., 0.],
                     [0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 1., 0., 0., 0.],
                     [0., 1., 0., 0., 0., 1., 0.],
                     [0., 0., 1., 1., 1., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0.]],
# open 1, close 1
                     [[0., 0., 0., 0., 0., 0., 0.],
                     [0., 1., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 1., 0., 0., 0.],
                     [0., 1., 0., 0., 0., 1., 0.],
                     [0., 0., 1., 1., 1., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0.]],
# open 0, close 2
                     [[0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0.]],
                     ])

y_true = y_true.reshape(*y_true.shape, 1).astype(np.float32)

y_pred = np.array([
# open 2, close 0
                     [[1., 0., 0., 0., 0., 0., 0.],
                     [0., 1., 0., 0., 0., 1., 0.],
                     [0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 1., 0., 0., 0.],
                     [0., 1., 0., 0., 0., 1., 0.],
                     [0., 0., 1., 1., 1., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0.]],
# open 1, close 1
                     [[0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 1., 0., 0., 0.],
                     [0., 1., 0., 0., 0., 1., 0.],
                     [0., 0., 1., 1., 1., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0.]],
# open 0, close 2
                     [[0., 0., 0., 0., 0., 0., 0.],
                     [0., 1., 0., 0., 0., 1., 0.],
                     [0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 1., 0., 0., 0.],
                     [0., 1., 0., 0., 0., 1., 0.],
                     [0., 0., 1., 1., 1., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0.]],
                     ])

y_pred = y_pred.reshape(*y_pred.shape, 1).astype(np.float32)


#from sklearn.utils import class_weight
#class_weightx = class_weight.compute_class_weight('balanced', [0., 1.],y_true.flatten())
#print(class_weightx)
#
#
#class_weightx = class_weight.compute_class_weight('balanced', [0., 1.],y_true[0].flatten())
#print(class_weightx)
#
#class_weightx = class_weight.compute_class_weight('balanced', [0., 1.],y_true[1].flatten())
#print(class_weightx)
#
#
#class_weightx = class_weight.compute_class_weight('balanced', [0., 1.],y_true[2].flatten())
#print(class_weightx)


#######################
import keras.backend as K
import numpy as np

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

a_k = K.variable(y_true)
b_k = K.variable(y_pred)

myloss = create_weighted_binary_crossentropy(wc=10.)
value = K.eval(myloss(a_k, b_k))

print(value, value.shape)

############################################################################################3
import keras.backend as K

def create_weighted_binary_accuracy(zero_weight, one_weight):
    def weighted_binary_accuracy(y_true, y_pred):
        # Apply the weights
         zero_weight, one_weight = compute_binary_class_weight(y_true)
         zero_weight = K.reshape(zero_weight, (-1, 1, 1, 1))
         one_weight = K.reshape(one_weight, (-1, 1, 1, 1))
          
         weight_vector = one_weight*y_true + (1. - y_true) * zero_weight
         return K.mean(weight_vector*K.cast(K.equal(y_true, K.round(y_pred)), dtype=K.floatx()))
    return weighted_binary_accuracy


y_true = np.array([
                     [[0., 0., 0., 0., 0., 0., 0.],
                     [0., 1., 0., 0., 0., 1., 0.],
                     [0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 1., 0., 0., 0.],
                     [0., 1., 0., 0., 0., 1., 0.],
                     [0., 0., 1., 1., 1., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0.]],
                     ])

y_true = np.array([[0., 0., 0., 1]])
y_true = y_true.reshape(*y_true.shape, 1).astype(np.float32)

y_pred = np.array([
                     [[0., 0., 0., 0., 0., 0., 0.],
                     [0., 1., 0., 0., 0., 1., 0.],
                     [0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 1., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 1., 1., 1., 0., 0.],
                     [0., 1., 0., 0., 0., 1., 0.]],
                     ])

y_pred = np.array([[1., 1., 1., 1]])
y_pred = y_pred.reshape(*y_pred.shape, 1).astype(np.float32)


from sklearn.utils import class_weight
class_weight = class_weight.compute_class_weight('balanced', [0., 1.],y_true.flatten())
print(class_weight)


a_k = K.variable(y_true)
b_k = K.variable(y_pred)


b_acc = create_weighted_binary_accuracy(1., 1.)
print(K.eval(b_acc(y_true, y_pred)))

w_acc = create_weighted_binary_accuracy(class_weight[0], class_weight[1])
print(K.eval(w_acc(y_true, y_pred)))
















EPS=1.E-7
from scipy import signal
def entropia2(y_true, y_pred, w0, w1, wc=10):
    y_pred = np.clip(y_pred, EPS, 1.-EPS)
    b_ce = -((1-y_true)*np.log(1.-y_pred) + y_true*np.log(y_pred))
    weight_vector = y_true * w1 + (1. - y_true) * w0
    weighted_b_ce = weight_vector * b_ce
    
    filter = np.array([-1., -1., -1, -1., 8., -1, -1., -1., -1.]).reshape(3, 3, 1, 1)
    conv2d = signal.convolve2d(y_true.reshape(7,7), filter.reshape(3,3), mode='same')
    conv2d = np.clip(conv2d, 0, 1).reshape(*weighted_b_ce.shape)
    conv2d *= wc
    return np.mean(weighted_b_ce+conv2d)


for i in range(0, 100):
    y_pred = np.random.rand(1,7,7,1)
    a_k = K.variable(y_true)
    b_k = K.variable(y_pred)

    myloss = create_weighted_binary_crossentropy(class_weight[0], class_weight[1], wc=10.)
    value = K.eval(myloss(a_k, b_k))

    ent2 = entropia2(y_true, y_pred, class_weight[0], class_weight[1])
    print(ent2-value, ent2, value)