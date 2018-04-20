#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Are there GPUs there?
from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
print(get_available_gpus())

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# Load datasets and shuffle training datasets
from keras.datasets import mnist
from sklearn.utils import shuffle

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, y_train = shuffle(X_train, y_train)


#  60000x28x28 to 60000x784
RESHAPED = X_train.shape[1]*X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], RESHAPED).astype('float32')
X_test = X_test.reshape(X_test.shape[0], RESHAPED).astype('float32')

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Encoding categorical data
from keras.utils import np_utils
NB_CLASSES = 10
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# Importing the Keras libraries packages
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


def build_classifier(optimizer, nb_hidden, dropout, init, levels):
    classifier = Sequential()
    classifier.add(Dense(nb_hidden, input_shape=(RESHAPED,), activation='relu',
                         kernel_initializer=init,
                         bias_initializer=init))
    classifier.add(Dropout(dropout))
    for i in range(0, levels):
        classifier.add(Dense(nb_hidden, activation='relu',
                             kernel_initializer=init,
                             bias_initializer=init))
        classifier.add(Dropout(dropout))
    classifier.add(Dense(NB_CLASSES, activation='softmax',
                         kernel_initializer=init,
                         bias_initializer=init))
    # from keras.utils import multi_gpu_model
    # Replicates `model` on 2 GPUs.
    # This assumes that your machine has 2 available GPUs.
    # parallel_classifier = multi_gpu_model(classifier, gpus=2)
    # parallel_classifier.compile(optimizer=optimizer, loss='categorical_crossentropy',
    #                   metrics=['accuracy'])
    # return parallel_classifier
    classifier.compile(optimizer=optimizer, loss='categorical_crossentropy',
                        metrics=['accuracy'])
    return classifier
    


param_grid = {'batch_size': [32, 64, 128, 256], 'epochs': [16],
              'optimizer': ['sgd', 'adam', 'rmsprop'],
              'nb_hidden': [16, 32, 64, 128, 256, 512, 1024],
              'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
              'init': ['normal', 'glorot_uniform'],
              'levels': [1, 2, 3, 4],
             }

param_grid = {'batch_size': [256], 'epochs': [1024],
              'optimizer': ['adam'],
              'nb_hidden': [16],
              'dropout': [0.5],
              'init': ['normal'],
              'levels': [1],
             }

# $ tensorboard --logdir='~/logs'
from datetime import datetime
now = datetime.now()
log_dir = '/home/carmelocuenca/tmp/logs/' + now.strftime("%Y%m%d-%H%M%S") + "/"
fit_params = {'validation_split': 0.2, 'verbose': 1, 'callbacks': [
        TensorBoard(log_dir=log_dir, histogram_freq=0,
                    write_graph=False, write_images=False),
        ]
}
    
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


classifier = KerasClassifier(build_fn=build_classifier)
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=param_grid,
                           scoring=None,
                           n_jobs=1,
                           cv=2,
                           fit_params=fit_params
                           )


import time


start = time.time()
grid_search = grid_search.fit(X_train, y_train)
print("Tiempo de aprendizaje %f" % (time.time() - start))


best_index = grid_search.best_index_
best_parameters = grid_search.best_params_
best_accuaracy = grid_search.best_score_


