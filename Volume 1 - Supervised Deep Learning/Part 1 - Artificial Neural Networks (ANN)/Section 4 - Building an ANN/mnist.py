#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Load datasets
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
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


def build_classifier(optimizer, init, levels):
    N_HIDDEN = 128
    classifier = Sequential()
    classifier.add(Dense(N_HIDDEN, input_shape=(RESHAPED,), activation='relu',
                         kernel_initializer=init,
                         bias_initializer=init))
    classifier.add(Dropout(0.5))
    for i in range(1, levels-1):
        classifier.add(Dense(RESHAPED, activation='relu',
                             kernel_initializer=init,
                             bias_initializer=init))
        classifier.add(Dropout(0.5))
    classifier.add(Dense(NB_CLASSES, activation='softmax',
                         kernel_initializer=init,
                         bias_initializer=init))
    classifier.compile(optimizer=optimizer, loss='categorical_crossentropy',
                       metrics=['accuracy'])
    return classifier



from keras.wrappers.scikit_learn import KerasClassifier


classifier = KerasClassifier(build_fn=build_classifier)
param_grid = {'batch_size': [128], 'epochs': [16],
              'optimizer': ['adam'],
              'init': ['glorot_uniform'], 'levels': [2],
             }
# $ tensorboard --logdir='~/logs'
from datetime import datetime
now = datetime.now()
log_dir = '/Users/carmelo.cuenca/logs/' + now.strftime("%Y%m%d-%H%M%S") + "/"
fit_params = {'validation_split': 0.2, 'verbose': 1, 'callbacks': [
        TensorBoard(log_dir=log_dir, histogram_freq=0,
                    write_graph=True, write_images=False),
        ]
}
    
from sklearn.model_selection import GridSearchCV


grid_search = GridSearchCV(estimator=classifier,
                           param_grid=param_grid,
                           scoring=None,
                           n_jobs=1,
                           cv=3,
                           fit_params=fit_params
                           )
grid_search = grid_search.fit(X_train, y_train)

best_index = grid_search.best_index_
best_parameters = grid_search.best_params_
best_accuaracy = grid_search.best_score_


