# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here

# Importing the Keras libraries packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(6, input_shape=(11,), activation='relu', kernel_initializer='uniform', bias_initializer='uniform'))

# Adding the second hidden layer
classifier.add(Dense(6, activation='relu', kernel_initializer='uniform', bias_initializer='uniform'))

# Adding the output layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='uniform', bias_initializer='uniform'))


# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'] )
 

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) 


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Homework
y = classifier.predict(sc.transform(np.array([[0, 0, 600, 0, 40, 3, 60000, 2, 1, 1, 50000]])))


# Evaluating, improving and Tunning the ANN

# Evaluating the ANN


# Importing the Keras libraries packages
import keras
from keras.models import Sequential
from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(6, input_shape=(11,), activation='relu', kernel_initializer='uniform', bias_initializer='uniform'))
    classifier.add(Dense(6, activation='relu', kernel_initializer='uniform', bias_initializer='uniform'))
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='uniform', bias_initializer='uniform'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'] )
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)    
accuaracies = cross_val_score(classifier, X_train, y=y_train, cv=10, n_jobs=4)
mean = accuaracies.mean()
variance = accuaracies.std()

# Imprroving the ANN

# Tuning the ANN

# Importing the Keras libraries packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer, rate):
    print(optimizer, rate)
    classifier = Sequential()
    classifier.add(Dense(6, input_shape=(11,), activation='relu', kernel_initializer='uniform', bias_initializer='uniform'))
    classifier.add(Dropout(rate))
    classifier.add(Dense(6, activation='relu', kernel_initializer='uniform', bias_initializer='uniform'))
    classifier.add(Dropout(rate))
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='uniform', bias_initializer='uniform'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'] )
    return classifier


classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop'],
              'rate': [0.1, 0.2, 0.3, 0.4, 0.5]}
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           n_jobs=1,
                           cv=10,
                           verbose=1)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuaracy = grid_search.best_score_
