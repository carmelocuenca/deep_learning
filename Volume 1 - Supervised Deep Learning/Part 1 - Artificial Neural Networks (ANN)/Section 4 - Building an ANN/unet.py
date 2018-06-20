#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# pil not to work with spyder, using pillow instead of pil
#
# $ conda install pyopengl spyder pillow matplotlib scikit-learn 
#
# $ work-around in order to keras uses gpuf
#
# $ conda install keras
# $ conda uninstall --force tensorflow*
# $ conda install tensorflow-gpu
# $ conda install -c conda-forge nibabel 
# $ conda install -c anaconda scikit-image 
# $ conda install pydot graphviz

# Are there GPUs there?
from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
print(get_available_gpus())

#import tensorflow as tf
# TO GPU
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# TO CPU
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))



batch_size = 8; # 8
import gc
# $ git clone https://github.com/carmelocuenca/unet

###############################################################################
# TRAIN DATASET WITH GENERATOR
# 24K images preprocessed
###############################################################################
import sys
sys.path.append("/home/carmelocuenca/Documentos/cnn_projects/deep_learning/Volume 1 - Supervised Deep Learning/Part 1 - Artificial Neural Networks (ANN)/Section 4 - Building an ANN")

from importlib import reload
import image_data_provider
reload (image_data_provider)
from image_data_provider import ImageDataProvider

path_to_datasets = '/home/carmelocuenca/tmp/datasets'

XSIZE = 384; YSIZE=384
path_to_dataset = path_to_datasets + '/' + 'aorta'
img_rows = YSIZE; img_cols = XSIZE; img_type='tiff'

path_to_dataset = path_to_datasets + '/' + 'aorta/manual'
import os
os.chdir(path_to_dataset)

import numpy as np
import matplotlib.pyplot as plt

img_rows = YSIZE; img_cols = XSIZE; img_type='.tiff'

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


##############################################################################
# X T R A I N
# ############################################################################
   

X_train, y_train = ImageDataProvider(path_to_dataset + '/' + 'train/*.tiff',
                                     data_suffix=img_type,
                                     shuffle_data=True,
                                     mask_suffix='_mask' + img_type).load_data()

scaler.fit(X_train.reshape(-1,1))
print(np.min(X_train), np.max(X_train), np.mean(X_train), np.std(X_train))
X_train = scaler.transform(X_train.reshape(-1, 1)).reshape(X_train.shape)

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

def draw(X, y, n=4):
    for _ in range(n):
        i = np.random.randint(X.shape[0])
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(X[i,:,:,0], cmap='gray')
        ax[2].imshow(y[i,:,:,0], cmap='gray')
        dummy = X[i,:,:,0].copy()
        dummy[y[i,:,:,0]!=0] = np.max(dummy)*1.5
        ax[1].imshow(dummy, cmap='gray')
        plt.show()


draw(X_train, y_train)

from keras.preprocessing.image import ImageDataGenerator


rotation_range=1.0; width_shift_range=4; height_shift_range=4; shear_range=0.01; zoom_range=[0.96, 1.04]
zca_whitening=True
#rotation_range=0.0; width_shift_range=0; height_shift_range=0; shear_range=0.0; zoom_range=0.0
#zca_whitening=False
data_gen_args = dict(featurewise_center=False, samplewise_center=False,
                     featurewise_std_normalization=False,
                     samplewise_std_normalization=False,
                     zca_whitening=zca_whitening, zca_epsilon=1e-06,
                     rotation_range=rotation_range,
                     width_shift_range=width_shift_range,
                     height_shift_range=height_shift_range,
                     brightness_range=None,
                     shear_range=shear_range,
                     zoom_range=zoom_range,
                     channel_shift_range=0.0, fill_mode='constant', cval=0.0,
                     horizontal_flip=True, vertical_flip=True,
                     data_format=None,
                     rescale = None,
                     preprocessing_function=None,
                     validation_split=0.0)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)


image_generator = image_datagen.flow(
    X_train, y=None, batch_size=batch_size, shuffle=True,
    seed=3333,
    # save_to_dir='/home/carmelocuenca/tmp/datasets/aorta/manual/deform/images',
    save_to_dir=None,
    save_prefix='', save_format='png', subset=None)

mask_generator = mask_datagen.flow(
    y_train, y=None, batch_size=batch_size, shuffle=True,
    seed=3333,
    # save_to_dir='/home/carmelocuenca/tmp/datasets/aorta/manual/deform/labels',
    save_to_dir = None,
    save_prefix='', save_format='png', subset=None)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

#
# V A L I D A T I O N
#

import numpy as np
#import matplotlib.pyplot as plt


X_val, y_val = ImageDataProvider(path_to_dataset + '/' + 'validation/*.tiff',
                                     data_suffix=img_type,
                                     shuffle_data=False,
                                     mask_suffix='_mask' + img_type).load_data()
print(np.min(X_val), np.max(X_val), np.mean(X_val), np.std(X_val))
X_val = scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)

draw(X_val, y_val)
print(np.mean(X_val), np.std(X_val))

###############################################################################
# BUILD CLASSIFFIER

import classifier
reload(classifier)
from classifier import build_classifier

###############################################################################



##############################################################################
# ROC's
#
from sklearn.metrics import roc_curve, auc
def roc(y_test, y_score):
# Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    
    roc_auc = dict()
    
    fpr[0], tpr[0], thresholds[0] = roc_curve(y_test, y_score)
    roc_auc[0] = auc(fpr[0], tpr[0])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], thresholds["micro"] = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    return fpr[0], tpr[0], thresholds[0]

###############################################################################
# FITTING
###############################################################################
from keras.optimizers import Adam

param_grid = { 'layers': {5},
              'img_rows': [img_rows], 'img_cols': [img_cols],
              'features_root': [24], # [16, 32, 64],
              'dropout': [0.1], # 'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
              'optimizer': [Adam(lr = 1e-4, decay=4e-5)], #[Adam(lr = 1e-3, decay=4e-3)],
              'init': ['he_normal' ],  # ['glorot_uniform', 'he_normal' ]
              'wc': [10.]
             }

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

# $ tensorboard --logdir='~/logs'
from datetime import datetime

now = datetime.now()
log_dir = '/home/carmelocuenca/tmp/logs/' + now.strftime("%Y%m%d-%H%M%S") + "/"
import re
import time
start = time.time()

grid_search = []
# Grid search
import itertools
keys, values = zip(*param_grid.items())
for v in itertools.product(*values):
    experiment = dict(zip(keys, v))
    print(experiment)
    classifier = build_classifier(**experiment)
    
    filename = str(now) + str(experiment)
    filename = re.sub(',', '-', re.sub("[{}<>' ]", '', str(filename)))
    
    history = classifier.fit_generator(train_generator,
    #history = classifier.fit(X_train, y_train,                              
       steps_per_epoch=48*len(X_train)//batch_size,
       #steps_per_epoch=len(X_train)//batch_size,
       epochs=25000,
       verbose=1,
       validation_data=(X_val, y_val),
       callbacks=[
               TensorBoard(log_dir=log_dir,
                           histogram_freq=0,
                           write_graph=False,
                           write_images=False),
               EarlyStopping(monitor='val_loss',
                             min_delta=0,
                             patience=32,
                             verbose=0,
                             mode='min'),
               ModelCheckpoint('/home/carmelocuenca/tmp/hdf5s/' + filename + '.hdf5',
                               monitor='val_loss',
                               verbose=1,
                               save_best_only=True),
               ])

    
    grid_search.append((history, experiment, filename))
    
    # Display
    print(experiment)
    y_pred = grid_search[-1][0].model.predict(X_val, batch_size=1, verbose=0)
    y_pred[y_pred>0.5] = 1.0; y_pred[y_pred<=0.5] = 0.0
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    
    n = np.random.randint(X_val.shape[0])
    ax[0].imshow(X_val[n,:,:,0], cmap='gray')
    ax[1].imshow(y_pred[n,:,:,0], cmap='gray')
    plt.show()
    

    for i in range(0, 16): gc.collect()
    
##############################################################################

#


print("Tiempo de aprendizaje %f" % (time.time() - start))


def np_dsc(y_true, y_pred):
    y = np.round(y_pred)
    TP = np.sum(y_true*y)
    FN = np.sum(y_true*(1.-y))
    FP = np.sum((1-y_true)*y)
    if FN+TP+FP>0:
        return TP/(FN+TP+FP)
    else: return 1.

##############################################################################
# Histograma sobre X_val
###############################################################################   

from classifier import load_classifier

model = load_classifier('/home/carmelocuenca/tmp/hdf5s/' + grid_search[-1][2] + '.hdf5')  


y_pred = model.predict(X_val, batch_size=16, verbose=0)

# the histogram of the data
n, bins, patches = plt.hist(y_pred.ravel(), 50, density=True,
                            facecolor='g', alpha=0.75)
plt.xlabel('Probabilidad')
plt.ylabel('% Densidad de probabilidad')
plt.title('Histogram')
plt.axis([0, 1., 0., 5+ np.max(n[1:])])
plt.grid(True)
plt.show()
   


###############################################################################
# LOAD TEST DATASET, EVALUATED CONFUSION MATRIX, PRECISION AND RECALL
###############################################################################
import glob
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_similarity_score




nb_crop1 = 0
nb_crop2 = 0


for directory in glob.glob(path_to_dataset + '/' + 'test-*/'):
    i = 0
    print('Working  with directory ', directory)
    X_test, y_test = ImageDataProvider(directory + '/*.tiff',
                                     data_suffix=img_type,
                                     shuffle_data=False,
                                     mask_suffix='_mask' + img_type).load_data()
    X_test = X_test[nb_crop1:X_test.shape[0]-nb_crop2,...]
    y_test = y_test[nb_crop1:X_test.shape[0]-nb_crop2,...]
     
    
    X_test = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
    
    
    X_automatic, y_automatic = ImageDataProvider(
            directory.replace('manual', 'automatic') + '/*.tiff',
            data_suffix=img_type,
            shuffle_data=False,
            mask_suffix='_mask' + img_type).load_data()
    
   
        
    X_automatic = X_automatic[nb_crop1:X_automatic.shape[0]-nb_crop2,...]
    y_automatic = y_automatic[nb_crop1:X_automatic.shape[0]-nb_crop2,...]
    print(np.min(X_automatic), np.max(X_automatic))
    X_automatic = scaler.transform(X_automatic.reshape(-1, 1)).reshape(X_automatic.shape)
    
    assert(X_automatic.shape == X_test.shape)
    assert(y_automatic.shape == y_test.shape)

    
    # cm = confusion_matrix(y_test.ravel(), y_automatic.ravel(), labels=[0., 1.])
#    _, rc, _, _ = precision_recall_fscore_support(y_test.ravel(), y_automatic.ravel())   
    
    # f1_automatic = [jaccard_similarity_score(y_true.ravel(), y_pred.ravel()) for y_true, y_pred in zip(y_test, y_automatic)]
    f1_automatic = [np_dsc(y_true, y_pred) for y_true, y_pred in zip(y_test, y_automatic)]
    f1_automatic_percentile = np.percentile(f1_automatic, np.arange(0, 100, 1), interpolation='lower')

    
    print('-'*30)
    print('-'*30)
    print('-'*30)
    print(directory)
    # print('cm=', cm)
    #print('(precision, recall)=', rc[0], rc[1])



###############################################################################
# T E S T
###############################################################################    
 
    
    for history, params, filename in grid_search:  
        
        print("Loading model ", '/home/carmelocuenca/tmp/hdf5s/' + filename + '.hdf5')
        model = load_classifier('/home/carmelocuenca/tmp/hdf5s/' + filename + '.hdf5')  
        
        y_pred = model.predict(X_test, batch_size=batch_size, verbose=0)
        y_pred[y_pred>0.5] = 1.0; y_pred[y_pred<=0.5] = 0.0
#        confusion_matrixes[i] = confusion_matrix(y_test.ravel(), y_pred.ravel(),
#                          labels=[0., 1.])
#        _, recalls[i], _, _ = precision_recall_fscore_support(y_test.ravel(), y_pred.ravel())
        
        # print(jaccard_similarity_score(y_test.ravel(), y_pred.ravel()))
        f1_pred = [np_dsc(y_true, y) for y_true, y in zip(y_test, y_pred)]
        f1_pred_percentile = np.percentile(f1_pred, np.arange(0, 100, 1), interpolation='lower')
    
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].plot(f1_automatic, color='blue', label='Tracking...')
        ax[0].plot(f1_pred, color='green', label='U-net')
        ax[0].set_xlabel('#Image')
        ax[0].set_ylabel('DSC')
        ax[0].grid(True)
        ax[0].legend()
        plt.title(os.path.basename(os.path.normpath(directory)), ha='center')
        ax[1].plot(f1_automatic_percentile , color='blue', label='Tracking...')
        ax[1].plot(f1_pred_percentile , color='green', label='U-net')
        ax[1].set_xlabel('Percentile')
        ax[1].set_ylabel('DSC')
        ax[1].grid(True)
        ax[1].legend()
        plt.show()
        
        i = i +1;
        
        print('-'*30)
        print(directory, params)
        # print('cm=', confusion_matrixes[i])
        # print('(precision, recall)=', recalls[i][0], recalls[i][1])
       
        for n in np.argsort(-np.array(f1_automatic)+np.array(f1_pred))[0:1]:
            #n = np.random.randint(X_test.shape[0])
            print("number of frame", n, f1_automatic[n], f1_pred[n])
            fig, ax = plt.subplots(1, 4, figsize=(12, 24))
            plt.title('#%d image dsc\'s=(%f, %f)'%(n, f1_automatic[n], f1_pred[n]), ha='center')
            ax[0].imshow(X_test[n,:,:,0], cmap='gray')
            ax[0].set_title('CT')
            ax[1].imshow(y_test[n,:,:,0], cmap='gray')
            ax[1].set_title('Ground Truth')
            ax[2].imshow(y_automatic[n,:,:,0], cmap='gray')
            ax[2].set_title('Tracking...')
            ax[3].imshow(y_pred[n,:,:,0], cmap='gray')
            ax[3].set_title('U-Net')
            plt.show()
        for n in np.argsort(np.array(f1_automatic)-np.array(f1_pred))[0:1]:
            #n = np.random.randint(X_test.shape[0])
            print("number of frame", n, f1_automatic[n], f1_pred[n])
            fig, ax = plt.subplots(1, 4, figsize=(12, 24))
            plt.title('#%d image dsc\'s=(%f, %f)'%(n, f1_automatic[n], f1_pred[n]), ha='center')
            ax[0].imshow(X_test[n,:,:,0], cmap='gray')
            ax[0].set_title('CT')
            ax[1].imshow(y_test[n,:,:,0], cmap='gray')
            ax[1].set_title('Ground Truth')
            ax[2].imshow(y_automatic[n,:,:,0], cmap='gray')
            ax[2].set_title('Tracking...')
            ax[3].imshow(y_pred[n,:,:,0], cmap='gray')
            ax[3].set_title('U-Net')
            plt.show()
        