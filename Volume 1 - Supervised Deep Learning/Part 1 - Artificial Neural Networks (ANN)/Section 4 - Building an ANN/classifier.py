# -*- coding: utf-8 -*-

from metrics import precision, recall, dsc, compute_binary_class_weight, DL2
from metrics import create_weighted_binary_crossentropy, create_weighted_binary_accuracy

###############################################################################
# BUILD CLASSIFFIER
###############################################################################


# Importing the Keras libraries packages
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D,Dropout, Cropping2D
from keras.models import Model
from collections import OrderedDict
from keras.utils.vis_utils import plot_model

from keras.utils import multi_gpu_model

def build_classifier(layers=5, img_rows=512, img_cols=512, features_root=64,
                     optimizer='adam', dropout=0.5, init='he_normal', wc=10.):
    
    inputs = Input((img_rows, img_cols, 1))

    dw_h_convs = OrderedDict()
    
    # down layers
    for layer in range(0, layers):
        features = 2**layer*features_root
        if layer == 0:
            name='conv%da'%(layer+1)
            conv = Conv2D(features, 3, activation='relu', padding='same',
                kernel_initializer=init, name=name)(inputs)
        else:
            name = 'conv%da'%(layer+1)
            conv = Conv2D(features, 3, activation='relu', padding='same',
                kernel_initializer=init, name=name)(pool)
        
        name = 'conv%db'%(layer+1)
        conv = Conv2D(features, 3, activation ='relu',padding ='same',
                kernel_initializer=init, name=name)(conv)
        
        if layer < layers-2:
            dw_h_convs[layer] = conv
        else:
            name = 'dropout%d'%(layer+1)
            dropoutx =  Dropout(dropout, name=name)(conv)
            dw_h_convs[layer] = dropoutx
            
        if layer<layers-1:
            name='pool%d'%(layer+1)
            pool = MaxPooling2D(pool_size=(2, 2), name=name)(dw_h_convs[layer])

    # up layers
    for layer in range(layers-2, -1, -1):
        features = 2**layer*features_root
        
        name = 'up%d'%(2*layers-layer-1)
        if layer == layers-2:
            up = Conv2D(features, 2,activation='relu',padding='same',
                  kernel_initializer=init, name=name)(UpSampling2D(size=(2,2))(dw_h_convs[layer+1]))
        else:
            up = Conv2D(features, 2,activation='relu',padding='same',
                  kernel_initializer=init, name=name)(UpSampling2D(size=(2,2))(conv))
    
        name = 'merge%d'%(2*layers-layer-1)
        mergex = merge([dw_h_convs[layer], up], mode='concat', concat_axis=3,
                      name=name)
        name = 'conv%da'%(2*layers-layer-1)
        conv = Conv2D(features, 3, activation = 'relu', padding = 'same',
                kernel_initializer=init, name=name)(mergex)
        
        name = 'conv%db'%(2*layers-layer-1)
        conv = Conv2D(features, 3, activation = 'relu', padding = 'same',
                kernel_initializer = init, name=name)(conv)
        
    name = 'conv%d'%(2*layers)  
    conv = Conv2D(1, 1, activation = 'sigmoid', name=name)(conv)
    
    model = Model(input = inputs, output = conv)
    
    loss = create_weighted_binary_crossentropy(wc)
    weighted_binary_accuracy = create_weighted_binary_accuracy()

    # Multiple GPUS?
    # $ watch -n0.5 nvidia-smi # check?
    # parallel_model = multi_gpu_model(model)
    # model.compile(optimizer=optimizer, loss=loss,
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=['accuracy', weighted_binary_accuracy, dsc, precision, recall, DL2])
#               loss=loss, metrics=['accuracy', weighted_binary_accuracy])
    
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model

from keras.models import load_model
#from keras.utils.generic_utils import get_custom_objects

def load_classifier(filename):
    model = load_model(filename, custom_objects={
            'weighted_binary_crossentropy': create_weighted_binary_crossentropy(),
            'weighted_binary_accuracy': create_weighted_binary_accuracy(),
            'dsc': dsc,
            'precision': precision,
            'recall': recall,
            'DL2': DL2
            })
    return model
