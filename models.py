# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 09:47:41 2017

@author: User
"""

from __future__ import print_function

import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Lambda
from keras.optimizers import SGD,RMSprop,Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras import initializers
from keras.regularizers import l2
from yaml_pickle_hdf5 import *
from PARAMS import *


# def DNN(modelParmsFile):


#     modelParamsDict = loadfeatureParams(modelParmsFile)
#     d = modelParamsDict

#     model = Sequential()
#     model.add(Dense(256,input_dim = 60, activation = 'relu')) #uses glorot initialization by default
#     model.add(BatchNormalization())
#     model.add(Dropout(0.5))
#     model.add(Dense(128, activation = 'relu'))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.3))
#     model.add(Dense(32, activation = 'relu'))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.3))
#     model.add(Dense(1, activation = 'sigmoid'))
#     rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#     model.compile(loss = 'binary_crossentropy', optimizer = rmsprop, metrics = ['accuracy'])
    
#     return model


def DNN(modelParamsFile,model,event):

   # still need to add regularization

    modelParamsDict = loadfeatureParams(modelParamsFile)
    d = modelParamsDict
    
    model = Sequential()
    for i in d[model][event]['num_layers']:
        if(i==0):
            model.add(Dense(d[model][event][num_units][i],input_dim = 60, activation = 'relu'))
            if d[model][event]['BatchNorm'][i] == True:
                model.add(BatchNormalization())
            if d[model][event]['Dropout'][i] > 0.0:
                model.add(Dropout([model][event]['Dropout'][i]))


        else:
            model.add(Dense(d[model][event][num_units][i], activation = 'relu'))
            if d[model][event]['BatchNorm'][i] == True:
                model.add(BatchNormalization())
            if d[model][event]['Dropout'][i] > 0.0:
                model.add(Dropout([model][event]['Dropout'][i]))

    model.add(Dense(1, activation = 'sigmoid'))
    lr = d[model][event][lr]
    rmsprop = RMSprop(lr = lr, rho = 0.9, epsilon = 1e-08, decay = 0.0)
    model.compile(loss = 'binary_crossentropy', optimizer = rmsprop, metrics = ['accuracy'])

    return model 




def CNN(modelParmsFile, model, event):

    modelParamsDict = loadfeatureParams(modelParamsFile)
    d = modelParamsDict


    model = Sequential()
    channels = d[model][event]['num_channels']
    xdim = d[model][event]['xdim']
    ydim = d[model][event]['ydim']
    model.add(ZeroPadding2D((1,1), input_shape = (xdim,ydim,channels), 
                            data_format = "channels_last" ))
    model.add(Convolution2D(32,3,3, activation = 'relu', padding = 'same'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32,3,3,activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2), data_format = 'channels_last'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'sigmoid'))

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model


    