# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 09:47:41 2017

@author: User
"""

from __future__ import print_function

import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Dense,Dropout,Activation,Flatten,Lambda
from keras.optimizers import SGD,RMSprop,Adam,Adagrad
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras import initializers
from keras import regularizers
from yaml_pickle_hdf5 import *
from keras.layers import Bidirectional, TimeDistributed
from PARAMS import loadfeatureParams, createParamsFile
from keras.callbacks import EarlyStopping


def DNN_old(modelParams,modelname,event):

   # still need to add regularization
    

    modelParams = loadfeatureParams('modelParams.yaml')
   
    d = modelParams
    print("Input dimension: ",d[modelname][event]['input_dimension'])

    if d[modelname][event]['l2reg'] > 0:
        l2_reg = regularizers.l2( d[modelname][event]['l2reg'])
    else:
        l2_reg = None

    kernel_initializer = d[modelname][event]['weight_init']
    
    model = Sequential()
    for i in range(d[modelname][event]['num_layers']):
        if(i==0):
            model.add(Dense(d[modelname][event]['num_units'][i],input_dim = d[modelname][event]['input_dimension'],
             activation = 'relu', kernel_regularizer= l2_reg, kernel_initializer = kernel_initializer))
            if d[modelname][event]['batch_norm'][i] == True:
                model.add(BatchNormalization())
            if d[modelname][event]['Dropout'][i] > 0.0:
                model.add(Dropout(d[modelname][event]['Dropout'][i]))


        else:
            model.add(Dense(d[modelname][event]['num_units'][i], activation = 'relu'
                            , kernel_regularizer= l2_reg , kernel_initializer = kernel_initializer))
            if d[modelname][event]['batch_norm'][i] == True:
                model.add(BatchNormalization())
            if d[modelname][event]['Dropout'][i] > 0.0:
                model.add(Dropout(d[modelname][event]['Dropout'][i]))

    
    model.add(Dense(1, activation = 'sigmoid'))

    lr = d[modelname][event]['lr']
    decay = d[modelname][event]['decay']
    optimizer = d[modelname][event]['optimizer']

    if optimizer == 'sgd':
        optimizer = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True) 
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(lr = lr, rho = 0.9, epsilon = 1e-08, decay = decay)
    elif optimizer == 'adagrad':
        optimizer = Adagrad(lr=lr, epsilon=1e-08, decay=decay)

    loss = d[modelname][event]['loss']
    
    #rmsprop = RMSprop(lr = lr, rho = 0.9, epsilon = 1e-08, decay = 0.0)
    model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy'])

    return model 

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
# createParamsFile()
featureParams = loadfeatureParams('featureParams.yaml')



def DNN(modelParams,modelname,event):

   # still need to add regularization
    

    modelParams = loadfeatureParams('modelParams.yaml')
   
    d = modelParams
    print("Input dimension: ",d[modelname][event]['input_dimension'])

    if d[modelname][event]['l2reg'] > 0:
        l2_reg = regularizers.l2( d[modelname][event]['l2reg'])
    else:
        l2_reg = None

    kernel_initializer = d[modelname][event]['weight_init']
    
    model = Sequential()
    for i in range(d[modelname][event]['num_layers']):
        if(i==0):
            model.add(Dense(d[modelname][event]['num_units'][i],input_dim = d[modelname][event]['input_dimension'],
             activation = 'relu', kernel_regularizer= l2_reg, kernel_initializer = kernel_initializer))
            if d[modelname][event]['batch_norm'][i] == True:
                model.add(BatchNormalization())
            if d[modelname][event]['Dropout'][i] > 0.0:
                model.add(Dropout(d[modelname][event]['Dropout'][i]))


        else:
            model.add(Dense(d[modelname][event]['num_units'][i], activation = 'relu'
                            , kernel_regularizer= l2_reg , kernel_initializer = kernel_initializer))
            if d[modelname][event]['batch_norm'][i] == True:
                model.add(BatchNormalization())
            if d[modelname][event]['Dropout'][i] > 0.0:
                model.add(Dropout(d[modelname][event]['Dropout'][i]))

    if featureParams[modelname][event]['label_type'] == 'one_hot':
        model.add(Dense(2, activation = 'softmax'))
    else:
        model.add(Dense(1, activation = 'sigmoid'))


    lr = d[modelname][event]['lr']
    decay = d[modelname][event]['decay']
    optimizer = d[modelname][event]['optimizer']

    if optimizer == 'sgd':
        optimizer = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True) 
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(lr = lr, rho = 0.9, epsilon = 1e-08, decay = decay)
    elif optimizer == 'adagrad':
        optimizer = Adagrad(lr=lr, epsilon=1e-08, decay=decay)

    loss = d[modelname][event]['loss']
    
    #rmsprop = RMSprop(lr = lr, rho = 0.9, epsilon = 1e-08, decay = 0.0)
    model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy'])

    return model 




def CNN(modelParams, modelname, event):


    modelParams = loadfeatureParams('modelParams.yaml')
   
    d = modelParams
    

    if d[modelname][event]['l2reg'] > 0:
        l2_reg = regularizers.l2( d[modelname][event]['l2reg'])
    else:
        l2_reg = None

    kernel_initializer = d[modelname][event]['weight_init']

    model = Sequential()
    channels =  d[modelname][event]['num_channels']
    xdim = d[modelname][event]['xdim']   #no of cols, width
    ydim = d[modelname][event]['ydim']   #no of rows, height

    print(xdim)
    print(ydim)



    model.add(ZeroPadding2D((1,1), input_shape = (channels, ydim, xdim), 
                            data_format = "channels_first" ))
   
    model.add(Convolution2D(16, kernel_size = (3,3) , strides = (1,1),
                             activation = 'relu', kernel_regularizer= l2_reg
                             , kernel_initializer = kernel_initializer,
                              data_format = 'channels_first', padding='same'))

    

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(16, kernel_size = (3,3), strides = (1,1), activation = 'relu',
                                 kernel_regularizer= l2_reg
                             , kernel_initializer = kernel_initializer,
                             data_format = 'channels_first', padding='same'))
    
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2), data_format = 'channels_first'))
    model.add(ZeroPadding2D((1,1)))
    print(model.output_shape)
    model.add(Convolution2D(32, kernel_size = (3,3), strides = (1,1), activation = 'relu',
                                 kernel_regularizer= l2_reg
                             , kernel_initializer = kernel_initializer,
                             data_format = 'channels_first', padding='valid'))
    model.add(ZeroPadding2D((1,1)))
    print(model.output_shape)
    model.add(Convolution2D(32, kernel_size = (3,3), strides = (1,1), activation = 'relu',
                             kernel_regularizer= l2_reg
                         , kernel_initializer = kernel_initializer,
                         data_format = 'channels_first', padding='valid'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3), data_format = 'channels_first'))


    #print(model.output_shape)
    #sys.exit()

    model.add(Flatten())

    #model.add(Dropout(0.2))
    model.add(Dense(50, activation = 'relu', kernel_regularizer= l2_reg))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(50, activation = 'relu', kernel_regularizer= l2_reg))
    model.add(BatchNormalization())
    model.add(Dropout(0.35))
    if(featureParams[modelname][event]['label_type']) == 'one_hot':
        model.add(Dense(2, activation = 'softmax'))
    else:
        model.add(Dense(1, activation = 'sigmoid'))

    lr = d[modelname][event]['lr']
    decay = d[modelname][event]['decay']
    optimizer = d[modelname][event]['optimizer']

    if optimizer == 'sgd':
        optimizer = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True) 
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(lr = lr, rho = 0.9, epsilon = 1e-08, decay = decay) 
    elif optimizer == 'adam':
        optimizer = Adam(lr = lr, epsilon = 1e-08, decay = decay) 
    elif optimizer == 'adagrad':
        optimizer = Adagrad(lr=lr, epsilon=1e-08, decay=decay)


    loss = d[modelname][event]['loss']

    model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy'])

    return model


def CNN_new(modelParams, modelname, event):


    modelParams = loadfeatureParams('modelParams.yaml')
   
    d = modelParams
    

    if d[modelname][event]['l2reg'] > 0:
        l2_reg = regularizers.l2( d[modelname][event]['l2reg'])
    else:
        l2_reg = None

    kernel_initializer = d[modelname][event]['weight_init']

    model = Sequential()
    channels =  d[modelname][event]['num_channels']
    xdim = d[modelname][event]['xdim']   #no of cols, width
    ydim = d[modelname][event]['ydim']   #no of rows, height

    print(xdim)
    print(ydim)



    model.add(ZeroPadding2D((1,1), input_shape = (channels, ydim, xdim), 
                            data_format = "channels_first" ))
   
    model.add(Convolution2D(16, kernel_size = (3,3) , strides = (1,1),
                             activation = 'relu', kernel_regularizer= l2_reg
                             , kernel_initializer = kernel_initializer,
                              data_format = 'channels_first', padding='same'))

    

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(16, kernel_size = (3,3), strides = (1,1), activation = 'relu',
                                 kernel_regularizer= l2_reg
                             , kernel_initializer = kernel_initializer,
                             data_format = 'channels_first', padding='same'))
    
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2), data_format = 'channels_first'))
    model.add(ZeroPadding2D((1,1)))
    print(model.output_shape)
    model.add(Convolution2D(32, kernel_size = (3,3), strides = (1,1), activation = 'relu',
                                 kernel_regularizer= l2_reg
                             , kernel_initializer = kernel_initializer,
                             data_format = 'channels_first', padding='valid'))
    model.add(ZeroPadding2D((1,1)))
    print(model.output_shape)
    model.add(Convolution2D(32, kernel_size = (3,3), strides = (1,1), activation = 'relu',
                             kernel_regularizer= l2_reg
                         , kernel_initializer = kernel_initializer,
                         data_format = 'channels_first', padding='valid'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,1), data_format = 'channels_first'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(16, kernel_size = (3,3), strides = (1,1), activation = 'relu',
                                 kernel_regularizer= l2_reg
                             , kernel_initializer = kernel_initializer,
                             data_format = 'channels_first', padding='same'))

    model.add(ZeroPadding2D((1,1)))
    
    print(model.output_shape)
    model.add(Convolution2D(16, kernel_size = (3,3), strides = (1,1), activation = 'relu',
                                 kernel_regularizer= l2_reg
                             , kernel_initializer = kernel_initializer,
                             data_format = 'channels_first', padding='valid'))
    
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,1), data_format = 'channels_first'))
    
    
    #print(model.output_shape)
    #sys.exit()

    model.add(Flatten())

    #model.add(Dropout(0.2))
    model.add(Dense(50, activation = 'relu', kernel_regularizer= l2_reg))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(50, activation = 'relu', kernel_regularizer= l2_reg))
    model.add(BatchNormalization())
    model.add(Dropout(0.35))
    if(featureParams[modelname][event]['label_type']) == 'one_hot':
        model.add(Dense(2, activation = 'softmax'))
    else:
        model.add(Dense(1, activation = 'sigmoid'))

    lr = d[modelname][event]['lr']
    decay = d[modelname][event]['decay']
    optimizer = d[modelname][event]['optimizer']

    if optimizer == 'sgd':
        optimizer = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True) 
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(lr = lr, rho = 0.9, epsilon = 1e-08, decay = decay) 
    elif optimizer == 'adam':
        optimizer = Adam(lr = lr, epsilon = 1e-08, decay = decay) 
    elif optimizer == 'adagrad':
        optimizer = Adagrad(lr=lr, epsilon=1e-08, decay=decay)


    loss = d[modelname][event]['loss']

    model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy'])

    return model


    
def RNN(modelParams,modelname,event):

   # still need to add regularization



    

    modelParams = loadfeatureParams('modelParams.yaml')
   
    d = modelParams
    print("Input dimension: ",d[modelname][event]['input_dimension'])

    if d[modelname][event]['l2reg'] > 0:
        l2_reg = regularizers.l2( d[modelname][event]['l2reg'])
    else:
        l2_reg = None

    kernel_initializer = d[modelname][event]['weight_init']
    
    model = Sequential()
    
    model.add(Bidirectional(GRU(256, return_sequences=True, kernel_regularizer  = l2_reg),
                input_shape=(495, 567),  merge_mode='ave') )
    model.add(Bidirectional(GRU(128, return_sequences=True, kernel_regularizer  = l2_reg),   merge_mode='ave' ))
    model.add(BatchNormalization())

    model.add(TimeDistributed(Dense(100, activation = 'relu', kernel_regularizer= l2_reg)))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(TimeDistributed(Dense(50, activation = 'relu', kernel_regularizer= l2_reg)))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dropout(0.35)))

    if featureParams[modelname][event]['label_type'] == 'one_hot':
        model.add(TimeDistributed(Dense(2, activation = 'softmax')))
    else:
        model.add(TimeDistributed(Dense(1, activation = 'sigmoid')))


    lr = d[modelname][event]['lr']
    decay = d[modelname][event]['decay']
    optimizer = d[modelname][event]['optimizer']

    if optimizer == 'sgd':
        optimizer = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True) 
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(lr = lr, rho = 0.9, epsilon = 1e-08, decay = decay)
    elif optimizer == 'adagrad':
        optimizer = Adagrad(lr=lr, epsilon=1e-08, decay=decay)

    loss = d[modelname][event]['loss']
    
    #rmsprop = RMSprop(lr = lr, rho = 0.9, epsilon = 1e-08, decay = 0.0)
    model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy'])

    return model 


def RNN_lstm(modelParams,modelname,event):

   # still need to add regularization



    

    modelParams = loadfeatureParams('modelParams.yaml')
   
    d = modelParams
    print("Input dimension: ",d[modelname][event]['input_dimension'])

    if d[modelname][event]['l2reg'] > 0:
        l2_reg = regularizers.l2( d[modelname][event]['l2reg'])
    else:
        l2_reg = None

    kernel_initializer = d[modelname][event]['weight_init']
    
    model = Sequential()
    
    model.add(Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer  = l2_reg),
                input_shape=(495, 189),  merge_mode='ave') )
    model.add(Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer  = l2_reg),   merge_mode='ave' ))
    model.add(BatchNormalization())

    model.add(TimeDistributed(Dense(100, activation = 'relu', kernel_regularizer= l2_reg)))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(TimeDistributed(Dense(50, activation = 'relu', kernel_regularizer= l2_reg)))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dropout(0.35)))

    if featureParams[modelname][event]['label_type'] == 'one_hot':
        model.add(TimeDistributed(Dense(2, activation = 'softmax')))
    else:
        model.add(TimeDistributed(Dense(1, activation = 'sigmoid')))


    lr = d[modelname][event]['lr']
    decay = d[modelname][event]['decay']
    optimizer = d[modelname][event]['optimizer']

    if optimizer == 'sgd':
        optimizer = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True) 
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(lr = lr, rho = 0.9, epsilon = 1e-08, decay = decay)
    elif optimizer == 'adagrad':
        optimizer = Adagrad(lr=lr, epsilon=1e-08, decay=decay)

    loss = d[modelname][event]['loss']
    
    #rmsprop = RMSprop(lr = lr, rho = 0.9, epsilon = 1e-08, decay = 0.0)
    model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy'])

    return model 