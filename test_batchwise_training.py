from __future__ import print_function

import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Lambda
from keras.optimizers import SGD,RMSprop,Adam,Adagrad
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras import initializers
from keras import regularizers
from yaml_pickle_hdf5 import *
from PARAMS import loadfeatureParams, createBatchTrainingParams
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

import h5py
import numpy as np 


# def generate_arrays_from_file():
#     while 1:
#         f = open(path)
#         for line in f:
#             # create Numpy arrays of input data
#             # and labels, from each line in the file
#             x, y = process_line(line)
#             yield (x, y)
#             f.close()

# def myGenerator():
#     (X_train, y_train), (X_test, y_test) = mnist.load_data()
#     y_train = np_utils.to_categorical(y_train,10)
#     X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#     X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
#     X_train = X_train.astype('float32')
#     X_test = X_test.astype('float32')
#     X_train /= 255
#     X_test /= 255
#     while 1:
#         for i in range(1875): # 1875 * 32 = 60000 -> # of training samples
#             if i%125==0:
#                 print "i = " + str(i)
#             yield X_train[i*32:(i+1)*32], y_train[i*32:(i+1)*32]


#createBatchTrainingParams()




def generate_train(model, featureParams, event, featureFile, labelFile, numofFiles = float('inf')):
    
    createBatchTrainingParams()
    batchTrainingParams = loadfeatureParams('batchTrainingParams.yaml') 
    file1 = featureFile
    file2 = labelFile

    total = batchTrainingParams[model]['generate_train_total']
    #batchsize = 1964
    batchsize = batchTrainingParams[model]['generate_train_batchsize']
    
    if numofFiles != float('inf'):
        total = 1

    while 1:
        for i in range(total):  
            with h5py.File(file1, "r") as ffile, h5py.File(file2,"r") as lfile:
                temp  = ffile['features'][i*batchsize:(i+1)*batchsize] 
                if(featureParams[model][event]['label_type']) == 'one_hot':
                    yield (temp, to_categorical(lfile['labels'][i*batchsize:(i+1)*batchsize]) )
                else:
                     yield (temp, lfile['labels'][i*batchsize:(i+1)*batchsize] )




def generate_val(model, featureParams, event, featureFile, labelFile, numofFiles = float('inf')):
    
    batchTrainingParams = loadfeatureParams('batchTrainingParams.yaml')
    file1 = featureFile
    file2 = labelFile

    total = batchTrainingParams[model]['generate_val_total']
    #batchsize = 1964
    batchsize = batchTrainingParams[model]['generate_val_batchsize']

    if numofFiles != float('inf'):
        total = 1


    while 1:
        for i in range(total):
            with h5py.File(file1, "r") as ffile, h5py.File(file2,"r") as lfile:
                temp  = ffile['features'][i*batchsize:(i+1)*batchsize] 
                if(featureParams[model][event]['label_type']) == 'one_hot':
                    yield (temp, to_categorical(lfile['labels'][i*batchsize:(i+1)*batchsize]) )
                else:
                     yield (temp, lfile['labels'][i*batchsize:(i+1)*batchsize] )

                


def generate_pred(model, featureParams, event, featureFile):
    #file1 = '../cnn_07_01_PM_July_17_2017_babycry_devtest_features.h5'
   file1 = featureFile

   batchTrainingParams = loadfeatureParams('batchTrainingParams.yaml')

    # total = 373
    # batchsize = 1984

   #total = batchTrainingParams[model]['generate_val_total']
    #batchsize = 1964
   #batchsize = batchTrainingParams[model]['generate_val_batchsize']
   total = 1
   batchsize = 742016

   while 1:
        for i in range(total):
            with h5py.File(file1, "r") as ffile:
                temp  = ffile['features'][i*batchsize:(i+1)*batchsize] 
                yield temp       












# model = Sequential()
# channels = 1
# xdim = 7   #no of cols, width
# ydim = 1025   #no of rows, height

# print(xdim)
# print(ydim)

# model.add(ZeroPadding2D((0,1), input_shape = (channels, ydim, xdim), 
#                         data_format = "channels_first" ))
# #print(model.output_shape)
# #sys.exit()
# model.add(Convolution2D(32, kernel_size = (7,3) , strides = (3,1), activation = 'relu', kernel_regularizer= None
#                          ,  data_format = 'channels_first', padding='valid'))
# #print(model.output_shape)

# model.add(ZeroPadding2D((0,1)))
# model.add(Convolution2D(16, kernel_size = (5,3), strides = (3,1), activation = 'relu'
#                          , data_format = 'channels_first', padding='valid'))
# #print(model.output_shape)
# model.add(BatchNormalization())
# model.add(MaxPooling2D((3,2), data_format = 'channels_first'))
# #print(model.output_shape)
# #sys.exit()

# model.add(Flatten())
# #print(model.output_shape)
# #sys.exit()
# model.add(Dropout(0.55))
# model.add(Dense(1, activation = 'sigmoid'))

# lr = 0.001


# decay = 5e-3
# optimizer = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True) 

# #adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])




# tef = 'cnn_07_01_PM_July_17_2017_babycry_devtest_features.h5'
# tel = 'cnn_07_01_PM_July_17_2017_babycry_devtest_labels.h5'
# trf = 'cnn_07_01_PM_July_17_2017_babycry_devtrain_features.h5'
# trl = 'cnn_07_01_PM_July_17_2017_babycry_devtrain_labels.h5'

# model.fit_generator(generate_train(trf,trl), steps_per_epoch=373, epochs=1,verbose=1,
#                      validation_data = generate_val(tef,tel),
#                     validation_steps = 373)

# predictions = model.predict_generator( generate_pred(), steps=373, verbose=1)



# print("shape of predictions: ", predictions.shape)


def generate_train_new(subEpochNumber, model, featureParams, event,
                         featureFile, labelFile, 
                         numofFiles = float('inf')):
    
    createBatchTrainingParams()
    batchTrainingParams = loadfeatureParams('batchTrainingParams.yaml') 
    file1 = featureFile
    file2 = labelFile

    total = batchTrainingParams[model]['generate_train_total']
    #batchsize = 1964
    batchsize = batchTrainingParams[model]['generate_train_batchsize']
    
    if numofFiles != float('inf'):
        total = 1

   
    i = subEpochNumber
    with h5py.File(file1, "r") as ffile, h5py.File(file2,"r") as lfile:
        
        if(featureParams[model][event]['label_type']) == 'one_hot':
            return ffile['features'][i*batchsize:(i+1)*batchsize],to_categorical(lfile['labels'][i*batchsize:(i+1)*batchsize]) 
        else:
             return ffile['features'][i*batchsize:(i+1)*batchsize],lfile['labels'][i*batchsize:(i+1)*batchsize] 



def generate_val_new(subEpochNumber,model, featureParams, event, featureFile, 
                        labelFile, numofFiles = float('inf')):
    
    batchTrainingParams = loadfeatureParams('batchTrainingParams.yaml')
    file1 = featureFile
    file2 = labelFile

    total = batchTrainingParams[model]['generate_val_total']
    #batchsize = 1964
    batchsize = batchTrainingParams[model]['generate_val_batchsize']

    if numofFiles != float('inf'):
        total = 1


    
    i = subEpochNumber
    with h5py.File(file1, "r") as ffile, h5py.File(file2,"r") as lfile:
        
        if(featureParams[model][event]['label_type']) == 'one_hot':
            return ffile['features'][i*batchsize:(i+1)*batchsize],to_categorical(lfile['labels'][i*batchsize:(i+1)*batchsize]) 
        else:
             return ffile['features'][i*batchsize:(i+1)*batchsize],lfile['labels'][i*batchsize:(i+1)*batchsize] 

                


def generate_pred_new(iterations,model, featureParams, event, featureFile,flag = None, mode  = None):
    #file1 = '../cnn_07_01_PM_July_17_2017_babycry_devtest_features.h5'

   if mode is None:
   
       file1 = featureFile

       batchTrainingParams = loadfeatureParams('batchTrainingParams.yaml')

        # total = 373
        # batchsize = 1984

       #total = batchTrainingParams[model]['generate_val_total']
        #batchsize = 1964
       #batchsize = batchTrainingParams[model]['generate_val_batchsize']
       
       batchsize = batchTrainingParams[model]['predict_batchsize']
       if flag is not None:
           batchsize = batchTrainingParams[model]['newTest_batchsize']

       i=iterations

       
       with h5py.File(file1, "r") as ffile:
           return ffile['features'][i*batchsize:(i+1)*batchsize]    

   elif mode=='eval':
        file1 = featureFile

        batchTrainingParams = loadfeatureParams('batchTrainingParams.yaml')
        batchsize = batchTrainingParams[model]['eval_batchsize']



        i=iterations

       
        with h5py.File(file1, "r") as ffile:
            return ffile['features'][i*batchsize:(i+1)*batchsize]    

