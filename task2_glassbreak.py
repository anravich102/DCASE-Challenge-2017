# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 09:25:43 2017

@author: User
"""
from __future__ import print_function

from LOADKERASDATA import *
import matplotlib.pyplot as plt
from FEATURE_EXT_NORM import *
from LABEL_EXTRACTION import *
from KERAS_MODELS import *
from PARAMS import *
from POSTPROCESS import *
import yaml
import os
from keras.callbacks import *
from POSTPROCESS_TUNE import *

createParamsFile()

modelParamsFile  ='modelParams.yaml'
featureParamsFile = 'featureParams.yaml'

modelParams = loadfeatureParams(modelParamsFile)
featureParams = loadfeatureParams(featureParamsFile)


def train_and_predict(xTrain,yTrain,xVal,yVal,modelname,event,modelParams, predictOnData = None):

    if modelname == 'dnn':
        model = DNN(modelParams,modelname,event)
    elif modelname == 'cnn':
        model = CNN(modelParams,modelname,event)



    print ("Priniting Model Architecture\n ")
    model.summary()
    print( "Done printing model summary! \n")


    #fit(self, x, y, batch_size=32, epochs=10, verbose=1, 
    #callbacks=None, validation_split=0.0, validation_data=None, shuffle=True,
    # class_weight=None, sample_weight=None, initial_epoch=0)

    batch_size=modelParams[modelname][event]['batch_size']
    epochs= modelParams[modelname][event]['epochs']
    
    validation_data = (xVal,yVal)

    min_delta = modelParams['min_delta']
    patience = modelParams['patience']
    earlystopping = EarlyStopping(monitor='val_loss',
                     min_delta=min_delta, patience=patience, verbose=0, mode='auto')

    class_weight = None
    if modelParams['class_weights_enable'] ==True:
        class_weight = modelParams[modelname][event]['class_weights']

    hist = model.fit(xTrain,yTrain,batch_size = batch_size, epochs = epochs, verbose = 1, 
                 validation_data = validation_data, callbacks = [earlystopping], class_weight = class_weight)

    print(hist.history)

    if predictOnData is not None:
        predictions = model.predict(predictOnData)

    else:
        predictions = model.predict(xVal)


    return predictions,hist.history






def print_final_results(metrics,events,models):
    er = 0.0
    fs = 0.0
    for model in models:
        for event in events:

            fs = fs +  metrics[model][event]['fscore']
            er = er + metrics[model][event]['er']
            print("Model: ", model, "EVENT: " , event, " ER: " , er ,
                   "F Score: ", fs )

    print("Average F Score: ", fs/3.0)
    print("Average Error Rate: ", er/3.0)









dnnmetrics = {'babycry': {'er': 0.0, 'fscore': 0.0},
           'glassbreak': {'er': 0.0, 'fscore': 0.0},
           'gunshot': {'er': 0.0, 'fscore': 0.0} }

cnnmetrics = {'babycry': {'er': 0.0, 'fscore': 0.0},
           'glassbreak': {'er': 0.0, 'fscore': 0.0},
           'gunshot': {'er': 0.0, 'fscore': 0.0} }

global_metrics = {}

models = ['dnn']
events = ['glassbreak']

global_metrics['dnn'] = dnnmetrics
global_metrics['cnn'] = cnnmetrics

tuningMode = modelParams['tuning_mode']

for model in models:

    directory = None
    for event in events:

        evaluate = False

        directory = 'dnn_11_57_AM_July_14_2017'

        # print("Starting Feature Extraction....")
        # timestamp = do_feature_extraction( model, event, source_data = True, mode = 'devtrain',directory = directory)
        # print("Done with  Feature Extraction !")

        # directory = timestamp

        # print("Starting Feature Normalization....")
        # ignore = do_feature_normalization(model, event, directory, source_data = True, mode = 'devtrain')
        # print("Done with  Feature Normalization !")

        # print("Starting Label Extraction....")
        # ignore = do_label_extraction(directory, model,event, source_data = True, mode = 'devtrain')
        # print("Done With Label Extraction!")


        print("Loading Training Data....")
        xTrain,yTrain,fileList,numFramesList = loadKerasData(model, directory, event,
                        mode = 'devtrain', source_data = False,  normalized_data = False)
        print("Done Loading Training Data!")


        # print("Starting Feature Extraction for Validation data....")
        # directory = do_feature_extraction( model, event, source_data = True, mode = 'devtest', directory = directory)
        # print("Done with  Feature Extraction !")

        # print("Starting Feature Normalization for validation data....")
        # ignore = do_feature_normalization(model, event, directory, source_data = True, mode = 'devtest')
        # print("Done with  Feature Normalization !")

        # print("Starting Label Extraction for validation data....")
        # ignore = do_label_extraction(directory, model,event, source_data = True, mode = 'devtest',)
        # print("Done With Label Extraction!")





        print("Loading Validation Data....")
        xVal,yVal,fileList,numFramesList = loadKerasData(model, directory, event,
                        mode = 'devtest', source_data = False, normalized_data = False)
        print("Done Loading Validation Data!")


        predictOnData = xVal
        evaluate = True 

        '''
           do_feature_extraction(model, event, source_data = False, mode = 'eval')
           do_feature_normalization(model, event, directory, source_data = False, mode = 'eval')
           predictOnData, fileList,numFramesList = loadKerasData(model, directory, event,
                       numofFiles = float('inf'), mode = 'eval', source_data = False)

           evaluate = False 
        '''

        ''' tuning module begins '''
        printlist = []
        if tuningMode:
            #same as train and predict but with varying parameters
            

            print("Tuning Hyperparameters...")
            for count in range(modelParams['tuning_countmax']):
                
                
                reg = 10**np.random.uniform(-5,5)
                lr = 10**np.random.uniform(-2,-5)

                modelParams[model][event]['l2reg'] = reg
                modelParams[model][event]['lr'] = lr

                print("Fetching Model Predictions...")
                predictions,history = train_and_predict(xTrain,yTrain,xVal,yVal,model,event,modelParams,
                                                predictOnData = xVal)

                print("shape of predictions: ", predictions.shape)

                print("Preparing Annotated File...")
                userAnnotatedFile = process_predictions(predictions, model, directory,
                                 featureParams, modelParams,
                                 event, numFramesList, fileList) #produce files

                print("done Producing annotated Files! ")

                if evaluate:
                    

                    pathToFile = os.path.join(os.getcwd(),
                                 '../data/mixture_data/devtest/20b255387a2d0cddc0a3dff5014875e7/meta')
                    listofFiles  = os.listdir(pathToFile) 
                    filename = 'event_list_devtest_'+event +'.csv'
                    groundTruthFile = os.path.join(pathToFile, filename)

                    er, fscore,fp,fn,ins,dele,tp= get_metrics(userAnnotatedFile, groundTruthFile, event) #produce metrics
                    global_metrics[model][event]['er'] = er
                    global_metrics[model][event]['fscore'] = fscore

                    l = ['reg:', reg, 'lr:', lr, 'er:', er, 'fscore:', fscore,history]
                    printlist.append(l)


                    #print("model: ", model, "event: ", event, "er :",er , "Fscore :",fscore )



            print("Done tuning hyperparameters...")

            for list_item in printlist:
                print(list_item)

            sys.exit()

        '''tuning moodule ends'''

        print("Training and Fetching Model Predictions...")
        predictions,history = train_and_predict(xTrain,yTrain,xVal,yVal,model,event,modelParams,
                                        predictOnData = xVal)

        # print("shape of predictions: ", predictions.shape)
        # print("shape of numFramesList: ", len(numFramesList))
        # print("shape of files list: ", len(fileList))

        # filename = 'numFramesFile.txt'
        # with open(filename,'w') as fd:
        #     fd.write(str(numFramesList))


        if modelParams['postprocess_tune'] == True:

            pathToFile = save_raw_predictions(predictions, model, directory, featureParams, modelParams,
                         event, numFramesList, fileList)
            ans = input('Saved raw predictions to file. tune parameters?')
            if ans=='y':
                print("Saved File for post process tuning. Tuning PP parameters...")
                simulate(pathToFile)
                print("Done! exiting...")
                sys.exit()



        print("Preparing Annotated File...")
        userAnnotatedFile = process_predictions(predictions, model, directory, featureParams, modelParams,
                         event, numFramesList, fileList) #produce files

        print("done Producing annotated Files! ")

        if evaluate:
            

            pathToFile = os.path.join(os.getcwd(), '../data/mixture_data/devtest/20b255387a2d0cddc0a3dff5014875e7/meta')
            listofFiles  = os.listdir(pathToFile) 
            filename = 'event_list_devtest_'+event +'.csv'
            groundTruthFile = os.path.join(pathToFile, filename)

            er, fscore,fp,fn,ins,dele,tp= get_metrics(userAnnotatedFile, groundTruthFile, event) #produce metrics
            print("fp:",fp,"fn:",fn, "ins:", ins, "dele", dele, "tp:", tp)
            global_metrics[model][event]['er'] = er
            global_metrics[model][event]['fscore'] = fscore

            print("model: ", model)
            print("event: ", event)
            print("er :",er )
            print("Fscore :",fscore)


            ''' Savve results to file '''
            metricsFileName = model + '_' + directory + event + 'METRICS.txt'
            metricsFilePath = os.path.join(os.getcwd(),'../Results',metricsFileName)

            with open(metricsFilePath,'w') as f:
                f.write(model + "\t" + event + "\t" + 'er:' + "\t" + str(er) + '\t' + 
                        'fscore: '+ str(fscore) + "\t" + "\n")

if evaluate:
    print("########################")
    print("FINAL RESULTS: ")
    print_final_results(global_metrics,events,models)




































# xTrain, yTrain = loadKerasData(numofFiles = 20,event = 'babycry', mode = 'devtrain',featuretype = 'mfcc', nmfcc = 60)

# xVal,yVal =  loadKerasData(numofFiles = 20,event = 'babycry', mode = 'devtest',featuretype = 'mfcc', nmfcc = 60)

# model = DNN()

# print ("Priniting Model Architecture\n ")
# model.summary()
# print( "Done printing model summary! \n")


# #fit(self, x, y, batch_size=32, epochs=10, verbose=1, 
# #callbacks=None, validation_split=0.0, validation_data=None, shuffle=True,
# # class_weight=None, sample_weight=None, initial_epoch=0)

# batch_size=20
# epochs=1
# validation_data = (xVal,yVal)

# hist = model.fit(xTrain,yTrain,batch_size = batch_size, epochs = epochs, verbose = 1, 
#                  validation_data = validation_data)

# print(hist.history)

# result = model.predict(xVal)

# print("shape of predictions: ", result.shape)
# print("some predictions..", result[0:20][:])

#Summarize history for accuracy
#plt.plot(hist.history['acc'])
#plt.plot(hist.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['Training', 'Validation'], loc='upper left')
#plt.show()

#Summarize history for loss
#plt.plot(hist.history['loss'])
#plt.plot(hist.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['Training', 'Validation'], loc='upper left')
#plt.show()


