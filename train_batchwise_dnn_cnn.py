from __future__ import print_function

from LOADKERASDATA import *
import matplotlib.pyplot as plt
from FEATURE_EXT_NORM import *
from LABEL_EXTRACTION import *
from KERAS_MODELS import * 
from PARAMS import createParamsFile
from POSTPROCESS import *
import yaml
import os,sys
from keras.callbacks import *
from POSTPROCESS_TUNE import *
from test_batchwise_training import generate_pred,generate_train,generate_val
from dump_cnn_features import stack_and_dump_data,getshapes
from SHUFFLING_SCRIPT import shuffle_data
from  TRAIN_AND_PREDICT import train_and_predict_cnn,train_and_predict


createParamsFile()

modelParamsFile  ='modelParams.yaml'
featureParamsFile = 'featureParams.yaml'



modelParams = loadfeatureParams(modelParamsFile)
featureParams = loadfeatureParams(featureParamsFile)







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

models = ['cnn']
events = ['babycry']

global_metrics['dnn'] = dnnmetrics
global_metrics['cnn'] = cnnmetrics

tuningMode = modelParams['tuning_mode']



directory = None
model  = 'cnn'
for event in events:

    evaluate = False

    #directory = 'cnn_07_01_PM_July_17_2017'  
    directory = modelParams[model][event]['directory']

    if modelParams['train_feature_ext']:

        print("Starting TrainData Feature Extraction....")
        timestamp = do_feature_extraction( model, event, source_data = False, mode = 'devtrain',
                                            directory = directory, mixture_data=True)
        print("Done with TrainData Feature Extraction !")

        directory = timestamp

    if modelParams['train_feature_norm']:
        print("Starting TrainData Feature Normalization....")
        ignore = do_feature_normalization(model, event, directory, source_data = True,
                                             mode = 'devtrain', mixture_data = True)
        print("Done with  Feature Normalization !")



    if modelParams['train_label_ext']:
        print("Starting TrainData Label Extraction....")
        ignore = do_label_extraction(directory, model,event, source_data = False,
                                     mode = 'devtrain', mixture_data = True )
        print("Done With TrainData Label Extraction!")

    if modelParams['test_feature_ext']:
        print("Starting Feature Extraction for Validation data....")
        directory = do_feature_extraction( model, event, source_data = False,
                                     mode = 'devtest', directory = directory)
        print("Done with  Feature Extraction !")

    if modelParams['test_feature_norm']:
        print("Starting Feature Normalization for validation data....")
        ignore = do_feature_normalization(model, event, directory, source_data = False,
                                             mode = 'devtest')
        print("Done with  Feature Normalization !")

    if modelParams['test_label_ext']:
        print("Starting Label Extraction for validation data....")
        ignore = do_label_extraction(directory, model,event, source_data = False,
                                     mode = 'devtest', mixture_data = True)
        print("Done With Label Extraction!")


    if model == 'cnn':
        print()

        trainFeatureFile = ''
        trainLabelFile = ''
        if modelParams['dump_cnn_train_data']:
            print("Dumping All train data to a single file...")
            trainFeatureFile, trainLabelFile = stack_and_dump_data(directory, event,
                                                     mode = 'devtrain',
                                                source_data = False, mixture_data = True,
                                                    normalized_data = False)

            print("Done!")

        if modelParams['dump_cnn_test_data']:
            print("Dumping All validation data to a single file...")
            featureFile, labelFile = stack_and_dump_data(directory, event, mode = 'devtest',
                                                        source_data = False,
                                                        mixture_data = True, 
                                                        normalized_data = False)

            print("Done!")

        if modelParams['shuffle_data']:
            print("Shuffling Training data...")
            shuffle_data(trainFeatureFile,trainLabelFile, 373, 
                        1964, directory, event)
            print("Done Shuffling! ")

    elif model =='dnn':
        
        print()      
        # print("Loading Training Data....")
        # xTrain,yTrain,fileList,numFramesList = loadKerasData(model, directory, event,
        #                                         mode = 'devtrain', source_data = False, 
        #                                         normalized_data = False)
        # print("Done Loading Training Data!")



        # print("Loading Validation Data....")
        # xVal,yVal,fileList,numFramesList = loadKerasData(model, directory, event,
        #                 mode = 'devtest', source_data = False, normalized_data = False)
        # print("Done Loading Validation Data!")


        # predictOnData = xVal

        











        
    evaluate = True 

    
    fileList = []
    numFramesList = []
    print("Training System!")
    if model == 'cnn':
        print("Training  CNN!")
        fileList,numFramesList = get_files_list(model, directory, event, numofFiles = float('inf'),
                                 mode = 'devtest', source_data = False,
                                    mixture_data = True, normalized_data = False)

        #print(numFramesList)
        #sys.exit()



        predictions,history = train_and_predict_cnn(model,event,modelParams,directory)

        print("done fetching predictions!")



    elif model == 'dnn':
        print("Training  DNN!")
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







'''
       do_feature_extraction(model, event, source_data = False, mode = 'eval')
       do_feature_normalization(model, event, directory, source_data = False, mode = 'eval')
       predictOnData, fileList,numFramesList = loadKerasData(model, directory, event,
                   numofFiles = float('inf'), mode = 'eval', source_data = False)

       evaluate = False 
    '''

''' tuning module begins '''
# printlist = []
# if tuningMode:
#     #same as train and predict but with varying parameters
    

#     print("Tuning Hyperparameters...")
#     for count in range(modelParams['tuning_countmax']):
        
        
#         reg = 10**np.random.uniform(-5,5)
#         lr = 10**np.random.uniform(-2,-5)

#         modelParams[model][event]['l2reg'] = reg
#         modelParams[model][event]['lr'] = lr

#         print("Fetching Model Predictions...")
#         predictions,history = train_and_predict(xTrain,yTrain,xVal,yVal,model,event,modelParams,
#                                         predictOnData = xVal)

#         print("shape of predictions: ", predictions.shape)

#         print("Preparing Annotated File...")
#         userAnnotatedFile = process_predictions(predictions, model, directory,
#                          featureParams, modelParams,
#                          event, numFramesList, fileList) #produce files

#         print("done Producing annotated Files! ")

#         if evaluate:
            

#             pathToFile = os.path.join(os.getcwd(),
#                          '../data/mixture_data/devtest/20b255387a2d0cddc0a3dff5014875e7/meta')
#             listofFiles  = os.listdir(pathToFile) 
#             filename = 'event_list_devtest_'+event +'.csv'
#             groundTruthFile = os.path.join(pathToFile, filename)

#             er, fscore,fp,fn,ins,dele,tp= get_metrics(userAnnotatedFile, groundTruthFile, event) #produce metrics
#             global_metrics[model][event]['er'] = er
#             global_metrics[model][event]['fscore'] = fscore

#             l = ['reg:', reg, 'lr:', lr, 'er:', er, 'fscore:', fscore,history]
#             printlist.append(l)


#             #print("model: ", model, "event: ", event, "er :",er , "Fscore :",fscore )



#     print("Done tuning hyperparameters...")

#     for list_item in printlist:
#         print(list_item)

#     sys.exit()

'''tuning moodule ends'''


























