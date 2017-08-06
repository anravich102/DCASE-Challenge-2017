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
#from test_batchwise_training import generate_pred,generate_train,generate_val
from DUMP_FEATURES_DNN_CNN import stack_and_dump_data_cnn,getshapes,stack_and_dump_data_dnn,stack_and_dump_data_rnn
from DUMP_FEATURES_DNN_CNN import stack_and_dump_data_cnn_new,stack_and_dump_data_dnn_new,stack_and_dump_data_rnn_new
from SHUFFLING_SCRIPT import shuffle_data_new
#from  TRAIN_AND_PREDICT import train_and_predict_dnn,train_and_predict_cnn
from  TRAIN_AND_PREDICT import train_and_predict_dnn_save,train_and_predict_cnn_save,train_and_predict_rnn_save

createParamsFile()

modelParamsFile  ='modelParams.yaml'
featureParamsFile = 'featureParams.yaml'



modelParams = loadfeatureParams(modelParamsFile)
featureParams = loadfeatureParams(featureParamsFile)



def get_results(userAnnotatedFile,userAnnotatedFile2, userAnnotatedFile3 = None):

    print("directory: ", directory)
    print("VAL SET RESULTS:")
    pathToFile = os.path.join(os.getcwd(), '../data/mixture_data/devtest/20b255387a2d0cddc0a3dff5014875e7/meta')
    listofFiles  = os.listdir(pathToFile) 
    filename = 'event_list_devtest_'+event +'.csv'
    groundTruthFile = os.path.join(pathToFile, filename)

    er, fscore,fp,fn,ins,dele,tp= get_metrics(userAnnotatedFile, groundTruthFile, event) #produce metrics
    

    print("fp:",fp,"fn:",fn, "ins:", ins, "dele", dele, "tp:", tp)
    print("model: ", model)
    print("event: ", event)
    print("er :",er )
    print("Fscore :",fscore)

    metricsFileName = model + '_' + directory + event + '_VAL_METRICS.txt'
    metricsFilePath = os.path.join(os.getcwd(),'../Results',metricsFileName)

    with open(metricsFilePath,'w') as f:
        f.write(model + "\t" + event + "\t" + 'er:' + "\t" + str(er) + '\t' + 
                'fscore: '+ str(fscore) + "\t" + "\n")






    print("TEST SET RESULTS:")
    print("directory: ", directory)
    pathToFile = os.path.join(os.getcwd(), '../data/mixture_data/newTest/81fc1201069905c5926c9c5c8369eacf/meta')
    listofFiles  = os.listdir(pathToFile) 
    filename = 'event_list_devtrain_'+event +'.csv'
    groundTruthFile = os.path.join(pathToFile, filename)

    er, fscore,fp,fn,ins,dele,tp= get_metrics(userAnnotatedFile2, groundTruthFile, event) #produce metrics
   
    print("fp:",fp,"fn:",fn, "ins:", ins, "dele", dele, "tp:", tp)
    print("model: ", model)
    print("event: ", event)
    print("er :",er )
    print("Fscore :",fscore)

    metricsFileName = model + '_' + directory + event + '_TEST_METRICS.txt'
    metricsFilePath = os.path.join(os.getcwd(),'../Results',metricsFileName)

    with open(metricsFilePath,'w') as f:
        f.write(model + "\t" + event + "\t" + 'er:' + "\t" + str(er) + '\t' + 
                'fscore: '+ str(fscore) + "\t" + "\n")


    # if userAnnotatedFile3 is not None: 
    #     print("ENSEMBLE TEST SET RESULTS:")
    #     print("directory: ", directory)
    #     er, fscore,fp,fn,ins,dele,tp= get_metrics(userAnnotatedFile3, groundTruthFile, event) #produce metrics

    #     print("fp:",fp,"fn:",fn, "ins:", ins, "dele", dele, "tp:", tp)
    #     print("model: ", model)
    #     print("event: ", event)
    #     print("er :",er )
    #     print("Fscore :",fscore)

    #     metricsFileName = model + '_' + directory + event + '_TEST_ENSEMBLE_METRICS.txt'
    #     metricsFilePath = os.path.join(os.getcwd(),'../Results',metricsFileName)

    #     with open(metricsFilePath,'w') as f:
    #         f.write(model + "\t" + event + "\t" + 'er:' + "\t" + str(er) + '\t' + 
    #                 'fscore: '+ str(fscore) + "\t" + "\n")


	       	





tuningMode = modelParams['tuning_mode']



directory = None
model  = 'dnn'
events = ['glassbreak']
print(events)

num_models = 1

print("num_Models: ", num_models)

PPFileList2 = []

for i in range(num_models):
    for event in events:

        evaluate = False

        #directory = 'cnn_07_01_PM_July_17_2017'  
        directory = modelParams[model][event]['directory']

        print("directory:", directory)


        if modelParams['train_feature_ext']:

            print("Starting TrainData Feature Extraction....")
            timestamp = do_feature_extraction( model, event, source_data = False, 
                                                mode = 'devtrain',
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
                                         mode = 'devtrain', mixture_data = True)
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




        if modelParams['newTest_feature_ext']:
            print("Starting Feature Extraction for newTest data....")
            directory = do_feature_extraction( model, event, source_data = False,
                                         mode = 'newTest', directory = directory,)
            print("Done with  Feature Extraction !")


        if modelParams['newTest_label_ext']:
            print("Starting Label Extraction for newTest  data....")
            ignore = do_label_extraction(directory, model,event, source_data = False,
                                         mode = 'newTest', mixture_data = True)
            print("Done With Label Extraction!")


        if modelParams['eval_feature_ext']:
            print("Starting Feature Extraction for EVAL data....")
            directory = do_feature_extraction( model, event, source_data = False,
                                         mode = 'eval', directory = directory,)
            print("Done with  Feature Extraction !")




        # sys.exit()

        if model == 'cnn':
            print()

            trainFeatureFile = ''
            trainLabelFile = ''
            if modelParams['dump_cnn_train_data']:
                print("Dumping All train data to a single file...")
                trainFeatureFile, trainLabelFile = stack_and_dump_data_cnn(directory, event,
                                                         mode = 'devtrain',
                                                    source_data = False, mixture_data = True,
                                                        normalized_data = False)

                print("Done!")

               

            if modelParams['dump_cnn_test_data']:
                print("Dumping All validation data to a single file...")
                featureFile, labelFile = stack_and_dump_data_cnn(directory, event, 
                                                            mode = 'devtest',
                                                            source_data = False,
                                                            mixture_data = True, 
                                                            normalized_data = False)

                print("Done!")




            if modelParams['dump_cnn_newTest_data']:
                print("Dumping All newTest data to a single file...")
                featureFile2, labelFile2 = stack_and_dump_data_cnn(directory, event, 
                                                            mode = 'newTest',
                                                            source_data = False,
                                                            mixture_data = True, 
                                                            normalized_data = False)

                print("Done!")


            if modelParams['dump_cnn_eval_data']:
                print("Dumping All EVAL data to a single file...")
                featureFile2 = stack_and_dump_data_cnn_new(directory, event, 
                                                            mode = 'eval',
                                                            source_data = False,
                                                            mixture_data = True, 
                                                            normalized_data = False)

                print("Done!")

            # print("time to edit param file")
            # sys.exit()

            if modelParams['shuffle_data']:
                print("Shuffling Training data...")
                shuffle_data(trainFeatureFile,trainLabelFile, 373, 
                            1964, directory, event)
                print("Done Shuffling! ")





        elif model =='dnn':
            
            print()      

            trainFeatureFile = ''
            trainLabelFile = ''
            if modelParams['dump_dnn_train_data']:
                print("Dumping All train data to a single file...")
                trainFeatureFile, trainLabelFile = stack_and_dump_data_dnn(directory, event,
                                                         mode = 'devtrain',
                                                    source_data = False, mixture_data = True,
                                                        normalized_data = False)

                print("Done!")

            if modelParams['dump_dnn_test_data']:
                print("Dumping All validation data to a single file...")
                featureFile, labelFile = stack_and_dump_data_dnn(directory, event,
                                                             mode = 'devtest',
                                                            source_data = False,
                                                            mixture_data = True, 
                                                            normalized_data = False)

                print("Done!")

            if modelParams['dump_dnn_newTest_data']:
                print("Dumping All newTest data to a single file...")
                featureFile2, labelFile2 = stack_and_dump_data_dnn(directory, event,
                                                             mode = 'newTest',
                                                            source_data = False,
                                                            mixture_data = True, 
                                                            normalized_data = False)

                print("Done!")


            if modelParams['dump_dnn_eval_data']:
                print("Dumping All EVAL data to a single file...")
                featureFile2 = stack_and_dump_data_dnn_new(directory, event, 
                                                            mode = 'eval',
                                                            source_data = False,
                                                            mixture_data = True, 
                                                            normalized_data = False)

                print("Done!")

            # print("time to edit param file!")
            # sys.exit()


            if modelParams['shuffle_data']:


                #neglect = input("Enter the number of samples you want to discard: ")
                #total_iterations = input("Enter total iterations for shuffling data ")
                neglect = 0

                print("Shuffling Training data...")
                shuffle_data_new(trainFeatureFile,trainLabelFile,
                                    directory, event, neglect)
                print("Done Shuffling! ")




        elif model == 'rnn':
            print()

            trainFeatureFile = ''
            trainLabelFile = ''
            if modelParams['dump_rnn_train_data']:
                print("Dumping All train data to a single file...")
                trainFeatureFile, trainLabelFile = stack_and_dump_data_rnn(directory, event,
                                                         mode = 'devtrain',
                                                    source_data = False, mixture_data = True,
                                                        normalized_data = False)

                print("Done!")

            if modelParams['dump_rnn_test_data']:
                print("Dumping All validation data to a single file...")
                featureFile, labelFile = stack_and_dump_data_rnn(directory, event, 
                                                            mode = 'devtest',
                                                            source_data = False,
                                                            mixture_data = True, 
                                                            normalized_data = False)

                print("Done!")


            if modelParams['dump_rnn_newTest_data']:
                print("Dumping All newTest data to a single file...")
                featureFile2, labelFile2 = stack_and_dump_data_rnn(directory, event, 
                                                            mode = 'newTest',
                                                            source_data = False,
                                                            mixture_data = True, 
                                                            normalized_data = False)

                print("Done!")

            if modelParams['dump_rnn_eval_data']:
                print("Dumping All EVAL data to a single file...")
                featureFile2 = stack_and_dump_data_rnn_new(directory, event, 
                                                            mode = 'eval',
                                                            source_data = False,
                                                            mixture_data = True, 
                                                            normalized_data = False)

                print("Done!")


            if modelParams['shuffle_data']:
                print("Shuffling Training data...")
                shuffle_data(trainFeatureFile,trainLabelFile, 373, 
                            1964, directory, event)
                print("Done Shuffling! ")

        evaluate = True 


        # if modelParams['tuning_mode']:
        #     run_tuning_module(modelParams, model, directory, event)

        
        fileList = []
        numFramesList = []
        fileList2 = []
        numFramesList2 = []
        print("Training System!")

        if modelParams[model][event]['eval_predict']:
            fileList_pred,numFramesList_pred = get_files_list(model, directory, event, numofFiles = float('inf'),
                                 mode = 'eval', source_data = False,
                                    mixture_data = True, normalized_data = False)


        if model == 'cnn':
            print("Training  CNN!")
            fileList,numFramesList = get_files_list(model, directory, event, numofFiles = float('inf'),
                                     mode = 'devtest', source_data = False,
                                        mixture_data = True, normalized_data = False)

            fileList2,numFramesList2 = get_files_list(model, directory, event, numofFiles = float('inf'),
                                     mode = 'newTest', source_data = False,
                                        mixture_data = True, normalized_data = False)


            







            predictions,predictions2, eval_pred, history = train_and_predict_cnn_save(model,event,modelParams,
                                                            directory,index = i)

            print("done fetching predictions!")


            PPFileList2.append(save_processed_predictions(predictions2, model, directory, 
                                            featureParams, modelParams,
                                            event, numFramesList2, fileList2))



        elif model == 'dnn':
            print("Training  DNN!")

            fileList,numFramesList = get_files_list(model, directory, event,
                                     numofFiles = float('inf'),
                                     mode = 'devtest', source_data = False,
                                        mixture_data = True, normalized_data = False)

            fileList2,numFramesList2 = get_files_list(model, directory, event, numofFiles = float('inf'),
                             mode = 'newTest', source_data = False,
                                mixture_data = True, normalized_data = False)

            # print("time to edit param file")
            # sys.exit()
            





            predictions,predictions2,eval_pred, history = train_and_predict_dnn_save(model,event,
                                                            modelParams,directory,index = i)

            print("done fetching predictions!")

           
            PPFileList2.append(save_processed_predictions(predictions2, model, directory, 
                                            featureParams, modelParams,
                                            event, numFramesList2, fileList2))

        


        elif model == 'rnn':
            print("Training  RNN!")
            fileList,numFramesList = get_files_list(model, directory, event, numofFiles = float('inf'),
                                     mode = 'devtest', source_data = False,
                                        mixture_data = True, normalized_data = False)

            fileList2,numFramesList2 = get_files_list(model, directory, event, numofFiles = float('inf'),
                                     mode = 'newTest', source_data = False,
                                        mixture_data = True, normalized_data = False)

            # print("time to edit param file")
            # sys.exit()


            predictions,predictions2, eval_pred, history = train_and_predict_rnn_save(model,event,modelParams,directory,index = i)

            print("done fetching predictions!")


            PPFileList2.append(save_processed_predictions(predictions2, model, directory, 
                                            featureParams, modelParams,
                                            event, numFramesList2, fileList2))



        if modelParams['postprocess_tune'] == True:

            pathToFile = save_raw_predictions(predictions, model, directory, 
                                        featureParams, modelParams,
                                        event, numFramesList, fileList)
            ans = input('Saved raw predictions to file. tune parameters?')
            if ans=='y':
                print("Saved File for post process tuning. Tuning PP parameters...")
                simulate(pathToFile)
                print("Done! exiting...")
                sys.exit()



       

        print("Preparing VAL Annotated File...")
        userAnnotatedFile = process_predictions(predictions, model, directory,
                                         featureParams, modelParams,
                             event, numFramesList, fileList, mode = 'VAL') #produce files

        print("VAL Predictions saved to: ", userAnnotatedFile)


        print("Preparing TEST Annotated File...")
        userAnnotatedFile2 = process_predictions(predictions2, model, directory,
                                         featureParams, modelParams,
                             event, numFramesList2, fileList2, mode = 'TEST') #produce files
        print("TEST Predictions saved to: ", userAnnotatedFile2)

        


        if modelParams[model][event]['eval_predict']:
            print("Preparing EVAL_FINAL Annotated File...")
            userAnnotatedFileFinal  = process_predictions(eval_pred, model, directory,
                                         featureParams, modelParams,
                             event, numFramesList_pred, fileList_pred, mode = 'FINAL_EVAL')

            print("Final Predictions saved to: ", userAnnotatedFileFinal)


        # if i==(num_models-1):
        #     print("ensembling predictions...")
        #     ensemblePredictions = ensemble_similar_predictions(PPFileList2, type = 'average_before')

        #     print("Preparing ENSEMBLE_TEST Annotated File...")
        #     userAnnotatedFile3 = process_predictions(ensemblePredictions, model, directory,
        #                                         featureParams, modelParams,
        #                                          event, numFramesList2, fileList2, mode = 'TEST_ENSEMBLE')


        print("done Producing annotated Files! ")

        if evaluate:
           
            get_results(userAnnotatedFile,userAnnotatedFile2)
            # else:
        	   #  get_results(userAnnotatedFile,userAnnotatedFile2, userAnnotatedFile3)

            
        







