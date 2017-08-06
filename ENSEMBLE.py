import numpy as np 
import os,sys
from PARAMS import *
from LOADKERASDATA import *
from TRAIN_AND_PREDICT import *
import yaml
import deepdish as dd
import h5py


# x = 'dnn_11_18_AM_August_02_2017_gunshot_12_45_PM_August_02_2017_ModelFile.h5'


createParamsFile()

modelParams = loadfeatureParams('modelParams.yaml')
featureParams = loadfeatureParams('featureParams.yaml')

def save_predictions(predictions, model, directory, featureParams, modelParams,
                         event, numFramesList, fileList, mode = None):
    rawFile = directory+event+ mode +  '_raw_Predictions.h5'

    d = { 'predictions' : predictions,
           'model':model,
           'directory':directory,
           'featureParams':featureParams,
           'modelParams' :modelParams,
           'event':event,
           'numFramesList':numFramesList,
           'fileList':fileList


    }

    
    dd.io.save(rawFile, d)

    return rawFile


def ensemble_similar_predictions(ListofFiles, type = 'average_before', weights = None, numFrames = None
                                        model = None):

    name in ListofFiles[0]
    f = dd.io.load(name)

    total_length = f['predictions'].size
    iterations = int(total_length/numFrames)

    file_to_return = str(numFrames)+

    with h5py.File(pathToFeatureFile, "w") as predFile:
        for i in range(iterations):
            temp = []
            for name in ListofFiles:
                f = dd.io.load(name)
                temp.append(ffile['features'][i*numFrames:(i+1)*numFrames])

            toDump = sum(temp)/len(temp) #list of array
            toDump = toDump[0]

            if(i==0):
                dsetl = lfile.create_dataset("labels", data = reloadedLabels , shape = 
                                                    (numImages,1),
                                                    maxshape=(None, 1))



    
    return sum(temp)/len(temp)


def post_process(predictionsForFile, classify_threshold,kernel_size):
    

    classify = [1.0 if x > classify_threshold else 0.0 for x in predictionsForFile ]

    return scipy.signal.medfilt(classify,kernel_size = kernel_size)

def get_times(pp,file,numFramesInFile, event, model,
                                     modelParams, featureParams):
    
   

   
    if np.array_equal(pp, np.zeros(pp.shape)):
        startTime = -1.0
        endTime = -1.0
    else:
        y = np.where(pp == 1.0)



        index = y[0][0]
        endIndex = y[0][-1]
        startTime = index * ((100.0 - featureParams[model][event]['percent_overlap'])/100.0)*\
                        featureParams[model][event]['win_length']

        endTime = endIndex * ((100.0 - featureParams[model][event]['percent_overlap'])/100.0)*\
                        featureParams[model][event]['win_length']

    

 
    
    return startTime, endTime

def write_output_file(files, startTimes, endTimes , directory ,event, 
                        modelParams, featureParams, mode = None):

    '''start time contains -1 if event did not occur'''
    '''end time contains final window timestamp if event continues till end'''
    

    #first make the required directories:
    resultsDir = os.path.join(os.getcwd(), '../Results')
    if not os.path.exists(resultsDir):
        os.makedirs(resultsDir)
    

    annotatedFile = directory + '_' + event + '_' + mode + '.csv'

    pathToSave = os.path.join(resultsDir,annotatedFile)

    #length = len(files)
   
    length = 500

    mainList = [0]*length

    for file, start, end in zip(files, startTimes,endTimes):
        index =  int(file.split("_")[3])
        #print (index)
        
        mainList[index] = [file, start,end]

    

    with open(pathToSave,'w') as f:
        for item in mainList:
            if item == 0:
                continue
            file = item[0]
            start = float(item[1])
            end = float(item[2])
            if(start != -1.0):
                f.write(file + "\t" + str(round(start, 4)) + "\t" + str(round(end, 4)) + "\t" + event + "\n")
            else:
                f.write(file +  "\n")




    # save parameters  of the currrent model (using same name as  directory name)
    # for reference later:
    file1 = directory+'modelParams.yaml'
    pathToFile1 = os.path.join(resultsDir, file1)

    file2 = directory+'featureParams.yaml'
    pathToFile2 = os.path.join(resultsDir, file2)

    DumpfeatureParams(modelParams,pathToFile1)

    DumpfeatureParams(featureParams,pathToFile2)
    

    return pathToSave

def get_metrics(userAnnotatedFile, groundTruthFile, event):

    ins = 0
    dele = 0
    fp = fn = tp = 0.0
    total = 0.0

    with open(userAnnotatedFile) as userfile, open(groundTruthFile) as truthfile: 
        for x, y in zip(userfile, truthfile):

            userline = x.strip().split()
            truthline = y.strip().split()


            if(len(userline) > len(truthline) ):
                fp += 1
                ins += 1
                total+=1

            elif( len(truthline) > len(userline)):
                
                if(len(truthline) == 4):
                    fn+=1
                    dele+=1
                    total += 1

            elif( len(truthline) == len(userline) and len(truthline) == 4):
                total+=1
                #if event within collar
                collar = abs(float(userline[1]) - float(truthline[1]))
                if collar <= 0.5 and userline[3] == truthline[3]:
                    tp += 1


                #not within collar, but an event is present: Clarify this!
                else:
                    if(float(userline[1]) > float(truthline[1])):  #detected after actual event .missed actual event
                        dele+=1
                        fn+=1
                        
                    else:                                         #detected before actual event started
                        fp+=1
                        ins+=1
            else:
                total+=1
                        

    er = (ins+dele)/total
    Fscore = 2*tp/ (2*tp + fp + fn)


    return er, Fscore,fp,fn,ins,dele,tp


def print_results(er, Fscore,fp,fn,ins,dele,tp):
    print("fp:",fp,"fn:",fn, "ins:", ins, "dele", dele, "tp:", tp)
    print("model: ", model)
    print("event: ", event)
    print("er :",er )
    print("Fscore :",fscore)


def do_ensembling(modelfilesList,models_present = ['dnn','cnn','rnn']):   #from the same events
    
    predFiles  = {'dnn': [], 'cnn': [], 'rnn': []}

    pred2Files = {'dnn': [], 'cnn': [], 'rnn': []}

    pred3Files = {'dnn': [], 'cnn': [], 'rnn': []}

    predbyModel = {'dnn': '', 'cnn': '', 'rnn': ''}
    pred2byModel = {'dnn': '', 'cnn': '', 'rnn': ''}
    pred3byModel = {'dnn': '', 'cnn': '', 'rnn': ''}

    events = ['gunshot','glassbreak','babycry']
    models = ['dnn','cnn','rnn']

    flagDict = {'dnn': False, 'cnn':False, 'rnn':False}
    numFramesDict = {'dnn': 0, 'cnn':0, 'rnn':0}

    listofdict = [predFiles, pred2Files,pred3Files]



    eventname = ''

    j = 0
    for x in modelfilesList:
        j = j+1 #to be used for string identifier
        ss = x.split("_")
        
        print("processing " , x )
        

        for event in events:
            if event in ss:
                new_ss = x.split(event)
                directoryname = new_ss[0][:-1]
                modelname = directoryname[:3]
                eventname = event
                break

        print("model: ", modelname)

        if (flagDict[modelname] == False):
            print("getting filelist ... ")
            fileList,numFramesList = get_files_list(modelname, directoryname, eventname, numofFiles = float('inf'),
                                         mode = 'devtest', source_data = False,
                                            mixture_data = True, normalized_data = False)

            fileList2,numFramesList2 = get_files_list(modelname, directoryname, eventname, numofFiles = float('inf'),
                                     mode = 'newTest', source_data = False,
                                        mixture_data = True, normalized_data = False)

            fileList3,numFramesList3 = get_files_list(modelname, directoryname, eventname, numofFiles = float('inf'),
                                     mode = 'eval', source_data = False,
                                        mixture_data = True, normalized_data = False)

            flagDict[modelname] = True
            numFramesDict[modelname] = numFramesList[0]


        print("Done!")
        print("Fetching Predictions....")

        if modelname == 'dnn':                  #just loads the file and gives out predictions
            predictions,predictions2, predictions3,history = train_and_predict_dnn_save(modelname,eventname,modelParams,directoryname,
                                                        use_input_model_file = True, input_model_file = x)
        elif modelname == 'cnn':
            predictions,predictions2, predictions3, history = train_and_predict_cnn_save(modelname,eventname,modelParams,directoryname,
                                                        use_input_model_file = True, input_model_file = x)
        elif modelname == 'rnn':
            predictions,predictions2, predictions3, history = train_and_predict_rnn_save(modelname,eventname,modelParams,directoryname,
                                                        use_input_model_file = True, input_model_file = x)


        print("Done. Saving predictions ...")
        fileID = str(j)+'VAL'
        fileID2 = str(j)+'TEST'
        fileID3 = str(j)+'EVAL'
        savedfile = save_predictions(predictions, modelname, directoryname, featureParams, modelParams,
                         eventname, numFramesList, fileList, mode = fileID)


        savedfile2=save_predictions(predictions2, modelname, directoryname, featureParams, modelParams,
                         eventname, numFramesList2, fileList2, mode = fileID2)

        savedfile3=save_predictions(predictions3, modelname, directoryname, featureParams, modelParams,
                         eventname, numFramesList3, fileList3, mode = fileID3)




        print("done! ")

        predFiles[modelname].append(savedfile)
        pred2Files[modelname].append(savedfile2)
        pred3Files[modelname].append(savedfile3)


    print("ensembling similar predictions...")
    #first ensemble predictions of similar models by averaging

    for index,currentDict in enumerate(listofdict):

        for model in models_present:


            if currentDict == predFiles:
                print("prediction files: ", currentDict[model])
                predbyModel[model] =  ensemble_similar_predictions(currentDict[model], type = 'average_before',
                                                    weights  = None, numFrames = numFramesDict[model])

            elif currentDict == pred2Files:
                pred2byModel[model] =  ensemble_similar_predictions(currentDict[model], type = 'average_before',
                                            weights = None)

            elif currentDict == pred3Files:
                pred3byModel[model] =  ensemble_similar_predictions(currentDict[model], type = 'average_before',
                                            weights = None)



    #now we have a 2 sets of predictions for very model- one for val, one for test.


    #save the results of each model:
    if 'dnn' in models_present and 'cnn' not in models_present and 'rnn' not in models_present: #only dnn

    #first postprocess all files for dnn
        dnn_pp = []
       
        
        startTimes = []
        endTimes = []
        
        index_dnn = 0
        
        for file in fileList:
            
            
            predictionsForFile = predbyModel['dnn'][index_dnn:index_dnn+numFramesDict['dnn']]
            index_dnn = index_dnn + numFramesDict['dnn']

            classify_threshold = modelParams['dnn'][eventname]['classify_threshold']
            # pass predictionForFile as an array 
            kernel_size = modelParams['dnn'][eventname]['medfilt_kernel']

            dnnppForFile = post_process(predictionsForFile, classify_threshold,kernel_size)
            dnn_pp.append( dnnppForFile)
        
       

            startTime,endTime = get_times(dnn_pp,file,numFramesDict['dnn'], eventname, 'dnn',
                                     modelParams, featureParams)


            startTimes.append(startTime)
            endTimes.append(endTime)



    #now for test set

    dnn_pp = []
    
    startTimes2 = []
    endTimes2 = []
    

    index_dnn = 0
 
    for file in fileList2:
        
        
        predictionsForFile = pred2byModel['dnn'][index_dnn:index_dnn+numFramesDict['dnn']]
        index_dnn = index_dnn+numFramesDict['dnn']

        classify_threshold = modelParams['dnn'][eventname]['classify_threshold']
        # pass predictionForFile as an array 
        kernel_size = modelParams['dnn'][eventname]['medfilt_kernel']

        dnnppForFile = post_process(predictionsForFile, classify_threshold,kernel_size)
        dnn_pp.append( dnnppForFile)
    


        startTime,endTime = get_times(dnn_pp,file,numFramesDict['dnn'], eventname, 'dnn',
                                 modelParams, featureParams)


        startTimes2.append(startTime)
        endTimes2.append(endTime)


    # now for eval set:
    dnn_pp = []
    
    startTimes3 = []
    endTimes3 = []
    

    index_dnn = 0
 
    for file in fileList3:
        
        
        predictionsForFile = pred3byModel['dnn'][index_dnn:index_dnn+numFramesDict['dnn']]
        index_dnn = index_dnn+numFramesDict['dnn']

        classify_threshold = modelParams['dnn'][eventname]['classify_threshold']
        # pass predictionForFile as an array 
        kernel_size = modelParams['dnn'][eventname]['medfilt_kernel']

        dnnppForFile = post_process(predictionsForFile, classify_threshold,kernel_size)
        dnn_pp.append( dnnppForFile)
    


        startTime,endTime = get_times(dnn_pp,file,numFramesDict['dnn'], eventname, 'dnn',
                                 modelParams, featureParams)


        startTimes3.append(startTime)
        endTimes3.append(endTime)


    if 'dnn' not in models_present and 'cnn' in models_present and 'rnn' not in models_present:  #only cnn

        #first postprocess all files for dnn
        dnn_pp = []
       
        
        startTimes = []
        endTimes = []
        
        index_dnn = 0
        
        for file in fileList:
            
            
            predictionsForFile = predbyModel['cnn'][index_dnn:index_dnn+numFramesDict['cnn']]
            index_dnn = index_dnn + numFramesDict['cnn']

            classify_threshold = modelParams['cnn'][eventname]['classify_threshold']
            # pass predictionForFile as an array 
            kernel_size = modelParams['cnn'][eventname]['medfilt_kernel']

            dnnppForFile = post_process(predictionsForFile, classify_threshold,kernel_size)
            dnn_pp.append( dnnppForFile)
        
       

            startTime,endTime = get_times(dnn_pp,file,numFramesDict['cnn'], eventname, 'cnn',
                                     modelParams, featureParams)


            startTimes.append(startTime)
            endTimes.append(endTime)



        #now for test set

        dnn_pp = []
        
        startTimes2 = []
        endTimes2 = []
        

        index_dnn = 0
     
        for file in fileList2:
            
            
            predictionsForFile = predbyModel2['cnn'][index_dnn:index_dnn+numFramesDict['cnn']]
            index_dnn = index_dnn+numFramesDict['cnn']

            classify_threshold = modelParams['cnn'][eventname]['classify_threshold']
            # pass predictionForFile as an array 
            kernel_size = modelParams['cnn'][eventname]['medfilt_kernel']

            dnnppForFile = post_process(predictionsForFile, classify_threshold,kernel_size)
            dnn_pp.append( dnnppForFile)
        


            startTime,endTime = get_times(dnn_pp,file,numFramesDict['cnn'], eventname, 'cnn',
                                     modelParams, featureParams)


            startTimes2.append(startTime)
            endTimes2.append(endTime)


        #now for eval data:
        dnn_pp = []
        
        startTimes3 = []
        endTimes3 = []
        

        index_dnn = 0
     
        for file in fileList3:
            
            
            predictionsForFile = pred3byModel['cnn'][index_dnn:index_dnn+numFramesDict['cnn']]
            index_dnn = index_dnn+numFramesDict['cnn']

            classify_threshold = modelParams['cnn'][eventname]['classify_threshold']
            # pass predictionForFile as an array 
            kernel_size = modelParams['cnn'][eventname]['medfilt_kernel']

            dnnppForFile = post_process(predictionsForFile, classify_threshold,kernel_size)
            dnn_pp.append( dnnppForFile)
        


            startTime,endTime = get_times(dnn_pp,file,numFramesDict['cnn'], eventname, 'cnn',
                                     modelParams, featureParams)


            startTimes3.append(startTime)
            endTimes3.append(endTime)



    if 'dnn' not in models_present and 'cnn' not in models_present and 'rnn' in models_present:  #only rnn

        #first postprocess all files for dnn
        dnn_pp = []
       
        
        startTimes = []
        endTimes = []
        
        index_dnn = 0
        
        for file in fileList:
            
            
            predictionsForFile = predbyModel['rnn'][index_dnn:index_dnn+numFramesDict['rnn']]
            index_dnn = index_dnn + numFramesDict['rnn']

            classify_threshold = modelParams['rnn'][eventname]['classify_threshold']
            # pass predictionForFile as an array 
            kernel_size = modelParams['rnn'][eventname]['medfilt_kernel']

            dnnppForFile = post_process(predictionsForFile, classify_threshold,kernel_size)
            dnn_pp.append( dnnppForFile)
        
       

            startTime,endTime = get_times(dnn_pp,file,numFramesDict['rnn'], eventname, 'rnn',
                                     modelParams, featureParams)


            startTimes.append(startTime)
            endTimes.append(endTime)



        #now for test set

        dnn_pp = []
        
        startTimes2 = []
        endTimes2 = []
        

        index_dnn = 0
     
        for file in fileList2:
            
            
            predictionsForFile = predbyModel2['rnn'][index_dnn:index_dnn+numFramesDict['rnn']]
            index_dnn = index_dnn+numFramesDict['rnn']

            classify_threshold = modelParams['rnn'][eventname]['classify_threshold']
            # pass predictionForFile as an array 
            kernel_size = modelParams['rnn'][eventname]['medfilt_kernel']

            dnnppForFile = post_process(predictionsForFile, classify_threshold,kernel_size)
            dnn_pp.append( dnnppForFile)
        


            startTime,endTime = get_times(dnn_pp,file,numFramesDict['rnn'], eventname, 'rnn',
                                     modelParams, featureParams)


            startTimes2.append(startTime)
            endTimes2.append(endTime)

        #now for eval data
        dnn_pp = []
        
        startTimes3 = []
        endTimes3 = []
        

        index_dnn = 0
     
        for file in fileList3:
            
            
            predictionsForFile = pred3byModel['rnn'][index_dnn:index_dnn+numFramesDict['rnn']]
            index_dnn = index_dnn+numFramesDict['rnn']

            classify_threshold = modelParams['rnn'][eventname]['classify_threshold']
            # pass predictionForFile as an array 
            kernel_size = modelParams['rnn'][eventname]['medfilt_kernel']

            dnnppForFile = post_process(predictionsForFile, classify_threshold,kernel_size)
            dnn_pp.append( dnnppForFile)
        


            startTime,endTime = get_times(dnn_pp,file,numFramesDict['rnn'], eventname, 'rnn',
                                     modelParams, featureParams)


            startTimes3.append(startTime)
            endTimes3.append(endTime)

    elif ('dnn' in models_present and 'cnn' in models_present and 'rnn' not in models_present):

        #first postprocess all files for dnn
        dnn_pp = []
        cnn_pp = []
        dnn_weight = 0.5
        cnn_weight = 0.5
        startTimes = []
        endTimes = []
        
        index_dnn = 0
        index_cnn = 0
        for file in fileList:
            
            
            predictionsForFile = predbyModel['dnn'][index_dnn:index_dnn+numFramesDict['dnn']]
            index_dnn = index_dnn + numFramesDict['dnn']

            classify_threshold = modelParams['dnn'][eventname]['classify_threshold']
            # pass predictionForFile as an array 
            kernel_size = modelParams['dnn'][eventname]['medfilt_kernel']

            dnnppForFile = post_process(predictionsForFile, classify_threshold,kernel_size)
            dnn_pp.append( dnnppForFile)
        
       
            
            
            predictionsForFile = predbyModel['cnn'][index_cnn:index_cnn+numFramesDict['cnn']]
            
            index_cnn = index_cnn + numFramesDict['dnn']
            classify_threshold = modelParams['cnn'][eventname]['classify_threshold']

            # pass predictionForFile as an array 
            kernel_size = modelParams['cnn'][eventname]['medfilt_kernel']
            cnnppForFile = post_process(predictionsForFile, classify_threshold,kernel_size) 
            cnn_pp.append(cnnppForFile)



            cnn_length = len(cnnppForFile)
            dnn_length = len(dnnppForFile)

            if(dnn_length > cnn_length):
                extra = (dnn_length-cnn_length) 
                temp = dnnppForFile[-extra:]
                dnnppForFile = dnnppForFile[:-extra]

                ensemblepredForFile = [(dnn_weight*x+cnn_weight*y) for x,y in zip(dnnppForFile,cnnppForFile)]

                pp_ensemble_for_file = post_process(ensemblepredForFile, 0.5 ,3)

            else:
                print("not written")
                sys.exit()



            startTime,endTime = get_times(pp_ensemble_for_file,file,numFramesDict['cnn'], eventname, 'cnn',
                                     modelParams, featureParams)


            startTimes.append(startTime)
            endTimes.append(endTime)



        #now for test set

        dnn_pp = []
        cnn_pp = []
        dnn_weight = 0.5
        cnn_weight = 0.5
        startTimes2 = []
        endTimes2 = []
        

        index_dnn = 0
        index_cnn = 0
        for file in fileList2:
            
            
            predictionsForFile = predbyModel2['dnn'][index_dnn:index_dnn+numFramesDict['dnn']]
            index_dnn = index_dnn+numFramesDict['dnn']

            classify_threshold = modelParams['dnn'][eventname]['classify_threshold']
            # pass predictionForFile as an array 
            kernel_size = modelParams['dnn'][eventname]['medfilt_kernel']

            dnnppForFile = post_process(predictionsForFile, classify_threshold,kernel_size)
            dnn_pp.append( dnnppForFile)
        
       
            
            
            predictionsForFile = predbyModel2['cnn'][index_cnn:index_cnn+numFramesDict['cnn']]
            
            index_cnn = index_cnn + numFramesDict['cnn']
            classify_threshold = modelParams['cnn'][eventname]['classify_threshold']

            # pass predictionForFile as an array 
            kernel_size = modelParams['cnn'][eventname]['medfilt_kernel']
            cnnppForFile = post_process(predictionsForFile, classify_threshold,kernel_size) 
            cnn_pp.append(cnnppForFile)



            cnn_length = len(cnnppForFile)
            dnn_length = len(dnnppForFile)

            if(dnn_length > cnn_length):
                extra = (dnn_length-cnn_length) 
                temp = dnnppForFile[-extra:]
                dnnppForFile = dnnppForFile[:-extra]

                ensemblepredForFile = [(dnn_weight*x+cnn_weight*y) for x,y in zip(dnnppForFile,cnnppForFile)]

                pp_ensemble_for_file = post_process(ensemblepredForFile, 0.5 ,3)

            else:
                print("not written")
                sys.exit()



            startTime,endTime = get_times(pp_ensemble_for_file,file,numFramesDict['cnn'], eventname, 'cnn',
                                     modelParams, featureParams)


            startTimes2.append(startTime)
            endTimes2.append(endTime)






    elif 'dnn' in models_present and 'cnn' in models_present and 'rnn' in models_present:
        pass


    elif 'cnn' in models_present and 'dnn' not in models_present and 'rnn' not in models_present:
        pass


    userAnnotatedFile = write_output_file(fileList, startTimes, endTimes , directoryname ,eventname, 
                        modelParams, featureParams, mode = 'ENSEMBLED_VAL')

    userAnnotatedFile2 = write_output_file(fileList2, startTimes2, endTimes2 , directoryname ,eventname, 
                    modelParams, featureParams, mode = 'ENSEMBLED_TEST')


    print("VAL SET RESULTS:")
    print("directory: ", directory)
    pathToFile = os.path.join(os.getcwd(), '../data/mixture_data/devtest/20b255387a2d0cddc0a3dff5014875e7/meta')
    listofFiles  = os.listdir(pathToFile) 
    filename = 'event_list_devtrain_'+event +'.csv'
    groundTruthFile = os.path.join(pathToFile, filename)

    er, Fscore,fp,fn,ins,dele,tp = get_metrics(userAnnotatedFile, groundTruthFile, eventname)

    print_results(er, Fscore,fp,fn,ins,dele,tp )


    print("TEST SET RESULTS:")
    print("directory: ", directory)
    pathToFile = os.path.join(os.getcwd(), '../data/mixture_data/newTest/81fc1201069905c5926c9c5c8369eacf/meta')
    listofFiles  = os.listdir(pathToFile) 
    filename = 'event_list_devtrain_'+event +'.csv'
    groundTruthFile = os.path.join(pathToFile, filename)

    er, Fscore,fp,fn,ins,dele,tp = get_metrics(userAnnotatedFile2, groundTruthFile, eventname)

    print_results(er, Fscore,fp,fn,ins,dele,tp)    




modelfilesList =  ['dnn_10_06_PM_August_02_2017_glassbreak_12_07_AM_August_03_2017_ModelFile.h5','dnn_10_06_PM_August_02_2017_glassbreak_11_52_PM_August_02_2017_ModelFile.h5']


do_ensembling(modelfilesList,models_present = ['dnn'])












