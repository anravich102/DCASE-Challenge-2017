# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 11:03:11 2017

@author: User
"""
from __future__ import print_function
import wave
import contextlib
import os
import h5py
import numpy as np
import math
import deepdish as dd
from builtins import input
from PARAMS import  loadfeatureParams,DumpfeatureParams
import sys



def do_label_extraction(pathFromFeature, model,event, source_data = False, mode = 'devtrain', 
                        mixture_data = True, newTest = False):
    '''do label extraction for a specific event and model'''

    #First make the required directories: 


    directory = pathFromFeature #model+timestamp string
    dirPath = os.path.join(os.getcwd(), '../Labels',directory)
    

    if not os.path.exists(dirPath):
        os.mkdir(dirPath)


    modePath = os.path.join(dirPath,mode)

    if not os.path.exists(modePath):
        os.mkdir(modePath)

    eventdir = os.path.join(modePath,event)
    if not os.path.exists(eventdir):
        os.mkdir(eventdir)

    sourceFeaturesDir = os.path.join(eventdir, 'source_labels')
    mixtureFeaturesDir = os.path.join(eventdir, 'mixture_labels')
    if not os.path.exists(sourceFeaturesDir):
        os.mkdir(sourceFeaturesDir)
    if not os.path.exists(mixtureFeaturesDir):
        os.mkdir(mixtureFeaturesDir)

    print("mixtureFeaturesDir", mixtureFeaturesDir)

    #load feature param YAML file:
    filename = 'featureParams.yaml' #present in the CWD
    featureParams = loadfeatureParams(filename)

    filename = 'modelParams.yaml'
    modelParams = loadfeatureParams(filename)

    # print(modelParams['extraction_mode'])
    # sys.exit()

    #now extract labels

    labelPath = ExtractAndDumpLabels( model, directory, featureParams, modelParams,
                event, source_data = source_data, mode = mode, mixture_data = mixture_data )


    return labelPath


def ExtractAndDumpLabels(model, directory, featureParams, modelParams,
                        event, source_data = False , mode = 'devtrain' , mixture_data = True):


    if  mode is 'devtrain':
        
        print("Dumping mixture  (training) labels ....")
    else:
       
        print("Dumping mixture  (testing) labels ....")

    WindowLength = featureParams[model][event]['win_length']
    PercentOverlap = featureParams[model][event]['percent_overlap'] 


    print("window length: ", WindowLength)

    TimeResolution = (100 - PercentOverlap)*WindowLength*0.01

    skip = { 'gunshot' : ['350745.wav','365256.wav','127846.wav'],
         'glassbreak' : ['233605.wav', '233605 (1).wav'], 
                         'babycry' :[] }
                


    if mixture_data == True:

        pathUptoMixture = os.path.join(os.getcwd(),'../data/mixture_data',mode)
            
        x = os.listdir(pathUptoMixture) 
        
        paramHashes = [] 




        if mode == 'newTest':
            if modelParams['extraction_mode'] == 'ask':
               
                for i in x:
                    str1 = 'consider '+ str(i) +'?'
                    userinput = input(str1)
                    if userinput == 'y':
                        paramHashes.append(i)      
            else:
                paramHashes = [y for y in x if y[0]=='8']

        else:
            
            if modelParams['extraction_mode'] == 'ask':
               
                for i in x:
                    str1 = 'consider '+ str(i) +'?'
                    userinput = input(str1)
                    if userinput == 'y':
                        paramHashes.append(i)      
            else:
                paramHashes = [y for y in x if (y[0]=='2' )]
                #paramHashes = [y for y in x if (y[0]=='2' or (y[0:1]=='95') or (y[0:1]=='17') )]

                #paramHashes = [y for y in x if ((y[0:1]=='95') or (y[0:1]=='17') )]


        #paramHashes = ['956e330cf71c8a37e535d751c371bbb3','17ef7d1e096f3c83c6f52dc175f20461']   
        print(paramHashes)       
        labelpath = os.path.join(os.getcwd(), '../Labels',directory, mode,
                                event,'mixture_labels')
        

                
        for paramHash in paramHashes:


            
            currPath = os.path.join(pathUptoMixture,paramHash)
            #print("curr path: {}".format(currPath))
            pathToMetaFiles = os.path.join(currPath,'meta')
            listofMetaDataFiles = os.listdir(pathToMetaFiles)
                
                    
            for metafile in listofMetaDataFiles:
                #print("metafile: {}".format(metafile))
                
                if not metafile.endswith('.csv'):
                    continue
                
                splitstring = metafile.split("_")
                
                
                if not event+'.csv' in splitstring:
                    continue
                
             
                metaFilePath = os.path.join(pathToMetaFiles, metafile)
                
                
                print("metaFilePath: {}".format(metaFilePath))
                    
                with open(metaFilePath,'r') as f:
                    
                    while True:
                        line = f.readline()
                        #print("line read: {}".format(line))
                        
                        
                        if not line:
                            break
                        else:
                            line = line.split('\t')
                            audioFile = line[0].strip('\n')
                            
                            audioFilesPath = os.path.join(currPath,'audio')
                            audioFilePath = os.path.join(audioFilesPath, audioFile)
                           
                            with contextlib.closing(wave.open(audioFilePath,'r')) as f1:
                                
                                frames = f1.getnframes()
                                rate = f1.getframerate()
                                duration = frames / float(rate)
                                samplesInFile = int( (duration - WindowLength)/TimeResolution)
                                                                
                                

                                if(len(line) >= 2):
                                    eventOnset = float(line[1])
                                    eventOffset = float(line[2])
                                    
                                    labels = np.zeros((samplesInFile, 1))
                                    
                                    startIndex =int( math.ceil( (eventOnset - WindowLength)/float(TimeResolution) ))
                                    #print("startIndex: {}".format(startIndex))
                                    
                                    if(eventOffset > (samplesInFile*TimeResolution + WindowLength) ):
                                        #event offset continues beyond file duration
                                        labels[startIndex:][:] = 1
                                    else:
                                        stopIndex = int(eventOffset/TimeResolution)
                                        labels[startIndex:stopIndex][:] = 1 #check this later
                                        #print("stopIndex: {}".format(stopIndex))
                                        
                                      
                                    
                                    
                                else: #no event
                                    labels = np.zeros((samplesInFile,1))
                                    
                                
                                #print('labels.shape for this file: {x}'.format(x = labels.shape) )

                                checkfile = 'LabelsCheck.txt'
                                if os.path.exists(checkfile):
                                    append_write = 'a' # append if already exists
                                else:
                                    append_write = 'w' 
                                with open(checkfile,append_write) as f1:
        							
                                    f1.write(audioFile + "\t"+ "num of ones: " + str(np.sum(labels)) + "\n")

                                # First, if the model is a dnn or cnn,
                                # check if label size matches feature size, else correct it:

                                #first get the size of the features of the corresponding file:
                                filename = audioFile[:-4]+'.h5'
                                featurePath = os.path.join(os.getcwd(),'../Features',directory,
                                              mode, event, 'mixture_features',filename)

                                if model == 'dnn' or model == 'rnn':

                                    

                                    reloadedFeatureDict = dd.io.load(featurePath)
                                    reloadedFeatures = reloadedFeatureDict['feat']    
                                    numFrames = reloadedFeatures.shape[0]

                                    context = modelParams[model][event]['context']

                                    contextLabels = np.empty((0,1))
                                    start = 0
                                    threshold = modelParams[model][event]['label_group_threshold'] 

                                    for start in range(numFrames):
                                        stop = start+context
                                        if np.mean(labels[start:stop,:]) > threshold:
                                            contextLabels = np.vstack((contextLabels, np.array([1]) ))
                                        else:
                                            contextLabels = np.vstack((contextLabels, np.array([0]) ))

                                                                
                                    #print(reloadedFeatures.shape)   

                                    if (numFrames != contextLabels.shape[0]): 
                                        print("Shapes dont match for file: ", audioFile)
                                        print("num frames (acc to features): ", numFrames)
                                        print("num calculates samples in File: ", contextLabels.shape[0])


                                        sys.exit()

                                    else:
                                        labelfilename = audioFile[:-4] + '.h5'
                                        labelfilepath = os.path.join(labelpath, labelfilename)
                                        
                                        print("creating Label File: ", labelfilename)
                                        
                                        with h5py.File(labelfilepath, 'w') as h5f:
                                            h5f.create_dataset('y', data=contextLabels)
                                            print("saving h5 file {}".format(labelfilename))
                                        


                                if model == 'cnn':

                                    reloadedFeatureDict = dd.io.load(featurePath)
                                    reloadedFeatures = reloadedFeatureDict['feat']    
                                    numImages = reloadedFeatures.shape[0]
                                    
                                    # if(labels.size != numImages):
                                    #     print("Shapes dont match for file: ", audioFile)
                                    #     print("Label.size : ", labels.size)
                                    #     print("numImages from features: ", numImages)


                                    #     sys.exit()

                                    
                                    
                                    context = modelParams[model][event]['context']

                                    contextLabels = np.empty((0,1))
                                    start = 0
                                    threshold = modelParams[model][event]['label_group_threshold'] 

                                    for start in range(numImages):
                                        stop = start+context
                                        if np.mean(labels[start:stop,:]) > threshold:
                                            contextLabels = np.vstack((contextLabels, np.array([1]) ))
                                        else:
                                            contextLabels = np.vstack((contextLabels, np.array([0]) ))

                                    #Now, if the model is a cnn, group the labels according to some parameter:
                                    

                                    

                                    if(contextLabels.shape[0] != numImages):
                                        print("Shapes dont match for file: ", audioFile)

                                        sys.exit()
                                    else:

                                        print("shapes match!")

                                        print("numImages from features: ", numImages)
                                        print("contextLabels.shape: ", contextLabels.shape)
                                        #sys.exit()
                                        


                                        labelfilename = audioFile[:-4] + '.h5'
                                        labelfilepath = os.path.join(labelpath, labelfilename)
                                        
                                        print("saving Label File: ", labelfilename)
                                        
                                        with h5py.File(labelfilepath, 'w') as h5f:
                                            h5f.create_dataset('y', data=contextLabels)
                                            print("saving h5 file {}".format(labelfilename))
                                        




                                    
                                    
    


    if source_data == True:

        #in case of source events, create labels using shape of features for source files!
        WindowLength = featureParams['source_data'][event]['win_length']
        PercentOverlap = featureParams['source_data'][event]['percent_overlap']

        TimeResolution = (100 - PercentOverlap)*WindowLength*0.01



        pathUptoMixture = os.path.join(os.getcwd(),'../data/source_data/events')
        
        print("dumping source event file labels...")
            
        temp = os.path.join(pathUptoMixture,event)
        
        sourcefiles = os.listdir(temp)  
            
        sourcefiles = [x for x in sourcefiles if x[-3:]=='wav']

        labelpath = os.path.join(os.getcwd(), '../Labels',directory, mode,
                                event,'source_labels')

        

        for sourcefile in sourcefiles:
            
            audioFilePath = os.path.join(temp,sourcefile)
            print("reading audio file: {}".format(sourcefile))
            
            if sourcefile in skip[event]:
                continue
            
            with contextlib.closing(wave.open(audioFilePath,'r')) as f1:
                                
                frames = f1.getnframes()
                rate = f1.getframerate()
                duration = frames / float(rate)
                samplesInFile = int( (duration - WindowLength)/TimeResolution)
                
                #labels = np.ones((samplesInFile,1))



                filename = sourcefile[:-4]+'.h5'
                featurePath = os.path.join(os.getcwd(),'../Features',directory,
                              mode, event, 'source_features',filename)

                if model == 'dnn' or model == 'rnn':

                    

                    reloadedFeatureDict = dd.io.load(featurePath)
                    reloadedFeatures = reloadedFeatureDict['feat']    
                    numFrames = reloadedFeatures.shape[0]
                   
                    # print("reloaded features: " ,reloadedFeatures.shape) 
                    # sys.exit()
                    
                    labels = np.ones((numFrames,1))
                    

                    
                    labelfilename = sourcefile[:-4] + '.h5'
                    labelfilepath = os.path.join(labelpath, labelfilename)
                    
                    print("saving Label File: ", labelfilename)
                    
                    with h5py.File(labelfilepath, 'w') as h5f:
                    
                        h5f.create_dataset('y', data=labels)
                        print("saving h5 file {}".format(labelfilename))
                        

                if model == 'cnn':

                    reloadedFeatureDict = dd.io.load(featurePath)
                    reloadedFeatures = reloadedFeatureDict['feat']    
                    numImages = reloadedFeatures.shape[0]
                    
                    
                    labels = np.ones((numImages,1))

                    
                    
                    labelfilename = sourcefile[:-4] + '.h5'
                    labelfilepath = os.path.join(labelpath, labelfilename)
                    
                    print("saving Label File: ", labelfilename)
                    
                    with h5py.File(labelfilepath, 'w') as h5f:
                        h5f.create_dataset('y', data=labels)
                        print("saving h5 file {}".format(labelfilename))
                    


                
                            
        





                          
    return directory






    
    







        
        
        
        
        

                    
