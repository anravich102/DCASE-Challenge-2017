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
from PARAMS import *



def do_label_extraction(pathFromFeature, model,event, source_data = False, mode = 'devtrain'):
    '''do label extraction for a specific event and model'''

    #First make the required directories: 

    directory = pathFromFeature #model+timestamp string
    dirPath = os.path.join(os.getcwd(), '../Labels',directory)
    

    if not os.path.exists(dirPath):
        os.mkdir(dirPath)


    modePath = os.path.join(dirPath,mode)

    os.mkdir(modePath)

    eventdir = os.path.join(modePath,event)
    os.mkdir(eventdir)

    sourceFeaturesDir = os.path.join(eventdir, 'source_features')
    mixtureFeaturesDir = os.path.join(eventdir, 'mixture_features')
    os.mkdir(sourceFeaturesDir)
    os.mkdir(mixtureFeaturesDir)

    print("mixtureFeaturesDir", mixtureFeaturesDir)

    #load feature param YAML file:
    filename = 'FeatureParams.yaml' #present in the CWD
    featureParams = loadfeatureParams(filename)

    filename = 'modelParams.yaml'
    modelParams = loadfeatureParams(filename)

    #now extract labels

    labelPath = ExtractAndDumpLabels( model, directory, featureParams, modelParams,
                event, source_data = source_data, mode = mode )


    return labelPath


def ExtractAndDumpLabels(model, directory, featureParams, modelParams,
                        event, source_data = False , mode = 'devtrain' ):


    if  mode is 'devtrain':
        
        print("Dumping mixture data (training) labels ....")
    else:
       
        print("Dumping mixture data (testing) labels ....")


    pathUptoMixture = os.path.join(os.getcwd(),'../data/mixture_data',mode)
        
    x = os.listdir(pathUptoMixture) 
        
    paramHashes = []
    for i in x:
        str1 = 'consider '+ str(i) +'?'
        userinput = input(str1)
        if userinput == 'y':
            paramHashes.append(i)
            
        
            
    labelpath = os.path.join(os.getcwd(), '../Labels',directory, mode,
                            event,'mixture_features')

    #print("labelPath:",labelpath)

    
    #os.mkdir(labelpath)
    
    
    WindowLength = featureParams[model][event]['win_length']
    PercentOverlap = featureParams[model][event]['percent_overlap']


                
    TimeResolution = (100 - PercentOverlap)*WindowLength*0.01
    
            
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
            
            
            #print("metaFilePath: {}".format(metaFilePath))
                
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
                                    labels[startIndex:][:] = 1.0
                                else:
                                    stopIndex = int(eventOffset/TimeResolution)
                                    labels[startIndex:stopIndex][:] = 1.0 #check this later
                                    #print("stopIndex: {}".format(stopIndex))
                                    
                                  
                                
                                
                            else: #no event
                                labels = np.zeros((samplesInFile,1))
                                
                            
                            #print('labels.shape for this file: {x}'.format(x = labels.shape) )

                            checkfile = 'LabelsCheck.txt'
                            if os.path.exists(checkfile):
                                append_write = 'a' # append if already exists
                            else:
                                append_write = 'w' 
                            with open(checkfile,append_write) as f:
								
                                f.write(audioFile + "\t"+ "num of ones: " + str(np.sum(labels)) + "\n")

                            # First, if the model is a dnn or cnn,
                            # check if label size matches feature size, else correct it:

                            #first get the size of the features of the corresponding file:
                            filename = audioFile[:-4]+'.h5'
                            featurePath = os.path.join(os.getcwd(),'../Features',directory,
                                          mode, event, 'mixture_features',filename)

                            if model == 'dnn':

                                

                                reloadedFeatureDict = dd.io.load(featurePath)
                                reloadedFeatures = reloadedFeatureDict['feat']    
                                numFrames = reloadedFeatures.shape[0]
                                
                                if samplesInFile == (numFrames - 1):
                                    labels = labels[:-1][:]




                            if model == 'cnn':
                                reloadedFeatureDict = dd.io.load(featurePath)
                                reloadedFeatures = reloadedFeatureDict['feat']    
                                numImages = reloadedFeatures.shape[0]
                                step = featureParams[model][event]['step']
                                numFrames = int(numImages*reloadedFeatures.shape[2])


                                # TO DO: exception handling for source event data!


                                #Now, if the model is a cnn, group the labels according to some parameter:
                                threshold = modelParams[model][event]['label_group_threshold']
                                splitLabelList = np.split(labels,step) #ASSUMING PERFECT DIVISION!
                                imageLabels = np.empty((0,1))
                                for image in splitLabelList:
                                    if np.mean(image) >= threshold:
                                        imageLabels = np.vstack((imageLabels, np.array([1])))
                                    else:
                                        imageLabels = np.vstack((imageLabels,  np.array([0]))) 


                                
                            labelfilename = audioFile[:-4] + '.h5'
                            labelfilepath = os.path.join(labelpath, labelfilename)
                            
                            #print("labelfilepath: ",labelfilepath)
                            
                            #finalLabels = np.vstack((finalLabels,labels))
                            #print('finalLabels.shape for this file : {x}'.format(x = finalLabels.shape) )
                            print("saving Label File: ", labelfilename)
                
                            h5f = h5py.File(labelfilepath, 'w')
                            h5f.create_dataset('y', data=labels)
                            print("saving h5 file {}".format(labelfilename))
                            h5f.close()




                                    
                                    #TO DO: write code for source data with exception handling!
                                
                          
        return directory






    
    







        
        
        
        
        

                    
