
from __future__ import print_function
import wave
import contextlib
import os
import h5py
import numpy as np
import math
from LABEL_EXTRACTION import *



class CreateLabel(object):
    
    def __init__(self):
        pass
        #print("inititialized")
    
    def CreateAndDump(self, mode = 'TrainLabels',  event = None,  WindowLength = None, PercentOverlap = None):
        
       #mode can be TrainLabels  or TestLabels 
        
        #cwd = 'C:\\Users\\User\\Google Drive\\Summer 2017\\Internship' + \
        #        '\\TUT-rare-sound-events-2017-development\\data'+\
        #        '\\source_data\\events'
                
        #pathUptoMixture = cwd
        
        pathUptoMixture = os.path.join(os.getcwd(),'../data/source_data/events')
        
        print("dumping source event file labels...")
        
        if event == None:
            event = 'babycry'

            
        temp = os.path.join(pathUptoMixture,event)
        
        sourcefiles = os.listdir(temp)  
            
        sourcefiles = [x for x in sourcefiles if x[-3:]=='wav']

        
        if WindowLength is None: 
            WindowLength = 0.04
        
        if PercentOverlap is None:
            PercentOverlap = 50
            
        
            
    
        TimeResolution = (100 - PercentOverlap)*WindowLength*0.01

        #labelsPath = 'C:\\Users\\User\\Google Drive\\Summer 2017\\Internship' + \
        #             '\\TUT-rare-sound-events-2017-development\\Labels\\'+\
        #             mode + '\\' + event 
                     
        labelsPath = os.path.join(os.getcwd(), '../Labels/TrainLabels')
        labelsPath = os.path.join(labelsPath,event)
        
        skip = { 'gunshot' : ['350745.wav','365256.wav','127846.wav'],
                 'glassbreak' : ['233605.wav', '233605 (1).wav'], 
                                 'babycry' :[] }
                
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
            
                
                labels = np.ones((samplesInFile,1))
                sourcefile = sourcefile[:-4]+'.h5'
                labelfilepath = os.path.join(labelsPath,sourcefile)
                h5f = h5py.File(labelfilepath, 'w')
                h5f.create_dataset('y', data=labels)
                print("saving data to h5 file {}".format(sourcefile))
                h5f.close()
                
                            
        return labelsPath
        
        
    def LoadData(self, filepath):
        
        #file path = file upto labels/TrainLabels/babycry
        
        x = np.empty((0,1))
        for labelfile in os.listdir(filepath):
            data = h5py.File(os.path.join(filepath,labelfile), 'r')
            #print("loading data from h5 file: {}".format(labelfile))
            x = np.vstack((x,data['y']))
        return x
            
            
                    


    



obj1 = CreateLabel()
SavedFilePath = obj1.CreateAndDump(event = 'glassbreak') 

obj2 = CreateLabel()
SavedFilePath = obj2.CreateAndDump(event = 'babycry') 

obj3 = CreateLabel()
SavedFilePath = obj3.CreateAndDump(event = 'gunshot') 

#==============================================================================
# labels = obj.LoadData(SavedFilePath)
# 
# print("Extracted Labels.shape {}".format(labels.shape)) 
# print(type(labels))
# print(labels)
# #print (np.sum((labels[np.where(labels == 1)]) ))      
#==============================================================================
        
        
        
        
        
        
        

                     