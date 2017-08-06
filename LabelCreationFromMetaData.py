import wave
import contextlib
import os
import h5py
import numpy as np
import math

class CreateLabel(object):
    
    def __init__(self):
        print("inititialized")
    
    def CreateAndDump(self, WindowLength, PercentOverlap, FileToSave, audioFilesPath, inputPath, ytype):
        
        
        finalLabels = np.empty((0,1))
        TimeResolution = (100 - PercentOverlap)*WindowLength*0.01
        listOfMetaDataFiles = os.listdir(inputPath)
        
        
        for metafile in listOfMetaDataFiles:
            metaFilePath = os.path.join(inputPath, metafile)
            
            print("metaFilePath: {}".format(metaFilePath))
            with open(metaFilePath) as f:
                while True:
                    line = f.readline()
                    print("line read: {}".format(line))
                    if not line:
                        break
                    else:
                        line = line.split('\t')
                        audioFile = line[0]
                        audioFilePath = os.path.join(audioFilesPath, audioFile)
                        print("audioFilePath: {}".format(audioFilePath))
                        with contextlib.closing(wave.open(audioFilePath,'r')) as f1:
                            
                            frames = f1.getnframes()
                            rate = f1.getframerate()
                            duration = frames / float(rate)
                            samplesInFile = int( (duration - WindowLength)/TimeResolution)
                            
                            print("Samples In File: {}".format(samplesInFile))
                            

                            if(len(line) >= 2):
                                eventOnset = float(line[1])
                                eventOffset = float(line[2])
                                
                                labels = np.zeros((samplesInFile, 1))
                                
                                startIndex = math.ceil( (eventOnset - WindowLength)/float(TimeResolution) )
                                print("startIndex: {}".format(startIndex))
                                
                                if(eventOffset > (samplesInFile*TimeResolution + WindowLength) ):
                                    #event offset continues beyond file duration
                                    labels[startIndex:][:] = 1.0
                                else:
                                    stopIndex = int(eventOffset/TimeResolution)
                                    labels[startIndex:stopIndex][:] = 1.0 #check this later
                                    print("stopIndex: {}".format(stopIndex))
                                    
                                  
                                
                                
                            else: #no event
                                labels = np.zeros((samplesInFile,1))
                                print("No event in this file")
                            
                            print('labels.shape : {x}'.format(x = labels.shape) )
                            
                            
                            finalLabels = np.vstack((finalLabels,labels))
                            print('finalLabels.shape (after concatenating) : {x}'.format(x = finalLabels.shape) )
                                
                    
        h5f = h5py.File(FileToSave, 'w')
        h5f.create_dataset(ytype, data=finalLabels)
        print("saving data to h5 file")
        h5f.close()
        return (os.path.join(os.getcwd(), FileToSave))
        
        
    def LoadData(self, filepath,dataToLoad):
        data = h5py.File(filepath, 'r')
        print("loading data from h5 file")
        return np.asarray(data[dataToLoad][:])
            
            
                    


    
print('cc')
audioFilesPath = os.path.join(os.getcwd(), "AudioFileDir")
inputPath =  os.path.join(os.getcwd(), "MetaFileDir")
FileToSave = "testFile.h5"
WindowLength = 0.04
PercentOverlap = 50
ytype = 'yLabelTest'


obj = CreateLabel()
SavedFilePath = obj.CreateAndDump( WindowLength, PercentOverlap, FileToSave, audioFilesPath, inputPath, ytype) 


labels = obj.LoadData(SavedFilePath, ytype)
print("Extracted Labels.shape {}".format(labels.shape))          
            
        
        
        
        
        
        
        

                    