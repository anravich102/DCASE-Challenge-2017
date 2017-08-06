import wave
import contextlib
import os
import h5py
import numpy as np
import pickle
import sys
import csv
import math

class CreateLabel(object):
    
    def __init__(self):
        pass
    
    def CreateAndDump(self, WindowLength, PercentOverlap, FileToSave, audioFilesPath, inputPath, ytype):
        
        
        finalLabels = np.empty((0,0))
        TimeResolution = (100-PercentOverlap)*WindowLength
        listOfMetaDataFiles = os.listdir(inputPath)
        
        
        for metafile in listOfMetaDataFiles:
            metaFilePath = os.path.join(inputPath, metafile)
            with open(metaFilePath) as f:
                while True:
                    line = f.readline()
                    print(line)
                    if not line:
                        break
                    else:
                        line = line.split('\t')
                        audioFile = line[0]
                        audioFilePath = os.path.join(audioFilesPath, audioFile)
                        with contextlib.closing(wave.open(audioFilePath,'r')) as f:
                            
                            frames = f.getnframes()
                            rate = f.getframerate()
                            duration = frames / float(rate)
                            samplesInFile = int( (duration - WindowLength)/TimeResolution)
                            

                            if(len(line) >= 2):
                                eventOnset = float(line[1])
                                eventOffset = float(line[2])
                                
                                labels = np.zeros((samplesInFile, 1))
                                
                                startIndex = math.ceil( (eventOnset - WindowLength)/float(TimeResolution) )
                                
                                if(eventOffset > (samplesInFile*TimeResolution + WindowLength) ):
                                    #event offset continues beyond file duration
                                    labels[startIndex:][:] = 1.0
                                else:
                                    stopIndex = int(eventOffset/TimeResolution)
                                    labels[startIndex:stopIndex][:] = 1.0 #check this later
                                    
                                  
                                
                                
                            else: #no event
                                labels = np.zeros((samplesInFile,1))
                                
                            finalLabels = np.vstack((finalLabels,labels))
                                
                    
        h5f = h5py.File(FileToSave, 'w')
        h5f.create_dataset(ytype, data=finalLabels)
        print("saving data to h5 file")
        h5f.close()
        
        
        def LoadData(self, filepath,dataToLoad):
            data = h5py.File(filepath, 'r')
            print("loading data from h5 file")
            return np.asarray(data[dataToLoad][:])
            
            
                    

def main():
    
    print('cc')
    audioFilesPath = os.path.join(os.getcwd(), "AudioFileDir")
    inputPath =  os.path.join(os.getcwd(), "MetaFileDir")
    FileToSave = "testFile.h5"
    WindowLength = 0.04
    PercentOverlap = 50
    ytype = 'yLabelTest'
    
    print('test')
    

    obj = CreateLabel()
    obj.CreateAndDump( WindowLength, PercentOverlap, FileToSave, audioFilesPath, inputPath, ytype)           
            
        
        
        
        
        
        
        

                    