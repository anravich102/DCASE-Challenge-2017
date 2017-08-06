# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 17:39:04 2017

@author: User
"""





#currPath = "C:\\Users\\User\\Google Drive\\Summer 2017\\Internship\\TUT-rare-sound-events-2017-development"
   
import os


             
def makeDirectories(path):
    
    #Makes normalized features directories
    
    currPath = path
    NormalizedFeatures = os.path.join(currPath, 'NormalizedFeatures')              
    os.mkdir(NormalizedFeatures)
    
    for mode in ['devtest','devtrain']:
        modePath = os.path.join(NormalizedFeatures,mode)
        os.mkdir(modePath)
        
        for event in ['babycry','glassbreak','gunshot']:
            eventPath =  os.path.join(modePath,event)
            os.mkdir(eventPath)
            
            for features in ['mfcc','spectrogram']:
                featureTypePath = os.path.join(eventPath,features)
                os.mkdir(featureTypePath)
            
                
makeDirectories(os.path.join(os.getcwd(), '..'))
























            
        
    
    
