# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 10:30:53 2017

@author: User
"""

from __future__ import print_function

import os
import h5py
import deepdish as dd
import numpy as np
import sys

def getshapes(file):
    reloadedFeatures = dd.io.load(Featurefilepath)
    l = [] 
    x = reloadedFeatures.shape
    for i in x:
        l.append(i)
    return l


def loadKerasData(model, directory, event, numofFiles = float('inf'), mode = 'devtrain', source_data = True):
    
    '''numofFiles = how many samples to load'''
    
    
    ''' Loads normalized data and labels in a form that can be passed to Keras '''
    
         
    
    NormalizedFeaturesPath = os.path.join(os.getcwd(),'../NormalizedFeatures', directory,event, 'mixture_data')
    
    
    ListofFeatureFiles = os.listdir(NormalizedFeaturesPath)
    
    
    LabelPath = os.path.join(os.getcwd(),'../Labels', directory, event, 'mixture_data')
    
 
    
    y = np.empty((0,1))
    i = 0


    #To define x,  get the shape of one file first:
    #Note: All images must have the same x,y,z dimension!

        
    Featurefilepath = os.path.join(NormalizedFeaturesPath,ListofFeatureFiles[0])
    shapeList = getShapes(Featurefilepath) 

    filesList = []
    numFramesList = []


    if len(shapeList) == 2:
        x = np.zeros((0,shapeList[1]))   

    if len(shapeList) == 3:
        x = np.zeros((0,shapeList[1], shapeList[2]))   


    for index,file in enumerate(ListofFeatureFiles):
        
        
        Labelfilepath = os.path.join(LabelPath,file)
        Featurefilepath = os.path.join(NormalizedFeaturesPath,file)
        
        reloadedFeatures = dd.io.load(Featurefilepath)        
        
        data = h5py.File(Labelfilepath, 'r')
        reloadedLabels = data['y']

        filesList.append(file)
        numFramesList.append(reloadedLabels.shape[0])
 
       
        if not( reloadedFeatures.shape[0] == reloadedLabels.shape[0]):
            print (" Features and Labels shape don't match! ")
            print ("For File : ", Featurefilepath)
	    print ("Features shape: ", reloadedFeatures.shape, "Labels shape: ", reloadedLabels.shape)
            print("Skipping File...")
            #sys.exit()
	    continue
            

        
        
        x = np.vstack((x,reloadedFeatures))
        y = np.vstack((y,reloadedLabels))
        i = i+1
        
        if i==numofFiles:
            return x,y
            
        
    
        
    return x,y,filesList,numFramesList



#x, y  = loadKerasData()
#print(x.shape)
#print(y.shape)


