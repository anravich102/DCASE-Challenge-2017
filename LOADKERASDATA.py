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
    reloadedFeatures = dd.io.load(file)
    l = [] 
    if isinstance(reloadedFeatures,dict):
        x = reloadedFeatures['feat'].shape  
    else:
        x = reloadedFeatures.shape    
    for i in x:
        l.append(i)
    return l


def loadKerasData(model, directory, event, numofFiles = float('inf'), mode = 'devtrain', source_data = False,
                    mixture_data = True, normalized_data = True):
    
    '''numofFiles = how many samples to load'''

    ''' normalized_data = True means normalized data loaded. else unnormalized data is loaded '''
    
    '''mode can be devtrain or devtest or eval'''
    ''' Loads normalized data and labels in a form that can be passed to Keras '''
    l = []
    c = ''
    if mixture_data==True and source_data==True:
        l.append('mixture_features')
        l.append('source_features')
    elif mixture_data==True and source_data==False:
        l.append('mixture_features')
    elif mixture_data==False and source_data==True:
        l.append('source_features')


    y = np.empty((0,1))

    i = 0
    #To define x,  get the shape of one file first:
        #Note: All images must have the same x,y dimension!

    if normalized_data == True:
        NormalizedFeaturesPath = os.path.join(os.getcwd(),'../NormalizedFeatures', directory,
                                          mode,event, 'mixture_features')
    else:
        NormalizedFeaturesPath = os.path.join(os.getcwd(),'../Features', directory,
                                          mode,event, 'mixture_features')
    
    ListofFeatureFiles = os.listdir(NormalizedFeaturesPath)

    Featurefilepath = os.path.join(NormalizedFeaturesPath,ListofFeatureFiles[0])
    shapeList = getshapes(Featurefilepath) 


    filesList = []
    numFramesList = []


    if len(shapeList) == 2:
        x = np.zeros((0,shapeList[1]))   

    if len(shapeList) == 3:
        x = np.zeros((0,shapeList[1], shapeList[2]))





    for index,path in enumerate(l) :
         
        if normalized_data == True:
            NormalizedFeaturesPath = os.path.join(os.getcwd(),'../NormalizedFeatures', directory,
                                             mode,event, path)
        else:
            NormalizedFeaturesPath = os.path.join(os.getcwd(),'../Features', directory,
                                             mode,event, path)


        ListofFeatureFiles = os.listdir(NormalizedFeaturesPath)


        if path == 'mixture_features':
            LabelPath = os.path.join(os.getcwd(),'../Labels', directory,mode, event, 'mixture_labels') 
        elif path == 'source_features':
            LabelPath = os.path.join(os.getcwd(),'../Labels', directory,mode, event, 'source_labels') 

        #should have been named mixture_labels!!  
            

        for index,file in enumerate(ListofFeatureFiles):
            
            
            Labelfilepath = os.path.join(LabelPath,file)
            Featurefilepath = os.path.join(NormalizedFeaturesPath,file)

            if normalized_data == False:
                reloadedFeatures = dd.io.load(Featurefilepath)
                reloadedFeatures = reloadedFeatures['feat']     
            else:
                reloadedFeatures = dd.io.load(Featurefilepath)
                     


            if mode != 'eval':
                try:
                    data = h5py.File(Labelfilepath, 'r')
                    reloadedLabels = data['y']
                except OSError as e:
                    continue
                

                filesList.append(file)
                numFramesList.append(reloadedLabels.shape[0])
         
               
                if not( reloadedFeatures.shape[0] == reloadedLabels.shape[0]):
                    print (" Features and Labels shape don't match! ")
                    print ("For File : ", Featurefilepath)
                    print ("Features shape: ", reloadedFeatures.shape, "Labels shape: ", reloadedLabels.shape)
                    print("Skipping File...")
                    #sys.exit()
                    continue
                    

                y = np.vstack((y,reloadedLabels))

            
            x = np.vstack((x,reloadedFeatures))
            
            i = i+1

            


            
            if mode != 'eval':
                if i==numofFiles:
                    if model == 'dnn':
                        return x,y,filesList,numFramesList
                    elif model == 'cnn':
                       

                        return x[:,np.newaxis,:,:],y,filesList,numFramesList

            else:
                if i==numofFiles:
                    if model == 'dnn':
                        return x,filesList,numFramesList
                    elif model == 'cnn':
                        return x[:,np.newaxis,:,:],filesList,numFramesList

        
    
    if mode != 'eval':   
       if model == 'dnn':
            return x,y,filesList,numFramesList
       elif model == 'cnn':
            return x[:,np.newaxis,:,:],y,filesList,numFramesList

    else:
         if model == 'dnn':
            return x,filesList,numFramesList
         elif model == 'cnn':
            return x[:,np.newaxis,:,:],filesList,numFramesList




#x, y  = loadKerasData()
#print(x.shape)
#print(y.shape)

# pathTofeatureFile, feature_dataset, pathTolabelfile, label_dataset, filesList, numFrameslist
#  = stack_and_dump_data(model, directry, event, mode = 'devtrain',
#                           source_daota = False, mixture_data = True, normalized_data = True)







def get_files_list(model, directory, event, numofFiles = float('inf'), mode = 'devtrain', 
                    source_data = False,
                    mixture_data = True, normalized_data = True):
    
    if mode != 'eval':
        l = []
        c = ''
        if mixture_data==True and source_data==True:
            l.append('mixture_features')
            l.append('source_features')
        elif mixture_data==True and source_data==False:
            l.append('mixture_features')
        elif mixture_data==False and source_data==True:
            l.append('source_features')


        

        i = 0
        #To define x,  get the shape of one file first:
            #Note: All images must have the same x,y dimension!

        if normalized_data == True:
            NormalizedFeaturesPath = os.path.join(os.getcwd(),'../NormalizedFeatures', directory,
                                              mode,event, 'mixture_features')
        else:
            NormalizedFeaturesPath = os.path.join(os.getcwd(),'../Features', directory,
                                              mode,event, 'mixture_features')
        
        ListofFeatureFiles = os.listdir(NormalizedFeaturesPath)




        filesList = []
        numFramesList = []


        




        for index,path in enumerate(l) :
             
            if normalized_data == True:
                NormalizedFeaturesPath = os.path.join(os.getcwd(),'../NormalizedFeatures', directory,
                                                 mode,event, path)
            else:
                NormalizedFeaturesPath = os.path.join(os.getcwd(),'../Features', directory,
                                                 mode,event, path)


            ListofFeatureFiles = os.listdir(NormalizedFeaturesPath)


            if path == 'mixture_features':
                LabelPath = os.path.join(os.getcwd(),'../Labels', directory,mode, event, 'mixture_labels') 
            elif path == 'source_features':
                LabelPath = os.path.join(os.getcwd(),'../Labels', directory,mode, event, 'source_labels') 

            #should have been named mixture_labels!!  
                

            for index,file in enumerate(ListofFeatureFiles):

                Labelfilepath = os.path.join(LabelPath,file)
                

                data = h5py.File(Labelfilepath, 'r')
                reloadedLabels = data['y']

                
                filesList.append(file)
                numFramesList.append(reloadedLabels.shape[0])
                data.close()
                   
                    
        return filesList, numFramesList
            
            
    elif mode == 'eval':
        l = []
        c = ''
        if mixture_data==True and source_data==True:
            l.append('mixture_features')
            l.append('source_features')
        elif mixture_data==True and source_data==False:
            l.append('mixture_features')
        elif mixture_data==False and source_data==True:
            l.append('source_features')


        

        i = 0
        #To define x,  get the shape of one file first:
            #Note: All images must have the same x,y dimension!

        if normalized_data == True:
            NormalizedFeaturesPath = os.path.join(os.getcwd(),'../NormalizedFeatures', directory,
                                              mode,event, 'mixture_features')
        else:
            NormalizedFeaturesPath = os.path.join(os.getcwd(),'../Features', directory,
                                              mode,event, 'mixture_features')
        
        ListofFeatureFiles = os.listdir(NormalizedFeaturesPath)




        filesList = []
        numFramesList = []


        




        for index,path in enumerate(l) :
             
            if normalized_data == True:
                NormalizedFeaturesPath = os.path.join(os.getcwd(),'../NormalizedFeatures', directory,
                                                 mode,event, path)
            else:
                NormalizedFeaturesPath = os.path.join(os.getcwd(),'../Features', directory,
                                                 mode,event, path)


            ListofFeatureFiles = os.listdir(NormalizedFeaturesPath)


            if path == 'mixture_features':
                LabelPath = os.path.join(os.getcwd(),'../Features', directory,mode, event, 'mixture_features') 
            elif path == 'source_features':
                LabelPath = os.path.join(os.getcwd(),'../Features', directory,mode, event, 'source_features') 

            #should have been named mixture_labels!!  
                

            for index,file in enumerate(ListofFeatureFiles):

                Labelfilepath = os.path.join(LabelPath,file)
                

                data = h5py.File(Labelfilepath, 'r')
                reloadedLabels = data['feat']

                
                filesList.append(file)
                numFramesList.append(reloadedLabels.shape[0])
                data.close()
                   
                    
        return filesList, numFramesList

            
            



