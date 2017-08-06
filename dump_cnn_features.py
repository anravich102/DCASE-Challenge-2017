from __future__ import print_function
import h5py
import numpy as np


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

def stack_and_dump_data( directory, event, mode = 'devtrain',
                          source_data = False, mixture_data = True, normalized_data = False):
    
    ''' stacks and dumps ALL  features and labels '''
    #path  = "D:\\GoogleDrive\\Summer 2017\\Internship\\TUT-rare-sound-events-2017-development\\data\\mixture_data\\devtrain\\20b255387a2d0cddc0a3dff5014875e7\\audio"
    
    #cwd = "D:/GoogleDrive/Summer 2017/Internship/TUT-rare-sound-events-2017-development/Test-Code"
    cwd = os.getcwd()
    path = ""

    featureFile = directory+ '_' + event + '_' + mode + '_features.h5'

    labelFile = directory+ '_' + event + '_' +  mode + '_labels.h5'

    pathToFeatureFile = os.path.join('/hdd4', featureFile)
    pathToLabelFile = os.path.join('/hdd4', labelFile)
    
    with h5py.File(pathToFeatureFile, "w") as ffile, h5py.File(pathToLabelFile,"w") as lfile:


        l = []
        c = ''
        if mixture_data==True and source_data==True:
            l.append('mixture_features')
            l.append('source_features')
        elif mixture_data==True and source_data==False:
            l.append('mixture_features')
        elif mixture_data==False and source_data==True:
            l.append('source_features')

        if normalized_data == True:
            NormalizedFeaturesPath = os.path.join(cwd,'../NormalizedFeatures', directory,
                                              mode,event, 'mixture_features')
        else:
            NormalizedFeaturesPath = os.path.join(cwd,'../Features', directory,
                                              mode,event, 'mixture_features')


        ListofFeatureFiles = os.listdir(NormalizedFeaturesPath)

        Featurefilepath = os.path.join(NormalizedFeaturesPath,ListofFeatureFiles[0])
        shapeList = getshapes(Featurefilepath) 

        filesList = []
        numFramesList = []

                
        if normalized_data == True:
            NormalizedFeaturesPath = os.path.join(cwd,'../NormalizedFeatures', directory,
                                             mode,event, 'mixture_features')
        else:
            NormalizedFeaturesPath = os.path.join(cwd,'../Features', directory,
                                             mode,event, 'mixture_features')

        ListofFeatureFiles = os.listdir(NormalizedFeaturesPath)

        Featurefilepath = os.path.join(NormalizedFeaturesPath,ListofFeatureFiles[0])
        shapeList = getshapes(Featurefilepath) 



        for i,path in enumerate(l):
             
            if normalized_data == True:
                NormalizedFeaturesPath = os.path.join(cwd,'../NormalizedFeatures', directory,
                                                 mode,event, path)
            else:
                NormalizedFeaturesPath = os.path.join(cwd,'../Features', directory,
                                                 mode,event, path)


            ListofFeatureFiles = os.listdir(NormalizedFeaturesPath)


            if path == 'mixture_features':
                LabelPath = os.path.join(cwd,'../Labels', directory,mode, event, 'mixture_labels') 
            elif path == 'source_features':
                LabelPath = os.path.join(cwd,'../Labels', directory,mode, event, 'source_labels') 

            #should have been named mixture_labels!!  
                

            for index,file in enumerate(ListofFeatureFiles):
                
                Labelfilepath = os.path.join(LabelPath,file)
                Featurefilepath = os.path.join(NormalizedFeaturesPath,file)
                x = 0
                y = 0
                reloadedFeatures = ''
                if normalized_data == False:
                    
                    f =  h5py.File(Featurefilepath, 'r')
                    reloadedFeatures = f['feat']
                    x =  reloadedFeatures.shape[0]
                       
                else:

                    reloadedFeatures = dd.io.load(Featurefilepath)
                         
                

                rows = shapeList[1]
                cols = shapeList[2]
                numImages = shapeList[0]

                
                data =  h5py.File(Labelfilepath, 'r') 
                reloadedLabels = data['y']
                numImages = reloadedLabels.shape[0]
                
                #print(file)
                #print(reloadedLabels[30:120][:] )
                #sys.exit()

                if not( x == numImages):
                    print (" Features and Labels shape don't match! ")
                    print ("For File : ", Featurefilepath)
                    print ("Features shape: ", x, "Labels shape: ", numImages)
                    print("Skipping File...")
                    #sys.exit()
                    continue


                if index==0:


                    dset = ffile.create_dataset("features", data = reloadedFeatures ,
                                             shape = (numImages,rows,cols),
                                                 maxshape=(None, rows, cols))



                else:
                    dset.resize(dset.shape[0]+numImages,  axis=0)   
                    dset[-numImages:] = reloadedFeatures

                    print("Features shape: ",dset.shape)


                if index == 0:
                    dsetl = lfile.create_dataset("labels", data = reloadedLabels , shape = 
                                                (numImages,1),
                                                maxshape=(None, 1))
                else:
                    dsetl.resize(dsetl.shape[0]+numImages,  axis=0)   
                    dsetl[-numImages:] = reloadedLabels

                    print("Labels shape: ",dsetl.shape)

                f.close() 
                data.close()

                filesList.append(file)
                numFramesList.append(numImages)
         
               

                  
    return pathToFeatureFile,pathToLabelFile




#stack_and_dump_data(  'cnn_07_01_PM_July_17_2017', 'babycry', mode = 'devtest',
#                           source_data = False, mixture_data = True, normalized_data = False)  

# stack_and_dump_data(  'cnn_07_01_PM_July_17_2017', 'babycry', mode = 'devtrain',
#                           source_data = False, mixture_data = True, normalized_data = False)             
                   
                    
                    

                    


                    
                    
