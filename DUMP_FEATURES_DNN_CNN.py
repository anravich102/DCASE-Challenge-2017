from __future__ import print_function
import h5py
import numpy as np
from PARAMS import loadfeatureParams,createParamsFile

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

createParamsFile()
modelParams = loadfeatureParams('modelParams.yaml')

#this is the function for stacking and dumpting CNN faetures to file
def stack_and_dump_data_cnn( directory, event, mode = 'devtrain',
                          source_data = False, mixture_data = True, normalized_data = False):
    
    ''' stacks and dumps ALL  features and labels '''
    #path  = "D:\\GoogleDrive\\Summer 2017\\Internship\\TUT-rare-sound-events-2017-development\\data\\mixture_data\\devtrain\\20b255387a2d0cddc0a3dff5014875e7\\audio"
    
    #cwd = "D:/GoogleDrive/Summer 2017/Internship/TUT-rare-sound-events-2017-development/Test-Code"
    cwd = os.getcwd()
    path = ""

    featureFile = directory+ '_' + event + '_' + mode + '_features.h5'

    labelFile = directory+ '_' + event + '_' +  mode + '_labels.h5'

    if modelParams['mycomp']:
        pathToFeatureFile = os.path.join('../../../../../', featureFile)
        pathToLabelFile = os.path.join('../../../../../', labelFile)
    else:
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

                



        for i,path in enumerate(l):
             
            if normalized_data == True:
                NormalizedFeaturesPath = os.path.join(cwd,'../NormalizedFeatures', directory,
                                                 mode,event, path)
            else:
                NormalizedFeaturesPath = os.path.join(cwd,'../Features', directory,
                                                 mode,event, path)


            ListofFeatureFiles = os.listdir(NormalizedFeaturesPath)


            if path == 'mixture_features':
                LabelPath = os.path.join(cwd,'../Labels', directory,mode, event,
                                             'mixture_labels') 
            elif path == 'source_features':
                LabelPath = os.path.join(cwd,'../Labels', directory,mode, event,
                                                 'source_labels') 

            #should have been named mixture_labels!!  
                

            for index,file in enumerate(ListofFeatureFiles):
                
                Labelfilepath = os.path.join(LabelPath,file)
                Featurefilepath = os.path.join(NormalizedFeaturesPath,file)

               
                    
                f =  h5py.File(Featurefilepath, 'r')
                reloadedFeatures = f['feat'][:]
                x =  reloadedFeatures.shape[0]
                   
                
                

                rows = shapeList[1]
                cols = shapeList[2]
                numImages = shapeList[0]

                
                data =  h5py.File(Labelfilepath, 'r') 
                reloadedLabels = data['y']
                numImages = reloadedLabels.shape[0]
                


                if not( x == numImages):
                    print (" Features and Labels shape don't match! ")
                    print ("For File : ", Featurefilepath)
                    print ("Features shape: ", x, "Labels shape: ", numImages)
                    print("Skipping File...")
                    #sys.exit()
                    continue


                if index==0 and i==0:

                    #print(reloadedFeatures.shape)
                    
                    dset = ffile.create_dataset("features", data = reloadedFeatures[:,np.newaxis,:,:] ,
                                             shape = (numImages,1,rows,cols),
                                                 maxshape=(None,1, rows, cols))

                    #print(reloadedFeatures[:,np.newaxis,:,:].shape)
                    #sys.exit()

                else:
                    dset.resize(dset.shape[0]+numImages,  axis=0)   
                    dset[-numImages:] = reloadedFeatures[:,np.newaxis,:,:]

                    print("Features shape: ",dset.shape)


                if index == 0 and i==0:
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



###eval data test

def stack_and_dump_data_cnn_new( directory, event, mode = 'devtrain',
                          source_data = False, mixture_data = True, normalized_data = False):
    
    ''' stacks and dumps ALL  features and labels '''


    if mode!='eval':
        cwd = os.getcwd()
        path = ""

        featureFile = directory+ '_' + event + '_' + mode + '_features.h5'

        labelFile = directory+ '_' + event + '_' +  mode + '_labels.h5'

        if modelParams['mycomp']:
            pathToFeatureFile = os.path.join('../../../../../', featureFile)
            pathToLabelFile = os.path.join('../../../../../', labelFile)
        else:
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

                    



            for i,path in enumerate(l):
                 
                if normalized_data == True:
                    NormalizedFeaturesPath = os.path.join(cwd,'../NormalizedFeatures', directory,
                                                     mode,event, path)
                else:
                    NormalizedFeaturesPath = os.path.join(cwd,'../Features', directory,
                                                     mode,event, path)


                ListofFeatureFiles = os.listdir(NormalizedFeaturesPath)


                if path == 'mixture_features':
                    LabelPath = os.path.join(cwd,'../Labels', directory,mode, event,
                                                 'mixture_labels') 
                elif path == 'source_features':
                    LabelPath = os.path.join(cwd,'../Labels', directory,mode, event,
                                                     'source_labels') 

                #should have been named mixture_labels!!  
                    

                for index,file in enumerate(ListofFeatureFiles):
                    
                    Labelfilepath = os.path.join(LabelPath,file)
                    Featurefilepath = os.path.join(NormalizedFeaturesPath,file)

                   
                        
                    f =  h5py.File(Featurefilepath, 'r')
                    reloadedFeatures = f['feat'][:]
                    x =  reloadedFeatures.shape[0]
                       
                    
                    

                    rows = shapeList[1]
                    cols = shapeList[2]
                    numImages = shapeList[0]

                    
                    data =  h5py.File(Labelfilepath, 'r') 
                    reloadedLabels = data['y']
                    numImages = reloadedLabels.shape[0]
                    


                    if not( x == numImages):
                        print (" Features and Labels shape don't match! ")
                        print ("For File : ", Featurefilepath)
                        print ("Features shape: ", x, "Labels shape: ", numImages)
                        print("Skipping File...")
                        #sys.exit()
                        continue


                    if index==0:

                        #print(reloadedFeatures.shape)
                        
                        dset = ffile.create_dataset("features", data = reloadedFeatures[:,np.newaxis,:,:] ,
                                                 shape = (numImages,1,rows,cols),
                                                     maxshape=(None,1, rows, cols))

                        #print(reloadedFeatures[:,np.newaxis,:,:].shape)
                        #sys.exit()

                    else:
                        dset.resize(dset.shape[0]+numImages,  axis=0)   
                        dset[-numImages:] = reloadedFeatures[:,np.newaxis,:,:]

                        print("Features shape: ",dset.shape)


                    if index == 0 and i==0:
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




    elif mode=='eval':

        cwd = os.getcwd()
        path = ""

        featureFile = directory+ '_' + event + '_' + mode + '_features.h5'

        
        if modelParams['mycomp']:
            pathToFeatureFile = os.path.join('../../../../../', featureFile)
            
        else:
            pathToFeatureFile = os.path.join('/hdd4', featureFile)
            


        with h5py.File(pathToFeatureFile, "w") as ffile:


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

                    



            for i,path in enumerate(l):
                 
                if normalized_data == True:
                    NormalizedFeaturesPath = os.path.join(cwd,'../NormalizedFeatures', directory,
                                                     mode,event, path)
                else:
                    NormalizedFeaturesPath = os.path.join(cwd,'../Features', directory,
                                                     mode,event, path)


                ListofFeatureFiles = os.listdir(NormalizedFeaturesPath)


    
                for index,file in enumerate(ListofFeatureFiles):
                    
                    
                    Featurefilepath = os.path.join(NormalizedFeaturesPath,file)

                   
                        
                    f =  h5py.File(Featurefilepath, 'r')
                    reloadedFeatures = f['feat'][:]
                    x =  reloadedFeatures.shape[0]
                       
                    
                    

                    rows = shapeList[1]
                    cols = shapeList[2]
                    numImages = shapeList[0]

                


                    if index==0:

                        #print(reloadedFeatures.shape)
                        
                        dset = ffile.create_dataset("features", data = reloadedFeatures[:,np.newaxis,:,:] ,
                                                 shape = (numImages,1,rows,cols),
                                                     maxshape=(None,1, rows, cols))

                        #print(reloadedFeatures[:,np.newaxis,:,:].shape)
                        #sys.exit()

                    else:
                        dset.resize(dset.shape[0]+numImages,  axis=0)   
                        dset[-numImages:] = reloadedFeatures[:,np.newaxis,:,:]

                        print("Features shape: ",dset.shape)


                    f.close() 
                    
         

                  
        return pathToFeatureFile



###eval data test ends


###eval data test begins

def stack_and_dump_data_dnn_new( directory, event, mode = 'devtrain',
                          source_data = False, mixture_data = True, normalized_data = False):
    
    ''' stacks and dumps ALL  features and labels '''


    if mode!='eval':
        cwd = os.getcwd()
        path = ""

        featureFile = directory+ '_' + event + '_' + mode + '_features.h5'

        labelFile = directory+ '_' + event + '_' +  mode + '_labels.h5'

        if modelParams['mycomp']:
            pathToFeatureFile = os.path.join('../../../../../', featureFile)
            pathToLabelFile = os.path.join('../../../../../', labelFile)
        else:
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
                    

                             
                    f =  h5py.File(Featurefilepath, 'r')
                    reloadedFeatures = f['feat']
                    x =  reloadedFeatures.shape[0]

                    
                    num_features = shapeList[1]
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
                                                 shape = (numImages,num_features),
                                                     maxshape=(None, num_features))



                    else:
                        dset.resize(dset.shape[0]+numImages,  axis=0)   
                        dset[-numImages:] = reloadedFeatures

                        print("Features shape: ",dset.shape)


                    if index == 0 and i ==0:
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
                
                print(mode, "- numofSamples: ", sum(numFramesList))    
                   

                      
        return pathToFeatureFile,pathToLabelFile


    elif mode=='eval':

        cwd = os.getcwd()
        path = ""

        featureFile = directory+ '_' + event + '_' + mode + '_features.h5'

       

        if modelParams['mycomp']:
            pathToFeatureFile = os.path.join('../../../../../', featureFile)
           
        else:
            pathToFeatureFile = os.path.join('/hdd4', featureFile)
           

        
        with h5py.File(pathToFeatureFile, "w") as ffile:


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


                
                #should have been named mixture_labels!!  
                    

                for index,file in enumerate(ListofFeatureFiles):
                    

                    Featurefilepath = os.path.join(NormalizedFeaturesPath,file)
                    

                             
                    f =  h5py.File(Featurefilepath, 'r')
                    reloadedFeatures = f['feat']
                    x =  reloadedFeatures.shape[0]

                    
                    num_features = shapeList[1]
                    numImages = shapeList[0]


                    

                

                    if index==0:


                        dset = ffile.create_dataset("features", data = reloadedFeatures ,
                                                 shape = (numImages,num_features),
                                                     maxshape=(None, num_features))



                    else:
                        dset.resize(dset.shape[0]+numImages,  axis=0)   
                        dset[-numImages:] = reloadedFeatures

                        print("Features shape: ",dset.shape)


                

                    f.close() 

                    filesList.append(file)
                    numFramesList.append(numImages)
                
                print(mode, "- numofSamples: ", sum(numFramesList))    
                   

                      
        return pathToFeatureFile


###eval data test ends

















#this is the function for stacking and dumpting DNN faetures to file
def stack_and_dump_data_dnn( directory, event, mode = 'devtrain',
                          source_data = False, mixture_data = True, normalized_data = False):
    
    ''' stacks and dumps ALL  features and labels '''

    cwd = os.getcwd()
    path = ""

    featureFile = directory+ '_' + event + '_' + mode + '_features.h5'

    labelFile = directory+ '_' + event + '_' +  mode + '_labels.h5'

    if modelParams['mycomp']:
        pathToFeatureFile = os.path.join('../../../../../', featureFile)
        pathToLabelFile = os.path.join('../../../../../', labelFile)
    else:
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
                

                         
                f =  h5py.File(Featurefilepath, 'r')
                reloadedFeatures = f['feat']
                x =  reloadedFeatures.shape[0]

                
                num_features = shapeList[1]
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


                if index==0 and i ==0:


                    dset = ffile.create_dataset("features", data = reloadedFeatures ,
                                             shape = (numImages,num_features),
                                                 maxshape=(None, num_features))



                else:
                    dset.resize(dset.shape[0]+numImages,  axis=0)   
                    dset[-numImages:] = reloadedFeatures

                    print("Features shape: ",dset.shape)


                if index == 0 and i==0:
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
            
            print(mode, "- numofSamples: ", sum(numFramesList))    
               

                  
    return pathToFeatureFile,pathToLabelFile


def stack_and_dump_data_rnn( directory, event, mode = 'devtrain',
                          source_data = False, mixture_data = True, normalized_data = False):
    
    ''' stacks and dumps ALL  features and labels '''

    cwd = os.getcwd()
    path = ""

    featureFile = directory+ '_' + event + '_' + mode + '_features.h5'

    labelFile = directory+ '_' + event + '_' +  mode + '_labels.h5'

    if modelParams['mycomp']:
        pathToFeatureFile = os.path.join('../../../../../', featureFile)
        pathToLabelFile = os.path.join('../../../../../', labelFile)
    else:
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
                

                         
                f =  h5py.File(Featurefilepath, 'r')
                reloadedFeatures = f['feat'][:]
                reloadedFeatures = reloadedFeatures[np.newaxis,:,:]
                x =  reloadedFeatures.shape[1]


                print(reloadedFeatures.shape)
                #sys.exit()
                
                num_features = reloadedFeatures.shape[2]
                numImages = reloadedFeatures.shape[1]

                
                data =  h5py.File(Labelfilepath, 'r') 
                reloadedLabels = data['y'][:]
                reloadedLabels = reloadedLabels[np.newaxis,:,:]
                temp = reloadedLabels.shape[1]
                
                #print(file)
                #print(reloadedLabels[30:120][:] )
                #sys.exit()

                if not( x == temp):
                    print (" Features and Labels shape don't match! ")
                    print ("For File : ", Featurefilepath)
                    print ("Features shape: ", x, "Labels shape: ", numImages)
                    print("Skipping File...")
                    #sys.exit()
                    continue


                if index==0:


                    dset = ffile.create_dataset("features", data = reloadedFeatures ,
                                             shape = (1,numImages,num_features),
                                                 maxshape=(None, numImages, num_features))



                else:
                    dset.resize(dset.shape[0]+1,  axis=0)   
                    dset[-1:] = reloadedFeatures

                    print("Features shape: ",dset.shape)


                if index == 0:
                    dsetl = lfile.create_dataset("labels", data = reloadedLabels , shape = 
                                                (1,numImages,1),
                                                maxshape=(None, numImages, 1))
                else:
                    dsetl.resize(dsetl.shape[0]+1,  axis=0)   
                    dsetl[-1:] = reloadedLabels

                    print("Labels shape: ",dsetl.shape)

                f.close() 
                data.close()

                filesList.append(file)
                numFramesList.append(numImages)
            
            print(mode, "- numofSamples: ", sum(numFramesList))    
               

                  
    return pathToFeatureFile,pathToLabelFile


#stack_and_dump_data(  'cnn_07_01_PM_July_17_2017', 'babycry', mode = 'devtest',
#                           source_data = False, mixture_data = True, normalized_data = False)  

# stack_and_dump_data(  'cnn_07_01_PM_July_17_2017', 'babycry', mode = 'devtrain',
#                           source_data = False, mixture_data = True, normalized_data = False)             
                   
                    
                    

def stack_and_dump_data_rnn_new( directory, event, mode = 'devtrain',
                          source_data = False, mixture_data = True, normalized_data = False):
    
    ''' stacks and dumps ALL  features and labels '''
    if mode!='eval':
        cwd = os.getcwd()
        path = ""

        featureFile = directory+ '_' + event + '_' + mode + '_features.h5'

        labelFile = directory+ '_' + event + '_' +  mode + '_labels.h5'

        if modelParams['mycomp']:
            pathToFeatureFile = os.path.join('../../../../../', featureFile)
            pathToLabelFile = os.path.join('../../../../../', labelFile)
        else:
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
                    

                             
                    f =  h5py.File(Featurefilepath, 'r')
                    reloadedFeatures = f['feat'][:]
                    reloadedFeatures = reloadedFeatures[np.newaxis,:,:]
                    x =  reloadedFeatures.shape[1]


                    print(reloadedFeatures.shape)
                    #sys.exit()
                    
                    num_features = reloadedFeatures.shape[2]
                    numImages = reloadedFeatures.shape[1]

                    
                    data =  h5py.File(Labelfilepath, 'r') 
                    reloadedLabels = data['y'][:]
                    reloadedLabels = reloadedLabels[np.newaxis,:,:]
                    temp = reloadedLabels.shape[1]
                    
                    #print(file)
                    #print(reloadedLabels[30:120][:] )
                    #sys.exit()

                    if not( x == temp):
                        print (" Features and Labels shape don't match! ")
                        print ("For File : ", Featurefilepath)
                        print ("Features shape: ", x, "Labels shape: ", numImages)
                        print("Skipping File...")
                        #sys.exit()
                        continue


                    if index==0:


                        dset = ffile.create_dataset("features", data = reloadedFeatures ,
                                                 shape = (1,numImages,num_features),
                                                     maxshape=(None, numImages, num_features))



                    else:
                        dset.resize(dset.shape[0]+1,  axis=0)   
                        dset[-1:] = reloadedFeatures

                        print("Features shape: ",dset.shape)


                    if index == 0:
                        dsetl = lfile.create_dataset("labels", data = reloadedLabels , shape = 
                                                    (1,numImages,1),
                                                    maxshape=(None, numImages, 1))
                    else:
                        dsetl.resize(dsetl.shape[0]+1,  axis=0)   
                        dsetl[-1:] = reloadedLabels

                        print("Labels shape: ",dsetl.shape)

                    f.close() 
                    data.close()

                    filesList.append(file)
                    numFramesList.append(numImages)
                
                print(mode, "- numofSamples: ", sum(numFramesList))    
                   

                      
        return pathToFeatureFile,pathToLabelFile       

    elif mode=='eval':
        cwd = os.getcwd()
        path = ""

        featureFile = directory+ '_' + event + '_' + mode + '_features.h5'

       
        if modelParams['mycomp']:
            pathToFeatureFile = os.path.join('../../../../../', featureFile)
           
        else:
            pathToFeatureFile = os.path.join('/hdd4', featureFile)


        
        with h5py.File(pathToFeatureFile, "w") as ffile:


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


               

                for index,file in enumerate(ListofFeatureFiles):
                    
                   
                    Featurefilepath = os.path.join(NormalizedFeaturesPath,file)
                    

                             
                    f =  h5py.File(Featurefilepath, 'r')
                    reloadedFeatures = f['feat'][:]
                    reloadedFeatures = reloadedFeatures[np.newaxis,:,:]
                    x =  reloadedFeatures.shape[1]


                    print(reloadedFeatures.shape)
                    #sys.exit()
                    
                    num_features = reloadedFeatures.shape[2]
                    numImages = reloadedFeatures.shape[1]

                    
                    



                    if index==0:


                        dset = ffile.create_dataset("features", data = reloadedFeatures ,
                                                 shape = (1,numImages,num_features),
                                                     maxshape=(None, numImages, num_features))



                    else:
                        dset.resize(dset.shape[0]+1,  axis=0)   
                        dset[-1:] = reloadedFeatures

                        print("Features shape: ",dset.shape)


                   

                    f.close() 


                    filesList.append(file)
                    numFramesList.append(numImages)
                
                print(mode, "- numofSamples: ", sum(numFramesList))    
                   

                      
        return pathToFeatureFile    


                    
                    
