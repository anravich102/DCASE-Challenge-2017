from __future__ import print_function
from LOADUTIL import *
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import deepdish as dd
import os   
from builtins import input
from PARAMS import *

import sys

import numpy as np
import deepdish as dd
import os



def detect_all_zero_col(feature_matrix):
  col_sum = np.sum(np.absolute(feature_matrix), axis=0)
  
  #print(col_sum)
  
  all_zero_cols = np.where(col_sum == 0)[0]


  if len(all_zero_cols) > 0:
      print ('%d all zero column detected:' % len(all_zero_cols), all_zero_cols)
      
  return all_zero_cols.tolist()
  
  
class FeatureNormalizer(object):
    """Feature normalizer class

    Accumulates feature statistics

    Examples
    --------

    >>> normalizer = FeatureNormalizer()
    >>> for feature_matrix in training_items:
    >>>     normalizer.accumulate(feature_matrix)
    >>>
    >>> normalizer.finalize()

    >>> for feature_matrix in test_items:
    >>>     feature_matrix_normalized = normalizer.normalize(feature_matrix)
    >>>     # used the features

    """
    def __init__(self, feature_matrix=None):
        """__init__ method.

        Parameters
        ----------
        feature_matrix : numpy.ndarray [shape=(frames, number of feature values)] or None
            Feature matrix to be used in the initialization

        """
        if feature_matrix is None:
            self.N = 0
            self.mean = 0
            self.S1 = 0
            self.S2 = 0
            self.std = 0
        else:
            self.mean = np.mean(feature_matrix, axis=0)
            self.std = np.std(feature_matrix, axis=0)
            self.N = feature_matrix.shape[0]
            self.S1 = np.sum(feature_matrix, axis=0)
            self.S2 = np.sum(feature_matrix ** 2, axis=0)
            self.finalize()

    def __enter__(self):
        # Initialize Normalization class and return it
        self.N = 0
        self.mean = 0
        
        self.S1 = 0
        self.S2 = 0
        self.std = 0
        return self

    def __exit__(self, type, value, traceback):
        # Finalize accumulated calculation
        self.finalize()

    def accumulate(self, stat):
        """Accumalate statistics

        Input is statistics dict, format:

            {
                'mean': numpy.mean(feature_matrix, axis=0),
                'std': numpy.std(feature_matrix, axis=0),
                'N': feature_matrix.shape[0],
                'S1': numpy.sum(feature_matrix, axis=0),
                'S2': numpy.sum(feature_matrix ** 2, axis=0),
            }

        Parameters
        ----------
        stat : dict
            Statistics dict

        Returns
        -------
        nothing

        """
        self.N += stat['N']
        self.mean += stat['mean']
        self.S1 += stat['S1']
        self.S2 += stat['S2']

    def finalize(self):
        """Finalize statistics calculation

        Accumulated values are used to get mean and std for the seen feature data.

        Parameters
        ----------
        nothing

        Returns
        -------
        nothing

        """

        # Finalize statistics
        self.mean = self.S1 / self.N
        self.std = np.sqrt((self.N * self.S2 - (self.S1 * self.S1)) / (self.N * (self.N - 1)))


        # In case we have very brain-death material we get std = Nan => 0.0
        self.std = np.nan_to_num(self.std)

        self.mean = np.reshape(self.mean, [1, -1])
        self.std = np.reshape(self.std, [1, -1])
        nan_idx = np.where(np.isnan(self.mean))[0]
        if len(nan_idx) > 0:
            print ('nan dim in mean:', nan_idx)
            
        #print("detecting from finalize..")
        return detect_all_zero_col(self.std)

    def normalize(self, feature_matrix):
        """Normalize feature matrix with internal statistics of the class

        Parameters
        ----------
        feature_matrix : numpy.ndarray [shape=(frames, number of feature values)]
            Feature matrix to be normalized

        Returns
        -------
        feature_matrix : numpy.ndarray [shape=(frames, number of feature values)]
            Normalized feature matrix

        """

        nan_idx = np.where(np.isnan(self.mean))[0]
        if len(nan_idx) > 0:
            print ('nan dim in mean:', nan_idx)
            
        #print("detecting from normalize..")    
        detect_all_zero_col(self.std)

        #print("self.std: ", self.std)
        
        return (feature_matrix - self.mean) / self.std 






def ExtractAndDumpFeatures( model, directory,  dirPath, paramDict, event, modelParams,
                             source_data = False,
                            mode = 'devtrain', mixture_data = True):
        
        

        #mode can be either train or test
        #kind can be either 'mixture_data' or 'source_data'
        
        

    if  mode == 'devtrain':
        
        print("Dumping mixture data (training) features ....")
    elif mode == 'devtest':
       
        print("Dumping mixture data (testing) features ....")
    elif mode == 'eval':
        print("dumping Evaluation dataset features...")
    elif mode == 'newTest':
        print("Dumping newTest features....")
        
    if mixture_data == True:
    
        
        AudioFilesPath = os.path.join(os.getcwd(),'../data/mixture_data',mode)
        
        x = os.listdir(AudioFilesPath)

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


        elif mode == 'eval':
            if modelParams['extraction_mode'] == 'ask':
               
                for i in x:
                    str1 = 'consider '+ str(i) +'?'
                    userinput = input(str1)
                    if userinput == 'y':
                        paramHashes.append(i)      
            else:
                paramHashes = [y for y in x if y[0]=='d']


        else:
            
            if modelParams['extraction_mode'] == 'ask':
               
                for i in x:
                    str1 = 'consider '+ str(i) +'?'
                    userinput = input(str1)
                    if userinput == 'y':
                        paramHashes.append(i)      
            else:
                paramHashes = [y for y in x if (y[0]=='2' )]
                #paramHashes = [y for y in x if ( (y[0:1]=='95') or (y[0:1]=='17') )]

    

        
        #paramHashes = ['956e330cf71c8a37e535d751c371bbb3','17ef7d1e096f3c83c6f52dc175f20461']

        print(paramHashes)
        #20b255387a2d0cddc0a3dff5014875e7
        for paramHash in paramHashes:


            

            AudioFilesPath = os.path.join(os.getcwd(),'../data/mixture_data',mode)

            AudioFilesPath = os.path.join(AudioFilesPath,paramHash,'audio')
                
            ListOfAudioFiles = os.listdir(AudioFilesPath)

            for AudioFile in ListOfAudioFiles:

                # count = count +1

                # if( (paramHash[0:1] == '95' or paramHash[0:1] == '17') and count >20):
                #     break
                
                splitstring = AudioFile.split("_")
                
                if event in splitstring:
                
                    PathToAudioFile = os.path.join(AudioFilesPath,AudioFile)
                    
                    print("loading : ", AudioFile)
                
                    data , sr = load_audio(PathToAudioFile)

                    
                    FeatureDict = GetFeatures(data,sr,paramDict,model,event, modelParams)
                        
                  
                    PathToSave = os.path.join(dirPath, mode ,event, 'mixture_features')

                                
                    FileToSave = os.path.join(PathToSave, (AudioFile[:-4]+'.h5'))
                    
                    print("Saving features to file ", (AudioFile[:-4]+'.h5') )
                    dd.io.save(FileToSave, FeatureDict)
                
                    
     #for source data:
    if source_data == True:
            
        print("Dumping source event (training) features ....")
        
        
        
        #remember to skip a few problematic files:
        skip = { 'gunshot' : ['350745.wav','365256.wav','127846.wav'],
                 'glassbreak' : ['233605.wav'], 
                 'babycry' : [] }

        AudioFilesPath = os.path.join(os.getcwd(),'../data/source_data/events',event)
        
        
        #print(AudioFilesPath)
        
        ListofAudioFiles = os.listdir(AudioFilesPath)
        ListofAudioFiles = [x for x in ListofAudioFiles if x[-4:] == '.wav']

        #print(len(ListofAudioFiles))

        for AudioFile in ListofAudioFiles:
            
            if AudioFile in skip[event]:
                continue
            
            print(("Loading audio file: ", AudioFile ))
            
            PathToAudioFile = os.path.join(AudioFilesPath,AudioFile)
            
            data , sr = load_audio(PathToAudioFile) 
            
            FeatureDict = GetFeatures(data,sr,paramDict,model,event,modelParams, source_data = True)
            
            PathToSave = os.path.join(dirPath, mode, event, 'source_features')
            
            print("Saving Audio File: ", (AudioFile[:-4]+'.h5'))
            FileToSave = os.path.join(PathToSave, (AudioFile[:-4]+'.h5'))
            
            dd.io.save(FileToSave, FeatureDict)

    # if mode=='eval':

    #     AudioFilesPath = os.path.join(os.getcwd(),'../TUT-rare-sound-events-2017-evaluation/audio')
            
    #     ListOfAudioFiles = os.listdir(AudioFilesPath)
        
    #     for AudioFile in ListOfAudioFiles:
            
    #         splitstring = AudioFile.split("_")
            
    #         if event in splitstring:
            
    #             PathToAudioFile = os.path.join(AudioFilesPath,AudioFile)
                
    #             print("loading : ", AudioFile)
            
    #             data , sr = load_audio(PathToAudioFile)

                
    #             FeatureDict = GetFeatures(data,sr,paramDict,model,event,modelParams)
                    
              
    #             PathToSave = os.path.join(dirPath, mode ,event, 'mixture_features')
                            
    #             FileToSave = os.path.join(PathToSave, (AudioFile[:-4]+'.h5'))
                
    #             print("Saving features to file ", (AudioFile[:-4]+'.h5') )
    #             dd.io.save(FileToSave, FeatureDict)

                            
                    
                
                
    return os.path.join(dirPath)      





def GetFeatures(data,sr,paramDict,model,event, modelParams, source_data = False):
    '''extract features'''

    d = paramDict
    
    eps = numpy.spacing(1)

    if source_data == True:
        win_length = d['source_data'][event]['win_length']
        percent_overlap = d['source_data'][event]['percent_overlap']
        n_fft = d['source_data'][event]['n_fft']
    else:
        win_length = d[model][event]['win_length']
        percent_overlap = d[model][event]['percent_overlap']
        n_fft = d[model][event]['n_fft']

    
            
    mel_basis = librosa.filters.mel(sr=d[model][event]['sr'],
                                n_fft=n_fft,
                                n_mels=d[model][event]['n_mels'],
                                fmin=d[model][event]['fmin'],
                                fmax=d[model][event]['fmax'],
                                )


    temp = win_length
    win_length = int(win_length*d[model][event]['sr'])
    fraction  = (100-percent_overlap)/100.0
    hop_length = int(fraction*temp*d[model][event]['sr'])




    magspec =  numpy.abs(librosa.stft(data + eps,
                                      n_fft=n_fft,
                                      win_length=win_length,
                                      hop_length=hop_length,
                                      center = False)) 

    powerspec =  magspec** 2


                                           
    if model == 'dnn' or model == 'rnn':

        print("extracting Mfcc features")
        mel_spectrum = numpy.dot(mel_basis, powerspec)

        mfcc = librosa.feature.mfcc(S=librosa.logamplitude(mel_spectrum),
                                    n_mfcc=d[model][event]['n_mfcc'])

        #print (d[model][event])


        #print("mfcc.shape = ", mfcc.shape)

        if not d[model][event]['mfcc0']:
            mfcc = mfcc[1:,:]

        feature_matrix = mfcc

        mfcc_delta = librosa.feature.delta(mfcc)

        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        if d[model][event]['delta']:

            feature_matrix = np.vstack((mfcc, mfcc_delta))

        if d[model][event]['doubledelta']:

            feature_matrix = np.vstack((mfcc, mfcc_delta, mfcc_delta2))

        feature_matrix = feature_matrix.T


        #group according to context:
        context = modelParams[model][event]['context']

        feature_matrix = slice_array_new(feature_matrix, model, context)

        print("shape of feature matrix after slicing: ", feature_matrix.shape)

        #set input dimension in yaml file:

        input_dimension = feature_matrix.shape[1]

        modelParams[model][event]['input_dimension'] = input_dimension

        DumpfeatureParams(modelParams,'modelParams.yaml')

        #print(feature_matrix.shape)
        #sys.exit()

        #print("shape of feature matrix: ", feature_matrix.shape)  
        return  {
                    'feat': feature_matrix,
                    'stat': {
                        'mean': np.mean(feature_matrix, axis=0),
                        'std': np.std(feature_matrix, axis=0),
                        'N': feature_matrix.shape[0],
                        'S1': np.sum(feature_matrix, axis=0),
                        'S2': np.sum(feature_matrix ** 2, axis=0),
                            }
                }

    elif model == 'cnn':

        print("extracting spectrogram")


        D = np.abs(librosa.stft(data + eps,
                                      n_fft=d[model][event]['n_fft'],
                                      win_length=win_length,
                                      hop_length=hop_length,
                                      center = False))**2

        S = librosa.feature.melspectrogram(S=D, n_mels = d[model][event]['n_mels'])
    
        S  = librosa.power_to_db(S, ref=np.max)

        feature_matrix = S


        context = modelParams[model][event]['context']

        feature_matrix = slice_array_new(S, model, context)


        print("shape of feature matrix after slicing: ", feature_matrix.shape)
        

        modelParams[model][event]['xdim'] = feature_matrix.shape[1]
        modelParams[model][event]['ydim'] = feature_matrix.shape[2]
        DumpfeatureParams(modelParams,'modelParams.yaml')


        #print(modelParams[model][event]['ydim'])
        #sys.exit()



        return  {
                    'feat': feature_matrix,
                    'stat': {
                        'mean': np.mean(feature_matrix, axis=0),
                        'std': np.std(feature_matrix, axis=0),
                        'N': feature_matrix.shape[0],
                        'S1': np.sum(feature_matrix, axis=0),
                        'S2': np.sum(feature_matrix ** 2, axis=0),
                            }
                }



def slice_array_new(inp, model, context = 1):
    
    
    #inp = mfcc or spectrogram
    if model == 'dnn' or model == 'rnn':
        
        
        #time resolution here is the original hop length while extracting mfcc features
         
         numSamples = int(inp.shape[0]-context) +1
         
         featureLength = int(inp.shape[1]*context)
         x = np.empty((0,featureLength))
         
         start = 0
         #print("numSamples: ", numSamples)
         for start in range(numSamples):
             stop = start+context
             
             x = np.vstack((x, np.reshape(inp[start:stop, :], (1,-1)) ))
             
             
         return np.asarray(x)
        
        
    #inp is the spectrogram    
    elif model == 'cnn':
        
        numSamples = int(inp.shape[1]-context) +1
        
        x = np.empty((numSamples, inp.shape[0],context))
        
        start = 0
        for start in range(numSamples):
            stop = start+context
            #print(x.shape)
            temp  = inp[:, start:stop]
            #print(temp.shape)
            x[start,:,:] =   inp[:, start:stop]
            
        return x

# def slice_array(x, step):
#     numofslices = int(x.shape[1]/step)
    
#     truncate_index = int(numofslices*step)
#     return np.asarray(np.hsplit(x[:, :truncate_index],numofslices))


import datetime
import os
#from createParams import *
#listofModels, listofInstances

#clear contents of feature directory before drunning this  

def do_feature_extraction(model, event, source_data = False, mode = 'devtrain', directory = None,
                            mixture_data = True):

    ''' mode can be 'devtrain', 'devtest' or 'eval' '''

    #First make the required directories: 
    if directory is None:
        directory = model + '_' + datetime.datetime.now().strftime("%I_%M_%p_%B_%d_%Y")

    print("directory: ", directory)


    dirPath = os.path.join(os.getcwd(), '../Features',directory)
    

    if not os.path.exists(dirPath):
        os.mkdir(dirPath)



    modePath = os.path.join(dirPath,mode)
    if not os.path.exists(modePath):
        os.mkdir(modePath)

    eventdir = os.path.join(modePath,event)
    if not os.path.exists(eventdir):
        os.mkdir(eventdir)

    sourceFeaturesDir = os.path.join(eventdir, 'source_features')
    mixtureFeaturesDir = os.path.join(eventdir, 'mixture_features')
    if not os.path.exists(sourceFeaturesDir):
        os.mkdir(sourceFeaturesDir)
    if not os.path.exists(mixtureFeaturesDir):
        os.mkdir(mixtureFeaturesDir)

    #load feature param YAML file:
    filename = 'featureParams.yaml' #present in the CWD
    filename2 = 'modelParams.yaml'

    paramDict = loadfeatureParams(filename)

    modelParams = loadfeatureParams(filename2)



    #now extract features:

    featurepath = ExtractAndDumpFeatures( model,directory, dirPath, paramDict, event, modelParams,
                         source_data = source_data, mode = mode , mixture_data = mixture_data)

        
    return directory #pass to label extraction, pass to feature normalization




########################## Feature Normalization ############################

def do_feature_normalization(model, event, directory, source_data = False, mode = 'devtrain', 
                                mixture_data = True):
    ''' mode can be 'devtrain', 'devtest' or 'eval' '''

    ''' featurePath is passed from Feature extraction function above'''

    '''directory is the string 'model+timestamp' '''

    #First make the required directories: 

    #directory = model + datetime.datetime.now().strftime("%I:%M%p %B %d, %Y")
    dirPath = os.path.join(os.getcwd(), '../NormalizedFeatures',directory)
    

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

    #load feature param YAML file:
    filename = 'featureParams.yaml' #present in the CWD

    paramDict = loadfeatureParams(filename)

    featurepath = NormalizeAndDumpFeatures( model, directory, paramDict, event,
                         source_data = source_data, mode = mode, mixture_data = mixture_data)



    return featurepath


def NormalizeAndDumpFeatures( model, directory, paramDict, event,
                               source_data = False, mode = 'devtrain', mixture_data = True):
    
    if mode=='devtrain':
        print("dumping  Normalized Training features")
    if mode=='devtest':
        print("dumping Normalized Testing(validation) features")

    PathtoDump = os.path.join(os.getcwd(),'../NormalizedFeatures')
    featurePath = os.path.join(os.getcwd(), '../Features',directory)
    
    if mixture_data == True: 

        

        #first normalize features

        normalizer = FeatureNormalizer()
            
            
        featuresFilesPath = os.path.join(featurePath,mode, event, 'mixture_features')

     
        ListofFeatureFiles = os.listdir(featuresFilesPath)
     
        for FeatureFile in ListofFeatureFiles:
         
            print("opening mixture feature file: ", FeatureFile)
         
            filepath = os.path.join(featuresFilesPath,FeatureFile)
            reloaded_featureDict = dd.io.load(filepath)
         
         
            #returns a list of all zero colums:
            normalizer.accumulate(reloaded_featureDict['stat'])

    if source_data == 'True':

        if mixture_data == False:
            normalizer = FeatureNormalizer()

        featuresFilesPath = os.path.join(featurePath,mode, event, 'source_features')

        ListofFeatureFiles = os.listdir(featuresFilesPath)
 
        for FeatureFile in ListofFeatureFiles:
     
            print("opening source feature file: ", FeatureFile)
         
            filepath = os.path.join(featuresFilesPath,FeatureFile)
            reloaded_featureDict = dd.io.load(filepath)
         
         
            #returns a list of all zero colums:
            normalizer.accumulate(reloaded_featureDict['stat'])


     
    
         
    allzerocols = normalizer.finalize()


    #dump normalized features:
            
    #featuresFilesPath = os.path.join(featurePath,mode, event, 'mixture_features')
    if mixture_data == True:
 
        ListofFeatureFiles = os.listdir(featuresFilesPath)



        for FeatureFile in ListofFeatureFiles:
             
            filepath = os.path.join(featuresFilesPath,FeatureFile)
            reloaded_featureDict = dd.io.load(filepath)
             
             
            #returns a list of all zero colums:
            normalizedFeatureMatrix = normalizer.normalize(reloaded_featureDict['feat'])
             
            saveFilePath = os.path.join(PathtoDump,directory,mode,event,
                            'mixture_features',FeatureFile)
             
            print("saving normalized mixture feature file: ", FeatureFile)
            dd.io.save(saveFilePath, normalizedFeatureMatrix)

    if source_data==True:

        featuresFilesPath = os.path.join(featurePath,mode, event, 'source_features')
        ListofFeatureFiles = os.listdir(featuresFilesPath)

        for FeatureFile in ListofFeatureFiles:
         
            filepath = os.path.join(featuresFilesPath,FeatureFile)
            reloaded_featureDict = dd.io.load(filepath)
             
             
            #returns a list of all zero colums:
            normalizedFeatureMatrix = normalizer.normalize(reloaded_featureDict['feat'])
             
            saveFilePath = os.path.join(PathtoDump,directory,mode,event,
                            'source_features',FeatureFile)
             
            print("saving normalized source feature file: ", FeatureFile)
            dd.io.save(saveFilePath, normalizedFeatureMatrix)


    pathToReturn = os.path.join(PathtoDump, directory)


    return pathToReturn
    