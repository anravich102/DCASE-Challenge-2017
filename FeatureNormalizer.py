# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 20:08:08 2017

@author: User
"""

from __future__ import print_function
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
            
        print("detecting from finalize..")
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
            
        print("detecting from normalize..")    
        detect_all_zero_col(self.std)
        
        return (feature_matrix - self.mean) / self.std 




#make sure Makedir,py is run

def NormalizeAndDump(events= ['babycry','gunshot','glassbreak'], mode = 'devtrain', featuretype = 'mfcc' ):
    
    #filepath is a directory containing all training (or test features)
    #filepath contains upto Features/devtrain/babycry/mfcc
    
    #mode can be devtrain or devtest
    #events is a list of events
    
    #PathtoDump = 'C:\\Users\\User\\Google Drive\\Summer 2017\\Internship\\' + \
    #             'TUT-rare-sound-events-2017-development\\NormalizedFeatures'
    
    if mode=='devtrain':
        print("dumping  Normalized Training features")
    if mode=='devtest':
        print("dumping Normalized Testing(validation) features")
                
    PathtoDump = os.path.join(os.getcwd(),'../NormalizedFeatures')
    
    #FeaturesPath = 'C:\\Users\\User\\Google Drive\\Summer 2017\\Internship\\' + \
    #                'TUT-rare-sound-events-2017-development\\Features'
    
    for event in events:
        
         normalizer = FeatureNormalizer()
        
         FeaturesPath = os.path.join(os.getcwd(),'../Features',mode,event,featuretype)
        
         #FeaturesPath = os.path.join(FeaturesPath,mode,event,featuretype)
         
         ListofFeatureFiles = os.listdir(FeaturesPath)
         
         for FeatureFile in ListofFeatureFiles:
             
             print("opening feature file: ", FeatureFile)
             
             filepath = os.path.join(FeaturesPath,FeatureFile)
             reloaded_featureDict = dd.io.load(filepath)
             
             
             #returns a list of all zero colums:
             normalizer.accumulate(reloaded_featureDict['stat'])
             
            
                 
         allzerocols = normalizer.finalize()
        
        
             
             
        #dump normalized features:
            
         
         for FeatureFile in ListofFeatureFiles:
             
             filepath = os.path.join(FeaturesPath,FeatureFile)
             reloaded_featureDict = dd.io.load(filepath)
             
             
             #returns a list of all zero colums:
             normalizedFeatureMatrix = normalizer.normalize(reloaded_featureDict['feat'])
             
             saveFilePath = os.path.join(PathtoDump,mode,event,featuretype,FeatureFile)
             
             print("saving normalized feature file: ", FeatureFile)
             dd.io.save(saveFilePath, normalizedFeatureMatrix)
             
             
             
             
         
            
    
         
         
         
#NormalizeAndDump(mode='devtrain')       
#NormalizeAndDump(mode='devtest')         
         
         
    
    
    
    
    
    
    
    
    
    
    

