# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 18:26:32 2017

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 11:28:27 2017

@author: User
"""
#==============================================================================
# import hashlib
# 
# m = hashlib.sha256(str('sss').encode('utf-8')).hexdigest()
# print(m)
# 
# m = hashlib.sha256(str('ssss').encode('utf-8')).hexdigest()
# print(m)
#==============================================================================

#import print function here
from temp import *
import matplotlib.pyplot as plt
import numpy as np
import librosa.display


def mfcc(data):
    
        #mode can be either train or test
        #kind can be either 'mixture_data' or 'source_data'
        
        
        """Static MFCC
        Parameters
        ----------
        data : numpy.ndarray
            Audio data
        params : dict
            Parameters
        Returns
        -------
        list of numpy.ndarray
            List of feature matrices, feature matrix per audio channel
            
        """
        
    
            
            
            
        eps = numpy.spacing(1)
    
        mel_basis = librosa.filters.mel(sr=sr,
                                    n_fft=2048,
                                    n_mels=40,
                                    fmin=0,
                                    fmax=22050,
                                    )
    
        spectrogram_ =  numpy.abs(librosa.stft(data + eps,
                                          n_fft=2048,
                                          win_length=int(0.04*44100),
                                          hop_length=int(0.02*44100),
                                          center = False)) ** 2
                                               
        D = librosa.amplitude_to_db(librosa.stft(data + eps,
                                          n_fft=2048,
                                          win_length=int(0.04*44100),
                                          hop_length=int(0.02*44100),
                                          center = False))
                            

        mel_spectrum = numpy.dot(mel_basis, spectrogram_)
    
        mfcc = librosa.feature.mfcc(S=librosa.logamplitude(mel_spectrum),
                                    n_mfcc=20)
    
        mfcc_delta = librosa.feature.delta(mfcc)
        
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
#==============================================================================
#                 print("shape of spectrogram: {}".format(spectrogram_.shape))
#                 librosa.display.specshow(D, y_axis='log')
#                 plt.colorbar(format='%+2.0f dB')
#                 plt.title('Log-frequency power spectrogram')
#==============================================================================
        


#for source data:
    
    #remember to skip a few problematic files:
        feature_matrix = numpy.vstack((mfcc, mfcc_delta, mfcc_delta2))
               
               
        
        
        
            
        return feature_matrix.T


filename = 'C:\\Users\\User\\Google Drive\\Summer 2017\\Internship\\' +\
             'TUT-rare-sound-events-2017-development\\data\\' +\
             'mixture_data\\devtrain\\20b255387a2d0cddc0a3dff5014875e7'+\
             '\\audio\\mixture_devtrain_babycry_000_07a75692b15446e9fbf6cc3afaf96097.wav'
           
             
data , sr = load_audio(filename)        
feature_matrix = mfcc(data)

print("MFCC")
print(feature_matrix.shape)



        
        