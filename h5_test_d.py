# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 17:06:42 2017

@author: Anirudh
"""
from __future__ import print_function
from temp import *

import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy
import librosa
import librosa.display

def package_feature_matrix(feature_matrix):
  """Package a feature matrix (time x feature)
  """
  return {'feat': feature_matrix
            , 'stat': {
                'mean': numpy.mean(feature_matrix, axis=0),
                'std': numpy.std(feature_matrix, axis=0),
                'N': feature_matrix.shape[0],
                'S1': numpy.sum(feature_matrix, axis=0),
                'S2': numpy.sum(feature_matrix ** 2, axis=0),
            }
        }
            
def ExtractSpectrogram(y, fs=44100, spek_params=None):
    """
    y: numpy.array [shape=(signal_length, )]
        Audio

    fs: int > 0 [scalar]
        Sample rate
        (Default value=44100)

    spek_params['fs_bound']: int > 0. The max frequency in
    the spectrogram.
    """
    f, t, Sxx = scipy.signal.spectrogram(y, fs,
            nperseg=2048,
            noverlap=1536)
    # +1 because f[0] = 0, and another +1 for int() rounding down.
    num_intervals = int(f.shape[0] *
            750 / fs * 2) + 2
    # Truncate frequency
    f = f[:num_intervals]
    Sxx = Sxx[:num_intervals, :]
    feature_matrix = numpy.log(Sxx).T
    return package_feature_matrix(feature_matrix), f, t


def spec(data,sr, model):
    eps = numpy.spacing(1)
                
    mel_basis = librosa.filters.mel(sr=sr,
                                n_fft=2048,
                                n_mels=60,
                                fmin=0,
                                fmax=8192,
                                )

    magspec=  numpy.abs(librosa.stft(data + eps,
                                      n_fft=2048,
                                      win_length=int(0.04*44100),
                                      hop_length=int(0.02*44100),
                                      center = False))
    
    powerspec = magspec**2
    
    
    mel_spectrum = numpy.dot(mel_basis, powerspec)

    mfcc = librosa.feature.mfcc(S=librosa.logamplitude(mel_spectrum),
                                    n_mfcc=20)
    
    
    
    
    D = librosa.amplitude_to_db(librosa.stft(data + eps,
                                      n_fft=2048,
                                      win_length=int(0.04*44100),
                                      hop_length=int(0.02*44100),
                                      center = False))
    
    
    
        
    #print(D.shape)
    
    #print("shape of spectrogram: {}".format(magspec.shape))
    librosa.display.specshow(D,  y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-frequency power spectrogram')

    #print(D.shape)
    if model == 'dnn':
        return mfcc.T
    else:
        return D

import os
path  = "D:\\GoogleDrive\\Summer 2017\\Internship\\TUT-rare-sound-events-2017-development\\data\\mixture_data\\devtrain\\20b255387a2d0cddc0a3dff5014875e7\\audio"
path = os.path.join(path,'mixture_devtrain_babycry_000_07a75692b15446e9fbf6cc3afaf96097.wav' )
y,sr = load_audio(path)


def slice_array(x, step):
    numofslices = int(x.shape[1]/step)
    
    truncate_index = int(numofslices*step)
    return np.asarray(np.hsplit(x[:, :truncate_index],numofslices))
#melgram = librosa.logamplitude(librosa.feature.melspectrogram(y, sr=sr, n_mels=96),ref_power=1.0)[np.newaxis,np.newaxis,:,:]

#print(melgram.shape)
#==============================================================================
# spec,d,f= ExtractSpectrogram(y,sr)
# 
# x = spec['feat']
# print(x.shape)
#==============================================================================
step  = 3


def slice_array_new(inp, model, context = 1):
    
    
    #inp = mfcc or spectrogram
    if model == 'dnn':
        
        
        #time resolution here is the original hop length while extracting mfcc features
         
         numSamples = int(inp.shape[0]-context) +1
         
         featureLength = int(inp.shape[1]*context)
         x = np.empty((0,featureLength))
         
         start = 0
         print("numSamples: !", numSamples)
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
            
            

import h5py

spec  = spec(y,sr, model = 'cnn')
print(spec.shape)

x = slice_array_new(spec, 'cnn', 7) 

print(x.shape)


#dset = myfile.create_dataset("MyDataset", (10, 1024), maxshape=(None, 1024))
#myfile = h5py.File("test2.h5", "w")
#myfile = 'testh5.h5'
#dset = myfile.create_dataset("features", data = x , shape = (1492,1025,7), maxshape=(None, 1025, 7))
#dset.shape
#myfile.close()



with h5py.File("test3.h5", "w") as f:
    
    dset = f.create_dataset("features", data = x , shape = (1492,1025,7), maxshape=(None, 1025, 7))

    dset.resize(dset.shape[0]+1492,  axis=0)   
    dset[-1492:] = x
    print(dset.shape)


#print(slice_array(spec,step ).shape)



#==============================================================================
# 
# from skimage.color import gray2rgb
# img = gray2rgb(spec)
# 
# from skimage import io
# import numpy as np
# import matplotlib.pyplot as plt
# 
# imgplot = skimage.io.imshow(img)
#==============================================================================


