
from __future__ import print_function
import wave
import contextlib
import os
import h5py
import numpy as np
import math

from FeatureExtraction import *

audioFilePath = "D:\\GoogleDrive\\Summer 2017\\Internship\\TUT-rare-sound-events-2017-development\\" + \
				 "data\\source_data\\events\\babycry\\31527.wav"

WindowLength = 0.04
PercentOverlap = 50
TimeResolution = (100 - PercentOverlap)*WindowLength*0.01
x = 0

with contextlib.closing(wave.open(audioFilePath,'r')) as f1:
                                
                frames = f1.getnframes()
                rate = f1.getframerate()
                duration = frames / float(rate)
                samplesInFile = int( (duration - WindowLength)/TimeResolution)
                x = samplesInFile
                print("duration",duration)
                print("samplesinfile", samplesInFile)

args = [audioFilePath]

#print(args)
feat = CreateMfccAndDump( *args, onlyone = True)
mfcc = feat['feat']
print(mfcc.shape)

if(mfcc.shape[0] == x + 1):
    mfcc = mfcc[:-1][:]
    print(mfcc.shape)







