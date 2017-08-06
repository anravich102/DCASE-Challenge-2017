# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 11:28:27 2017

@author: User
"""

#import print function here


# def CreateMfccAndDump(*args, mode = 'train' ,events = None, kind = None ,paramFile = 'FeatureExtractionParams.yaml', onlyone = False):
    
#         #mode can be either train or test
#         #kind can be either 'mixture_data' or 'source_data'
        
        
#         """Static MFCC
#         Parameters
#         ----------
#         data : numpy.ndarray
#             Audio data
#         params : dict
#             Parameters
#         Returns
#         -------
#         list of numpy.ndarray
#             List of feature matrices, feature matrix per audio channel
            
#         """

#         if onlyone==True:
#             for file in args:
#                 print(file)
#                 data , sr = load_audio(file)     
                            
#                 FeatureDict = GetMfccFeatureDict(data,sr)

#                 return FeatureDict


        
#         if events is None:
#             events = ['babycry','glassbreak','gunshot']
        

#         if mode is None or mode is 'train':
#             mode = 'devtrain'
#             print("Dumping mixture data (training) features ....")
#         else:
#             mode = 'devtest'
#             print("Dumping mixture data (testing) features ....")
            
        
        
#         if kind is None or kind is 'mixture_data':
            
#             #print("boo")
            
#             #cwd1 = 'C:\\Users\\User\\Google Drive\\Summer 2017\\Internship\\TUT-rare-sound-events-2017-development\\data\\mixture_data'
#             #AudioFilesPath = os.path.join(cwd1,mode)
            
#             AudioFilesPath = os.path.join(os.getcwd(),'../data/mixture_data',mode)
            
#             paramHashes = os.listdir(AudioFilesPath)
            
#             for paramHash in paramHashes:
                
#                 AudioFilesPath = os.path.join(os.getcwd(),'../data/mixture_data',mode)

#                 AudioFilesPath = os.path.join(AudioFilesPath,paramHash,'audio')
                
                
           
#                 #upto /devtrain or /devtest
#                 # first dump for mixture data:
            
#                 for event in events:
                    
#                     ListOfAudioFiles = os.listdir(AudioFilesPath)
                    
#                     for AudioFile in ListOfAudioFiles:
                        
#                         splitstring = AudioFile.split("_")
                        
#                         if event in splitstring:
                        
#                             PathToAudioFile = os.path.join(AudioFilesPath,AudioFile)
                            
#                             print("loading : ", AudioFile)
                        
#                             data , sr = load_audio(PathToAudioFile)     
                            
#                             FeatureDict = GetMfccFeatureDict(data,sr)
                            
                          
#                             PathToSave = os.path.join(os.getcwd(),'../Features',mode,event,'mfcc')
                            
#                             #cwd2 = 'C:\\Users\\User\\Google Drive\\Summer 2017\\Internship\\TUT-rare-sound-events-2017-development'
#                             #PathToSave = os.path.join(cwd2,'Features',mode,event,'mfcc')
                            
                            
#                             FileToSave = os.path.join(PathToSave, (AudioFile[:-4]+'.h5'))
                            
#                             print("Saving features to file ", (AudioFile[:-4]+'.h5') )
#                             dd.io.save(FileToSave, FeatureDict)
                    
                        
#          #for source data:
#         if kind == 'source_data' :
                
#                 print("Dumping source event (training) features ....")
                
#                 for event in events:
                
#                     #remember to skip a few problematic files:
#                     skip = { 'gunshot' : ['350745.wav','365256.wav','127846.wav'],
#                              'glassbreak' : ['233605.wav'], 
#                              'babycry' : [] }
    
#                     AudioFilesPath = os.path.join(os.getcwd(),'../data/source_data/events',event)
                    
#                     #cwd2 = 'C:\\Users\\User\\Google Drive\\Summer 2017\\Internship\\TUT-rare-sound-events-2017-development\\data\\source_data\\events'
#                     #AudioFilesPath = os.path.join(cwd2,event)
                    
#                     print(AudioFilesPath)
                    
#                     ListofAudioFiles = os.listdir(AudioFilesPath)
#                     ListofAudioFiles = [x for x in ListofAudioFiles if x[-4:] == '.wav']

#                     print(len(ListofAudioFiles))
    
#                     for AudioFile in ListofAudioFiles:
                        
#                         if AudioFile in skip[event]:
#                             continue
                        
#                         print(("Loading audio file: ", AudioFile ))
                        
#                         PathToAudioFile = os.path.join(AudioFilesPath,AudioFile)
                        
#                         data , sr = load_audio(PathToAudioFile) 
                        
#                         FeatureDict = GetMfccFeatureDict(data,sr)
                        
                        
#                         #PathToSave = 'C:\\Users\\User\\Google Drive\\Summer 2017\\Internship\\TUT-rare-sound-events-2017-development'
#                         #PathToSave = os.path.join(PathToSave,'Features',mode,event,'mfcc')
                        
                        
#                         PathToSave = os.path.join(os.getcwd(),'../Features',mode,event,'mfcc')
                        
#                         print("Saving Audio File: ", (AudioFile[:-4]+'.h5'))
#                         FileToSave = os.path.join(PathToSave, (AudioFile[:-4]+'.h5'))
                        
#                         dd.io.save(FileToSave, FeatureDict)
                                
                        
                    
                    
#         return os.path.join(PathToSave,'..') 



        
        
        
# def GetMfccFeatureDict(data,sr):
#     eps = numpy.spacing(1)
                
#     mel_basis = librosa.filters.mel(sr=sr,
#                                 n_fft=2048,
#                                 n_mels=40,
#                                 fmin=0,
#                                 fmax=22050,
#                                 )

#     spectrogram_ =  numpy.abs(librosa.stft(data + eps,
#                                       n_fft=2048,
#                                       win_length=int(0.04*44100),
#                                       hop_length=int(0.02*44100),
#                                       center = False)) ** 2
                                           
#     D = librosa.amplitude_to_db(librosa.stft(data + eps,
#                                       n_fft=2048,
#                                       win_length=int(0.04*44100),
#                                       hop_length=int(0.02*44100),
#                                       center = False))
                        

#     mel_spectrum = numpy.dot(mel_basis, spectrogram_)

#     mfcc = librosa.feature.mfcc(S=librosa.logamplitude(mel_spectrum),
#                                 n_mfcc=20)

#     mfcc_delta = librosa.feature.delta(mfcc)
    
#     mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
#     feature_matrix = np.vstack((mfcc, mfcc_delta, mfcc_delta2))
  
#     feature_matrix = feature_matrix.T
    
#     #print(feature_matrix.shape)
#     #print(feature_matrix[0,:])
    
# #==============================================================================
# #                 print("shape of spectrogram: {}".format(spectrogram_.shape))
# #                 librosa.display.specshow(D, y_axis='log')
# #                 plt.colorbar(format='%+2.0f dB')
# #                 plt.title('Log-frequency power spectrogram')
# #==============================================================================
    
#     return  {
#                 'feat': feature_matrix,
#                 'stat': {
#                     'mean': np.mean(feature_matrix, axis=0),
#                     'std': np.std(feature_matrix, axis=0),
#                     'N': feature_matrix.shape[0],
#                     'S1': np.sum(feature_matrix, axis=0),
#                     'S2': np.sum(feature_matrix ** 2, axis=0),
#                         }
#             }

    


           
#first normaliza then load

# def LoadMfccFeatures(filepath, sourceEvent = None):
#     #filepath loads all mixture data
#     #source event = 'babycry' can load all babycry source files.
    
    
#     reloaded_featureDict = dd.io.load(filepath)
#     return reloaded_featureDict
    


#path  = CreateMfccAndDump(kind = 'mixture_data', mode = 'train')
#path  = CreateMfccAndDump(kind = 'source_data')

#path  = CreateMfccAndDump(kind = 'mixture_data', mode = 'test')


