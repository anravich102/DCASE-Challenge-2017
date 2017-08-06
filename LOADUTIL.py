

#==============================================================================
# import soundfile as sf
# import os
# import numpy as np
# #audio = np.array([-0.23, 0.025,0.025, 0.025, 0.02365])
# #print(audio.shape)
# fs = 44100
# path = "C:\\Users\\User\\Google Drive\\testaudio.npy"
# path1 = "C:\\Users\\User\\Google Drive\\testaudio.wav"
#  # Only needed here to simulate closing & reopening file
# audio = np.load(path)
# sf.write(file=path1, data=audio, samplerate=fs, subtype='PCM_24')
#==============================================================================


#==============================================================================
# from scipy.io.wavfile import read
# filename = 'C:\\Users\\User\\Google Drive\\Summer 2017\\Internship\\' +\
#             'TUT-rare-sound-events-2017-development\\data\\' +\
#             'mixture_data\\devtrain\\20b255387a2d0cddc0a3dff5014875e7'+\
#             '\\audio\\mixture_devtrain_babycry_000_07a75692b15446e9fbf6cc3afaf96097.wav'
#        
#                 
# sr, data = scipy.io.wavfile.read(filename, mmap=False)
# print(data.shape)
#==============================================================================


import os
import wave
import numpy
import librosa


def load_audio(filename, mono=True, fs=44100):
    """Load audio file into numpy array

    Supports 24-bit wav-format, and flac audio through librosa.

    Parameters
    ----------
    filename:  str
        Path to audio file

    mono : bool
        In case of multi-channel audio, channels are averaged into single channel.
        (Default value=True)

    fs : int > 0 [scalar]
        Target sample rate, if input audio does not fulfil this, audio is resampled.
        (Default value=44100)

    Returns
    -------
    audio_data : numpy.ndarray [shape=(signal_length, channel)]
        Audio

    sample_rate : integer
        Sample rate

    """

    file_base, file_extension = os.path.splitext(filename)
    if file_extension == '.wav':
        audio_file = wave.open(filename)

        # Audio info
        sample_rate = audio_file.getframerate()
        sample_width = audio_file.getsampwidth()
        number_of_channels = audio_file.getnchannels()
        number_of_frames = audio_file.getnframes()
        #print ('sample_rate: {}'.format(sample_rate))
        #print ('sample_width: {}'.format(sample_width))
        #print ('number_of_channels: {}'.format(number_of_channels))
        #print ('number_of_frames: {}'.format(number_of_frames))

        # Read raw bytes
        data = audio_file.readframes(number_of_frames)
        audio_file.close()

        # Convert bytes based on sample_width
        num_samples, remainder = divmod(len(data),
            sample_width * number_of_channels)
        if remainder > 0:
            raise ValueError('The length of data is not a multiple of '
            'sample size * number of channels.')
        if sample_width > 4:
            raise ValueError('Sample size cannot be bigger than 4 bytes.')

        if sample_width == 3:
            # 24 bit audio
            a = numpy.empty((num_samples, number_of_channels, 4), dtype=numpy.uint8)
            raw_bytes = numpy.fromstring(data, dtype=numpy.uint8)
            a[:, :, :sample_width] = raw_bytes.reshape(-1, number_of_channels, sample_width)
            a[:, :, sample_width:] = (a[:, :, sample_width - 1:sample_width] >> 7) * 255
            audio_data = a.view('<i4').reshape(a.shape[:-1]).T
        else:
            # 8 bit samples are stored as unsigned ints; others as signed ints.
            dt_char = 'u' if sample_width == 1 else 'i'
            a = numpy.fromstring(data, dtype='<%s%d' % (dt_char, sample_width))
            audio_data = a.reshape(-1, number_of_channels).T

        if mono:
            # Down-mix audio
            audio_data = numpy.mean(audio_data, axis=0)

        # Convert int values into float
        audio_data = audio_data / float(2 ** (sample_width * 8 - 1) + 1)

        # Resample
        if fs != sample_rate:
            audio_data = librosa.core.resample(audio_data, sample_rate, fs)
            sample_rate = fs

        return audio_data, sample_rate

    elif file_extension == '.flac':
        audio_data, sample_rate = librosa.load(filename,
            sr=fs, mono=mono)

        return audio_data, sample_rate

    return None, None
    
    
#==============================================================================
# filename = 'C:\\Users\\User\\Google Drive\\Summer 2017\\Internship\\' +\
#              'TUT-rare-sound-events-2017-development\\data\\' +\
#              'mixture_data\\devtrain\\20b255387a2d0cddc0a3dff5014875e7'+\
#              '\\audio\\mixture_devtrain_babycry_000_07a75692b15446e9fbf6cc3afaf96097.wav'
#            
#              
# data , sr = load_audio(filename)
# 
# print(data.shape)
# 
# def DumpAudioAsArrray(mode, AudioPath = None): 
#     #AudioPath = os.getcwd(../data/mixture_data) by default
#     #mode is devtrain or devtest
#     
#     #again maintain file to check what param hashes are done with.
#     
#     files = os.listdir(AudioPath)
#     
#     return None
#     
# 
#==============================================================================




