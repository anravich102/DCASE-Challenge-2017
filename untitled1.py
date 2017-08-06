# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 13:27:12 2017

@author: Anirudh
"""
#==============================================================================
# import h5py
# with h5py.File('myfile.hdf5','w') as f:
#     group = f.create_group('a_group')
#     group.create_dataset(name='matrix', data=np.zeros((10, 10)), chunks=True, compression='gzip')
#==============================================================================


if userAnnotatedFile3 is not None: 
      print("ENSEMBLE TEST SET RESULTS:")
    	print("directory: ", directory)
    	er, fscore,fp,fn,ins,dele,tp= get_metrics(userAnnotatedFile3, groundTruthFile, event) #produce metrics
        
        nt("fp:",fp,"fn:",fn, "ins:", ins, "dele", dele, "tp:", tp)
    	print("model: ", model)
    	print("event: ", event)
       	print("er :",er )
    	print("Fscore :",fscore)

        metricsFileName = model + '_' + directory + event + '_TEST_ENSEMBLE_METRICS.txt'
        metricsFilePath = os.path.join(os.getcwd(),'../Results',metricsFileName)

        with open(metricsFilePath,'w') as f:
            f.write(model + "\t" + event + "\t" + 'er:' + "\t" + str(er) + '\t' + 
                    'fscore: '+ str(fscore) + "\t" + "\n")

#==============================================================================
# import numpy as np
# 
# 
# D = np.abs(librosa.stft(y))**2
# S = librosa.feature.melspectrogram(S=D)
#==============================================================================

#==============================================================================
# import numpy as np
# 
# def func1(x= None):
#     
#     
#     if x is None:
#         print("st")
#     else:
#         print('dd')
#         
#         
# s  = str('% shuffling complete ')
# 
# print(s)
# 
# l = [1,2,3]
# 
# print(sum(l))
#==============================================================================

def f():
    return np.array([1, 2, 3]), np.array(["a", "b", "c"])

arr1, arr2 = f()


print(arr1.shape)
print(arr2.shape)


x= np.arange(5).reshape(5,1)
y = np.arange(3).reshape(3,1)
e = np.vstack((x,y)).reshape(8,1)


print(e)