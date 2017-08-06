import numpy as np





def slice_array(x, step):
    numofslices = int(x.shape[0]/step)
    
    truncate_index = int(numofslices*step)
    return np.asarray(np.vsplit(x[:truncate_index, :],numofslices))

#==============================================================================
# print( slice_array(x,5) )
# print( slice_array(x,5).shape)
#==============================================================================


#==============================================================================
# x = np.ones((22,1))
# print( slice_array(x,5) )
# print( slice_array(x,5).shape ) 
# 
# c= slice_array(x,5)
# print ( c[0])
# 
# print ( c[0].shape)
# 
# print(np.mean(c[0]) ==1 )
# 
#==============================================================================



#==============================================================================
# imageLabels = np.empty((0,1))
# imageLabels = np.vstack((imageLabels, np.array([1])))
# imageLabels = np.vstack((imageLabels, np.array([1])))
# print(imageLabels)
# print(imageLabels.shape)
#==============================================================================

#==============================================================================
# x = np.zeros((55,1))
# y = [12,12,12]
# index = 0
# 
# 
# for numFramesInFile in y:
#         
#         #files.append(file)
#             
#             print("outer index ",index)
#             print("numFramesInFile", numFramesInFile)
#            
#             predictionsForFile = x[index:index+numFramesInFile]
#             print(predictionsForFile)
#             index = index + numFramesInFile
#==============================================================================
            




















#==============================================================================
# metrics = {'babycry': {'er': 0.0, 'fscore': 0.0},
#            'glassbreak': {'er': 0.0, 'fscore': 0.0},
#            'gunshot': {'er': 0.0, 'fscore': 0.0} }
# 
# global_metrics = {}
# 
# models = ['dnn','cnn']
# events = ['babycry','gunshot','glassbreak']
# 
# for model in models:
#     global_metrics[model] = metrics
#     
# global_metrics['dnn']['babycry']['er'] = 9
# 
# print(global_metrics)
#==============================================================================
