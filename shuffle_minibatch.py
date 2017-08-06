# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 10:51:23 2017

@author: Anirudh
"""

import numpy as np
#np.set_printoptions(threshold=np.inf)


#print(x)


#==============================================================================
# sort_idx  = np.argsort(x)
# x = x [sort_idx]
# y = y [sort_idx]
# 
# 
# print(x)
# print(y)
# 
# x = x[neglect:]
# y = y[neglect:]
# 
# total = x.shape[0]
# 
# temp = np.where(x==0)[0]
# numZeros = temp.shape[0] 
# numOnes = total - numZeros
# 
# 
# 
# one_indices = np.where(x==1)[0]
# 
# #print(y)
# 
# 
# ratio = int(numZeros/numOnes)
# 
# new_x = np.empty(x.shape)
# new_y = np.empty(y.shape)
# 
# 
# #==============================================================================
# # print(x.shape)
# # print(y.shape)
# # print(x)
# # print(y)
# # print(ratio)
# # print(one_indices)
# # print(numOnes)
# # print(total)
# #==============================================================================
# 
# 
# count = 0
# for i in range(numOnes):
#     
#     
#         new_x[count:count+ratio] = 0
#         new_y[count:count+ratio] = y[count:count+ratio]
#        
#         #print(count+ratio)
#         new_x[count+ratio] = 1
#         new_y[count+ratio] = y[one_indices[i]]
#         
# 
#         count = count+ratio+1
#==============================================================================

 




#==============================================================================
# print(x)
# print(y)
#==============================================================================


#==============================================================================
# print(new_x)
# print(new_y)
#==============================================================================



import numpy
import numpy  as np
import h5py,os


neglect = 0

# x = labels, y = features

x = np.hstack( (np.ones(5), np.zeros(20) ) ).reshape(25,1)
y = np.array(range(25)).reshape(25,1)

with h5py.File("featureTest.h5",'w') as ffile, h5py.File("labelTest.h5",'w') as lfile:

    dsetl = lfile.create_dataset("labels", data = x , shape = 
                                                    (25,1),
                                                    maxshape=(None, 1)) 
    
    dset = ffile.create_dataset("features", data = y , shape = 
                                                    (25,1),
                                                    maxshape=(None, 1)) 






with h5py.File("labelTest.h5",'r+') as lfile, h5py.File("featureTest.h5",'r+') as ffile:
    x = lfile['labels'][:]
    y = ffile['features'][:]
    
    xtemp = x.shape
    ytemp = y.shape
    
    sort_idx  = np.argsort(x,axis = 0)
    x = x[sort_idx].reshape(xtemp)
    y = y [sort_idx].reshape(ytemp)
    
    #print(x.shape)
    
    x = x[neglect:]
    y = y[neglect:]
    
    total = x.shape[0]
    
    temp = np.where(x==0)[0]
    numZeros = temp.shape[0] 
    numOnes = total - numZeros
    
    ratio = int(numZeros/numOnes)
    one_indices = np.where(x==1)[0]
    
    
    
    new_x = np.empty(x.shape)
    new_y = np.empty(y.shape)
    
    count = 0
    for i in range(numOnes):
        
        
            new_x[count:count+ratio] = 0
            new_y[count:count+ratio] = y[count:count+ratio]
           
            #print(count+ratio)
            new_x[count+ratio] = 1
            new_y[count+ratio] = y[one_indices[i]]
            
    
            count = count+ratio+1

    del lfile['labels']
    del ffile['features']
    
    lfile.create_dataset('labels', data= new_x)
    ffile.create_dataset('features', data= new_y)
    
        


with h5py.File("labelTest.h5",'r') as lfile, h5py.File("featureTest.h5",'r') as ffile:
    print(lfile['labels'][:])
    print(ffile['features'][:])
    

























