# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 15:09:27 2017

@author: User
"""

import h5py 
import numpy as np

x = np.zeros((10,1))
y = np.ones((10,1))

h5f = h5py.File("file1111.h5", 'w')
h5f.create_dataset('x', data=x)
print("saving data to h5 file")
h5f.close()
        
h5f = h5py.File("file2111.h5", 'w')
h5f.create_dataset('y', data=y)
print("saving data to h5 file")
h5f.close()

concat = h5py.File('file1111.h5', 'r')
print("loading data from h5 file")
reloaded = concat['x']

data = h5py.File('file2111.h5', 'r')
print("loading data from h5 file")
reloaded = np.vstack((reloaded, data['y']))

print (reloaded.shape)
print (type(reloaded))
print(reloaded)
