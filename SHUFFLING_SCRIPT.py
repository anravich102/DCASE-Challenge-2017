# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 14:29:23 2017

@author: Anirudh
"""

# 
import numpy
import numpy  as np
import h5py,os,sys
from copy import deepcopy



#==============================================================================
def shuffle_data(featureFile, labelFile, total, batchsize,directory,event):
   
    file1 = featureFile
    file2 = labelFile
    
    if file1 =='' and file2 =='':
        file1 = directory+ '_' + event + '_' + 'devtrain' + '_features.h5'
        file2 = directory+ '_' + event + '_' + 'devtrain' + '_labels.h5'
        file1 = os.path.join('/hdd4', file1)
        file2 = os.path.join('/hdd4', file2)

    with h5py.File(file1, "r+") as ffile, h5py.File(file2,"r+") as lfile:
        for i in range(total):
            print( str(i)+"/"+str(total))
            start = i*batchsize
            stop = (i+1)*batchsize
            #print(ffile['features'].shape)
            featuresBatch  = ffile['features'][start:stop]
            labelsBatch  = lfile['labels'][start:stop] 
            idx_map = np.arange(batchsize)
            numpy.random.shuffle(idx_map)
    
            ffile['features'][start:stop] =  featuresBatch[idx_map]
            lfile['labels'][start:stop]  =  labelsBatch[idx_map]
            
#==============================================================================


def shuffle_data_new(featureFile, labelFile, directory,event, neglect):    

   file1 = featureFile
   file2 = labelFile 

   if file1 =='' and file2 =='':
        file1 = directory+ '_' + event + '_' + 'devtrain' + '_features.h5'
        file2 = directory+ '_' + event + '_' + 'devtrain' + '_labels.h5'
        file1 = os.path.join('/hdd4', file1)
        file2 = os.path.join('/hdd4', file2)

   with h5py.File(file2,'r+') as lfile, h5py.File(file1,'r+') as ffile:
        x = lfile['labels'][:]
   
        y = ffile['features'][:]

        xtemp = x.shape
        ytemp = y.shape

        print("shape of labels: ", x.shape)
        print("shape of features: ", y.shape)


        sort_idx  = np.argsort(x,axis = 0)
        x = x[sort_idx].reshape(xtemp)
        y = y [sort_idx].reshape(ytemp)



        #print(x.shape)

        #x = x[neglect:]
        #y = y[neglect:]

        total = x.shape[0]

        print("total", total)

        temp = np.where(x==0)[0]
        numZeros = temp.size
        numOnes = total - numZeros

        print("Unique elements: " , np.unique(x))
        #sys.exit()

        idx_shuffle_map = np.arange(numZeros)
        numpy.random.shuffle(idx_shuffle_map)

        #x[:numZeros] = x[idx_shuffle_map] 
        temp1 = y[idx_shuffle_map]
        y[:numZeros] = temp1

        idx_shuffle_map = np.arange(numOnes)
        numpy.random.shuffle(idx_shuffle_map)

        #x[numZeros:] = x[idx_shuffle_map] 
        temp2 = y[idx_shuffle_map]
        y[numZeros:] = temp2


        print("numZeros:", numZeros)
        print("numZeros: ", numZeros)
        print("numOnes: ", numOnes)

        ratio = int(numZeros/numOnes)

        print("Minimum batch size: ", ratio)
        one_indices = np.where(x==1 )[0]

        zero_indices = np.where(x==0)[0]

        # temp = np.setdiff1d(x, x[np.where(x!=1 )] )
        # weird_indices =np.setdiff1d(temp, temp[np.where(temp!=0 ) ] )
        # weird_indices = weird_indices[:10]

        # print(weird_indices)
        

        # print("shape of weird_indices" , weird_indices.shape)
        # print(x[weird_indices])
        # sys.exit()

        print("shape of one_indices" ,one_indices.shape)
        print("shape of zero_indices" ,zero_indices.shape)


        new_x = np.empty(x.shape)
        new_y = np.empty(y.shape)

        count = 0
        s  = str('% shuffling complete ')
        for i in range(numOnes):
            
            
                new_x[count:count+ratio] = 0
                new_y[count:count+ratio] = y[count:count+ratio]
               
                #print(count+ratio)
                new_x[count+ratio] = 1
                new_y[count+ratio] = y[one_indices[i]]
                

                count = count+ratio+1
                t  = (i+0.0)*100/numOnes

                if(i%200 == 0):
                    print("%.2f" % t, "% completed")
                



        del lfile['labels']
        del ffile['features']

        lfile.create_dataset('labels', data= new_x)
        ffile.create_dataset('features', data= new_y) 
        


    
    
