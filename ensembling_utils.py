# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 13:46:28 2017

@author: Anirudh
"""

x = 'dnn_11_18_AM_August_02_2017_gunshot_12_45_PM_August_02_2017_ModelFile.h5'

ss = x.split("_")
events = ['gunshot','glassbreak','babycry']
eventname = ''
for event in events:
    if event in ss:
        new_ss = x.split(event)
        directoryname = new_ss[0][:-1]
        modelname = directoryname[:3]
        eventname = event
        break
    
    
d = {'s':1}

d1 = {'s1':1}
    
l = [d,d1]

for name in l:
    if name == d:
        print(name['s'])



l = [1,2,3,4,5,6,7,8,9,10]
import numpy as np
x = np.array(l)

print([l[i] for i in range(len(l)) if i%2==0 ])

print([x[i] for i in range(x.size) if i%2==0 ])