# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 20:21:46 2017

@author: Anirudh
"""


from __future__ import print_function

l1 = os.listdir('rnn_02_30_PM_August_03_2017/devtest/babycry/mixture_features')


l2 = os.listdir('rnn_01_57_PM_July_31_2017/devtest/babycry/mixture_features')



for name1,name2 in zip(l1,l2):
    if name1!=name2:
        print("nope")
        
    else:
        print("yes")