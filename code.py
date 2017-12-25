#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 13:22:38 2017

@author: daniel
"""
#%%

"""
Necessary imports
"""
import pandas as pa
import numpy as np
import matplotlib.pyplot as plt
import DecisionTree as dt

#%%

data = pa.read_csv('Data/mushrooms.csv')
y = data['class']                   #class
X = data.iloc[:,range(1,23)]        #attributes

attributes = X.columns.values
X_arr = X.values
y_arr = y.values

#%% test data

testX = np.array([[0,0],[0,0],[0,0],[0,0],[0,1],[0,1],[0,1],[0,1],[1,0],[1,0],[1,0],[1,0],[1,1],[1,1],[1,1],[1,1]])
testy = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3])

#%%

tree = dt.DecisionTree()
tree.fitTree(X = X_arr, y = y_arr, max_depth = 4)

#%%
#testing
tree.predict(X_arr[0])

