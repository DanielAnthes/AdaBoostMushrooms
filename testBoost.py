'''
use for testing of AdaBoost class
'''


import AdaBoost as ab
import numpy as np
import pandas as pa

boost = ab.AdaBoost()
X = np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]])
y = np.array(['a','b','a','b','a','a','b','a','a','a','a','a','a','a','a','a','a','a','b','a','b','b','b','b','b','b','b','b','b','b','b','b','b','a','b','b','b','b','b','b'])

data = pa.read_csv('Data/mushrooms.csv')
y = data['class']                   #class
X = data.iloc[:,range(1,23)]        #attributes

attributes = X.columns.values
X_arr = X.values
y_arr = y.values

#boost.train(X,y)
boost.train(X_arr,y_arr, trainingSize=0.01,numClassifiers=200)