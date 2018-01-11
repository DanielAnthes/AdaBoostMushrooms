'''
use for testing of AdaBoost class
'''


import AdaBoost as ab
import numpy as np
import pandas as pa
import DecisionTree as dt

boost = ab.AdaBoost()
X = np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]])
y = np.array(['a','b','a','b','a','a','b','a','a','a','a','a','a','a','a','a','a','a','b','a','b','b','b','b','b','b','b','b','b','b','b','b','b','a','b','b','b','b','b','b'])

data = pa.read_csv('Data/mushrooms.csv')
y = data['class']                   #class
X = data.iloc[:,range(1,23)]        #attributes

attributes = X.columns.values
X_arr = X.values
y_arr = y.values

trainSize = 6000

X_train, X_test = np.split(X_arr,[trainSize])
y_train, y_test = np.split(y_arr, [trainSize])

'''
#boost.train(X,y)
boost.train(X_train,y_train,numClassifiers=3)

#now test with remaining data
pred_boost = boost.predict(X_test)
error_rate_boost = (sum([0 if pred == true else 1 for (pred, true) in zip(y_test, pred_boost)]) / float(len(y_test)))
print('Adaboost error rate: ',error_rate_boost)
'''
#now for decision tree
tree = dt.DecisionTree()
tree.fitTree(X_train,y_train)
pred_tree = tree.predict(X_test)
error_rate_tree = (sum([0 if pred == true else 1 for (pred, true) in zip(y_test, pred_tree)]) / float(len(y_test)))
print('DecisionTree error rate: ', error_rate_tree)
print("done")