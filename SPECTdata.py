import AdaBoost as ab
import numpy as np
import pandas as pa
import DecisionTree as dt
from collections import Counter
import matplotlib.pyplot as plt

print('***********************************************************')
'''
set up data
'''

train_data = pa.read_csv('Data/SPECT.train.csv', header = None)
test_data = pa.read_csv('Data/SPECT.test.csv', header = None)

y_train = train_data.iloc[:,range(0,1)]
X_train = train_data.iloc[:,range(1,23)]

X_train_arr = X_train.values
y_train_arr = y_train.values.flatten()

y_test = test_data.iloc[:,range(0,1)]
X_test = test_data.iloc[:,range(1,23)]

X_test_arr = X_test.values
y_test_arr = y_test.values.flatten()

print('DEBUG: ', X_train_arr.shape)

tree = dt.DecisionTree()
tree.fitTree(X_train_arr,y_train_arr, max_depth= 12)
pred_tree_train = tree.predict(X_train_arr)
pred_tree_test = tree.predict(X_test_arr)
error_rate_tree_test = (sum([0 if pred == true else 1 for (pred, true) in zip(y_test_arr, pred_tree_test)]) / float(len(y_test_arr)))
error_rate_tree_train = (sum([0 if pred == true else 1 for (pred, true) in zip(y_train_arr, pred_tree_train)]) / float(len(y_train_arr)))

print('Train Error: ', error_rate_tree_train)
print('Test Error:  ', error_rate_tree_test)
print('Depth: ', tree.depth)

'''
Repeat for Adaboost
'''

boost = ab.AdaBoost()
boost.train(X_train_arr, y_train_arr, numClassifiers=20000,verbose=False)
pred_boost_train = boost.predict(X_train_arr)
pred_boost_test = boost.predict(X_test_arr)
pred_boost_train = [int(x) for x in pred_boost_train]
pred_boost_test = [int(x) for x in pred_boost_test]
error_rate_boost_test = (sum([0 if pred == true else 1 for (pred, true) in zip(y_test_arr, pred_boost_test)]) / float(len(y_test_arr)))
error_rate_boost_train = (sum([0 if pred == true else 1 for (pred, true) in zip(y_train_arr, pred_boost_train)]) / float(len(y_train_arr)))

print('')
print('Train Error: ', error_rate_boost_train)
print('Test Error:  ', error_rate_boost_test)
