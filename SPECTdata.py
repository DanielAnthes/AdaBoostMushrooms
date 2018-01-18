import Boost as ab
import numpy as np
import pandas as pa
import matplotlib.pyplot as plt
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

y_test = test_data.iloc[:,range(0,1)]
X_test = test_data.iloc[:,range(1,23)]

X_frames = [X_train,X_test]
y_frames = [y_train,y_test]

X = pa.concat(X_frames).values
y = pa.concat(y_frames).values.flatten()

indices = range(0,len(y))
trainSize = 50
train_indices = np.random.choice(indices, trainSize, replace=False)
test_indices = np.setdiff1d(indices, train_indices, assume_unique=True)

X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]



tree = dt.DecisionTree()
tree.fitTree(X_train,y_train, max_depth= 10)
pred_tree_train = tree.predict(X_train)
pred_tree_test = tree.predict(X_test)
error_rate_tree_test = (sum([0 if pred == true else 1 for (pred, true) in zip(y_test, pred_tree_test)]) / float(len(y_test)))
error_rate_tree_train = (sum([0 if pred == true else 1 for (pred, true) in zip(y_train, pred_tree_train)]) / float(len(y_train)))
print('TREE')
print('Train Error: ', error_rate_tree_train)
print('Test Error:  ', error_rate_tree_test)
print('Depth: ', tree.depth)


#Repeat for Adaboost


train_errs = list()
test_errs  = list()
duplicate = 30
cNums = range(1,81)

for i in cNums:
    trains = list()
    tests = list()
    for j in range(duplicate):
        boost = ab.Boost()
        boost.train(X_train, y_train, cNum=i,verbose=False)
        pred_boost_train = boost.predict(X_train)
        pred_boost_test = boost.predict(X_test)
        pred_boost_train = [int(x) for x in pred_boost_train]
        pred_boost_test = [int(x) for x in pred_boost_test]
        error_rate_boost_test = (sum([0 if pred == true else 1 for (pred, true) in zip(y_test, pred_boost_test)]) / float(len(y_test)))
        error_rate_boost_train = (sum([0 if pred == true else 1 for (pred, true) in zip(y_train, pred_boost_train)]) / float(len(y_train)))
        trains.append(error_rate_boost_train)
        tests.append(error_rate_boost_test)
    train_errs.append(np.mean(trains))
    test_errs.append(np.mean(tests))

plt.figure()
plt.scatter(cNums,train_errs, c = 'red')
plt.scatter(cNums, test_errs, c = 'blue')
plt.plot(cNums, [error_rate_boost_train]*80, c = 'orange')
plt.plot(cNums,[error_rate_boost_test]*80, c = 'purple')
plt.legend()
plt.show()
