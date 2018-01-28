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

senate_data = pa.read_csv('Data/house_votes.csv')
senate_data = senate_data.replace("?","u")
X_senate = senate_data.iloc[:,range(1,17)]
y_senate = senate_data.iloc[:,range(0,1)]
X_arr = X_senate.values
y_arr = y_senate.values
y_arr = y_arr.flatten()

indices = range(0,len(y_arr))
trainSize = 50
train_indices = np.random.choice(indices, trainSize, replace=False)
test_indices = np.setdiff1d(indices, train_indices, assume_unique=True)

X_train = X_arr[train_indices]
y_train = y_arr[train_indices]
X_test = X_arr[test_indices]
y_test = y_arr[test_indices]

print(y_train.shape)


tree = dt.DecisionTree()
tree.fitTree(X_arr,y_arr,max_depth=100)

#test on training set
pred_tree_train = tree.predict(X_train)
error_rate_tree_train = (sum([0 if pred == true else 1 for (pred, true) in zip(y_train, pred_tree_train)]) / float(len(y_train)))

#test on test set
pred_tree_test = tree.predict(X_test)
error_rate_tree_test = (sum([0 if pred == true else 1 for (pred, true) in zip(y_test, pred_tree_test)]) / float(len(y_test)))

print('')
print('***** RESULTS DECISION TREE *****')
print('Depth: ', tree.depth)

print('')
print('Training Error: ', error_rate_tree_train)
print('Test Error    : ', error_rate_tree_test)


cNums = range(1,31)
train_errs = list()
test_errs  = list()
for i in cNums:
    trains = list()
    tests = list()
    for j in range(20):
        boost = ab.Boost()
        boost.train(X_train, y_train, cNum=i,verbose=False)
        pred_boost_train = boost.predict(X_train)
        pred_boost_test = boost.predict(X_test[0])
        error_rate_boost_test = (sum([0 if pred == true else 1 for (pred, true) in zip(y_test, pred_boost_test)]) / float(len(y_test)))
        error_rate_boost_train = (sum([0 if pred == true else 1 for (pred, true) in zip(y_train, pred_boost_train)]) / float(len(y_train)))
        trains.append(error_rate_boost_train)
        tests.append(error_rate_boost_test)
    train_errs.append(np.mean(trains))
    test_errs.append(np.mean(tests))


plt.figure()
plt.scatter(cNums,train_errs, c = 'red')
plt.scatter(cNums, test_errs, c = 'blue')
plt.plot(cNums, [error_rate_tree_train]*30, c = 'orange')
plt.plot(cNums,[error_rate_tree_test]*30, c = 'purple')
plt.legend()
plt.show()

