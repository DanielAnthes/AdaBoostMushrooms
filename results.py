'''
use for testing of AdaBoost class
'''


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




data = pa.read_csv('Data/mushrooms.csv')
y = data['class']                   #class
X = data.iloc[:,range(1,23)]        #attributes

attributes = X.columns.values
X_arr = X.values
y_arr = y.values
print(type(X_arr))
indices = range(0,len(y_arr))
trainSize = 1000
train_indices = np.random.choice(indices, trainSize, replace=False)
test_indices = np.setdiff1d(indices, train_indices, assume_unique=True)

X_train = X_arr[train_indices]
y_train = y_arr[train_indices]
X_test = X_arr[test_indices]
y_test = y_arr[test_indices]






'''
Helper function definitions
'''


def runTest(numClassifiers = 10):
    '''
    initialize AdaBoost
    '''

    boost = ab.AdaBoost()
    boost.train(X_train,y_train,numClassifiers=numClassifiers)

    #test with training data
    pred_boost_train = boost.predict(X_train)
    error_rate_boost_train = (sum([0 if pred == true else 1 for (pred, true) in zip(y_train, pred_boost_train)]) / float(len(y_train)))

    #now test with remaining data
    pred_boost_test = boost.predict(X_test)
    error_rate_boost_test = (sum([0 if pred == true else 1 for (pred, true) in zip(y_test, pred_boost_test)]) / float(len(y_test)))

    '''
    Add prints for diagnostics and results here:
    '''
    print('')
    print('***** RESULTS ADABOOST *****')
    decisions = [c.root.splitCriteria for c in boost.classifiers]
    print('Decision Criteria per trees used: ', decisions)
    decision_columns = [c for c,_ in decisions]
    unique_cols = np.unique(decision_columns)
    col_counts = Counter(decision_columns).most_common()
    print('unique columns and frequency of split columns: ', col_counts)
    print('')

    print('Training Error: ', error_rate_boost_train)
    print('Test Error    : ', error_rate_boost_test)

    return error_rate_boost_train,error_rate_boost_test

def runTree():

    '''
    initialize Decision Tree
    '''

    #now for decision tree
    tree = dt.DecisionTree()
    tree.fitTree(X_train,y_train) #calling fitTree without maxDepth argument sets max depth to 999

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

'''
Notes and preliminary results:

- Both AdaBoost and a simple Tree seem to generalize well for this dataset, even for very small training sizes. This could be due to the simple nature of the data
- currently the unrestricted DecisionTree performs slightly better than Adaboost with 20 classifiers
- try removing columns 4 and 19 since they seem (almost) sufficient to classify the data on their own, the other classes are almost never used

'''

'''attributes = X.columns.values
X_arr = X.values


attributes = np.delete(attributes,[4,6,19])
X_arr = np.delete(X_arr,[4,6,19],1)
print(X_arr.shape)
indices = range(0,len(y_arr))
train_indices = np.random.choice(indices, trainSize, replace=False)
test_indices = np.setdiff1d(indices, train_indices, assume_unique=True)

X_train = X_arr[train_indices]
y_train = y_arr[train_indices]
X_test = X_arr[test_indices]
y_test = y_arr[test_indices]

runTest()

results = list()
for i in range(1,201,10):
    b_train, b_test = runTest(i)
    results.append([b_train, b_test])

results_np = np.array(results)
df = pa.DataFrame(results_np)
print(df)
df.to_csv('Data/BoostResults')
print('')
print('*****done*****')
print('')
'''
runTree()