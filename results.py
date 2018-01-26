'''
use for testing of AdaBoost class
'''

import Boost as ab
import numpy as np
import pandas as pa
import DecisionTree as dt
from collections import Counter
import matplotlib.pyplot as plt

data_name = ('mushrooms', 'senate', 'connect4')[2]
depth_and_clssfr_limit = 100
optional_savefilename_extension = "_avg10_tree"  # if you don't want it to overwrite a file with the same data and depth conditions
include_tree = True
include_boost = False

global_dict = {
    'mushrooms': {
        'directory' : 'Data/mushrooms.csv',
        'save to'   : 'Data/Mushrooms/results/' + str(depth_and_clssfr_limit) +
                      '_depth' + optional_savefilename_extension + '.csv',
        'X range'   : range(1, 23),
        'y range'   : 0,
        'train size': 1000
    },
    'connect4' : {
        'directory' : 'Data/Connect4/connect-4.csv',
        'save to'   : 'Data/Connect4/results/' + str(depth_and_clssfr_limit) +
                      '_depth' + optional_savefilename_extension + '.csv',
        'X range'   : range(0, 42),
        'y range'   : 42,
        'train size': 1000
    },
    'senate'   : {
        'directory' : 'Data/house_votes.csv',
        'save to'   : 'Data/Senate/results/' + str(depth_and_clssfr_limit) +
                      '_depth' + optional_savefilename_extension + '.csv',
        'X range'   : range(1, 16),
        'y range'   : 0,
        'train size': 180
    }
}

print('***********************************************************')
'''
set up data
'''
data = pa.read_csv(global_dict[data_name]['directory'])
X = data.iloc[:, global_dict[data_name]['X range']].values
y = data.iloc[:, global_dict[data_name]['y range']].values.flatten()
if data_name == 'connect4':
    y = np.array([x if x == "win" else "loss" for x in y])

'''
Helper function definitions
'''


def runBoost(X_train, y_train, X_test, y_test, numClassifiers=10):
    '''
    initialize AdaBoost
    '''

    boost = ab.Boost()
    boost.train(X_train, y_train, cNum=numClassifiers)

    # test with training data
    pred_boost_train = boost.predict(X_train)
    error_rate_boost_train = (
            sum([0 if pred == true else 1 for (pred, true) in zip(y_train, pred_boost_train)]) / float(len(y_train)))

    # now test with remaining data
    pred_boost_test = boost.predict(X_test)
    error_rate_boost_test = (
            sum([0 if pred == true else 1 for (pred, true) in zip(y_test, pred_boost_test)]) / float(len(y_test)))

    '''
    Add prints for diagnostics and results here:
    '''
    '''print('')
    print('***** RESULTS ADABOOST *****')
    decisions = [c.root.splitCriteria for c in boost.classifiers]
    print('Decision Criteria per trees used: ', decisions)
    decision_columns = [c for c,_ in decisions]
    unique_cols = np.unique(decision_columns)
    col_counts = Counter(decision_columns).most_common()
    print('unique columns and frequency of split columns: ', col_counts)
    print('')

    print('Training Error: ', error_rate_boost_train)
    print('Test Error    : ', error_rate_boost_test)'''

    return error_rate_boost_train, error_rate_boost_test


def runTree(X_train, y_train, X_test, y_test, d):
    '''
    initialize Decision Tree
    '''

    # now for decision tree
    tree = dt.DecisionTree()
    tree.fitTree(X_train, y_train, max_depth=d)  # calling fitTree without maxDepth argument sets max depth to 999

    # test on training set
    pred_tree_train = tree.predict(X_train)
    error_rate_tree_train = (
            sum([0 if pred == true else 1 for (pred, true) in zip(y_train, pred_tree_train)]) / float(len(y_train)))

    # test on test set
    pred_tree_test = tree.predict(X_test)
    error_rate_tree_test = (
            sum([0 if pred == true else 1 for (pred, true) in zip(y_test, pred_tree_test)]) / float(len(y_test)))

    '''print('')
    print('***** RESULTS DECISION TREE *****')
    print('Depth: ', tree.depth)

    print('')
    print('Training Error: ', error_rate_tree_train)
    print('Test Error    : ', error_rate_tree_test)'''
    return error_rate_tree_train, error_rate_tree_test, tree.depth


# tree tests
iterations = depth_and_clssfr_limit
i_depth = range(1, iterations + 1)
#i_depth = [x if not x == 35 else 36 for x in i_depth]

t_train = list()
t_test = list()
t_depth = list()

b_train = list()
b_test = list()

for i in i_depth:
    print(i)
    t_tr = list()
    t_te = list()
    t_de = list()
    b_tr = list()
    b_te = list()
    for j in range(10):
        indices = range(0, len(y))
        trainSize = global_dict[data_name]['train size']  # ALTER SIZE OF TRAINING SET IN GLOBAL_DICT
        train_indices = np.random.choice(indices, trainSize, replace=False)
        test_indices = np.setdiff1d(indices, train_indices, assume_unique=True)

        X_tr = X[train_indices]
        y_tr = y[train_indices]
        X_te = X[test_indices]
        y_te = y[test_indices]

        if include_tree:
            tr, te, d = runTree(X_tr, y_tr, X_te, y_te, i)
        if include_boost:
            btr, bte = runBoost(X_tr, y_tr, X_te, y_te, i)
        if include_tree:
            t_tr.append(tr)
            t_te.append(te)
            t_de.append(d)
        if include_boost:
            b_tr.append(btr)
            b_te.append(bte)
    if include_tree:
        t_train.append(np.mean(t_tr))
        t_test.append(np.mean(t_te))
        t_depth.append(np.mean(t_de))
    if include_boost:
        b_train.append(np.mean(b_tr))
        b_test.append(np.mean(b_te))

print('Storing Results')
res = None
if not include_tree and include_boost:
    res = {'Boost class num': i_depth, 'boost train err': b_train, 'boost test err': b_test}
elif include_tree and not include_boost:
    res = {'tree train err': t_train, 'tree test err': t_test, 'Tree Depth': t_depth}
else:
    res = {'tree train err' : t_train, 'tree test err': t_test, 'Tree Depth': t_depth, 'Boost class num': i_depth,
           'boost train err': b_train, 'boost test err': b_test}
df = pa.DataFrame(data=res, index=i_depth)
df.to_csv(global_dict[data_name]['save to'])