'''
use for testing of AdaBoost class
'''


import AdaBoost as ab
import numpy as np
import pandas as pa
import DecisionTree as dt
from collections import Counter

boost = ab.AdaBoost()
X_arr = np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]])
y_arr = np.array(['a','b','a','b','a','a','b','a','a','a','a','a','a','a','a','a','a','a','b','a','b','b','b','b','b','b','b','b','b','b','b','b','b','a','b','b','b','b','b','b'])

data = pa.read_csv('Data/mushrooms.csv')
y = data['class']                   #class
X = data.iloc[:,range(1,23)]        #attributes

attributes = X.columns.values
X_arr = X.values
y_arr = y.values

indices = range(0,len(y_arr))
trainSize = 4000
train_indices = np.random.choice(indices, trainSize, replace=False)
test_indices = np.setdiff1d(indices, train_indices, assume_unique=True)

X_train = X_arr[train_indices]
y_train = y_arr[train_indices]
X_test = X_arr[test_indices]
y_test = y_arr[test_indices]



#boost.train(X,y)
boost.train(X_train,y_train,numClassifiers=20)

#now test with remaining data
pred_boost = boost.predict(X_test)
error_rate_boost = (sum([0 if pred == true else 1 for (pred, true) in zip(y_test, pred_boost)]) / float(len(y_test)))

#now for decision tree
tree = dt.DecisionTree()
tree.fitTree(X_train,y_train)
pred_tree = tree.predict(X_test)
error_rate_tree = (sum([0 if pred == true else 1 for (pred, true) in zip(y_test, pred_tree)]) / float(len(y_test)))

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


print('Adaboost error rate: ',error_rate_boost)




print('')
print('***** RESULTS DECISION TREE *****')
print('DecisionTree error rate: ', error_rate_tree)
print("done")


