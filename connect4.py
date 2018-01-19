import Boost as ab
import numpy as np
import pandas as pa
import DecisionTree as dt
from collections import Counter
import matplotlib.pyplot as plt

data = pa.read_csv('Data/connect-4.csv')
X = data.iloc[:,range(0,42)].values
y = data.iloc[:,42].values.flatten()
y = np.array([x if x == "win" else "loss" for x in y])
indices = range(0,len(y))
trainSize = 10000
train_indices = np.random.choice(indices, trainSize, replace=False)
test_indices = np.setdiff1d(indices, train_indices, assume_unique=True)

X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]
print("X Train shape",X_train.shape)
print("y Train shape", y_train.shape)
counts = Counter(y)
print(counts.most_common())

tree = dt.DecisionTree()
tree.fitTree(X_train,y_train, max_depth=10)
p_train = tree.predict(X_train)
p_test = tree.predict(X_test)

p_train = (sum([0 if pred == true else 1 for (pred, true) in zip(y_train, p_train)]) / float(len(y_train)))
p_test = (sum([0 if pred == true else 1 for (pred, true) in zip(y_train, p_test)]) / float(len(y_train)))

print("Train Error: ", p_train)
print("Test Error", p_test)
print("Tree Depth: ", tree.depth)

boost = ab.Boost()
boost.train(X_train, y_train, cNum=1000,verbose=True)
pred_boost_train = boost.predict(X_train)
pred_boost_test = boost.predict(X_test)
error_rate_boost_test = (sum([0 if pred == true else 1 for (pred, true) in zip(y_test, pred_boost_test)]) / float(len(y_test)))
error_rate_boost_train = (sum([0 if pred == true else 1 for (pred, true) in zip(y_train, pred_boost_train)]) / float(len(y_train)))
print("Boost Train Error: ",error_rate_boost_train)
print("Boost Test Error: ", error_rate_boost_test)