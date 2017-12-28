'''
use for testing of AdaBoost class
'''


import AdaBoost as ab
import numpy as np

boost = ab.AdaBoost()
X = np.array([[0,1],[1,0],[1,1],[0,0]])
y = np.array(['a','a','b','b'])

boost.train(X,y)