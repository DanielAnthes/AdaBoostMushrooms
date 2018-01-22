import matplotlib.pyplot as plt
import numpy as np
import pandas as pa

data_name = ('mush', 'senate', 'con4')[2]
depth_and_clssfr_limit = 300

global_dict = {
    'mush': {
        'directory': 'Data/Mushrooms/results/' + str(depth_and_clssfr_limit) + '_depth.csv'
    },
    'con4': {
        'directory': 'Data/Connect4/results/' + str(depth_and_clssfr_limit) + '_depth.csv'
    },
    'senate': {
        'directory': 'Data/Senate/results/' + str(depth_and_clssfr_limit) + '_depth.csv'
    }
}

results = pa.read_csv(global_dict[data_name]['directory'])
fig1 = plt.figure()
tr = plt.plot(results['Boost class num'], results['boost train err'])
te = plt.plot(results['Boost class num'], results['boost test err'])
plt.legend(labels= ['Train Error','Test Error'])
plt.title('AdaBoost')
plt.xlabel("number of weak classifiers")
plt.ylabel("misclassification in %")
plt.show()

fig2 = plt.figure()
tr = plt.plot(results['tree train err'])
te = plt.plot(results['tree test err'])
plt.legend(labels= ['Train Error','Test Error'])
plt.title('Decision Tree')
plt.xlabel('maximum allowed depth')
plt.ylabel('misclassification in %')
plt.show()