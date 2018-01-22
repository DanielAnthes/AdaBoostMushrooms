import matplotlib.pyplot as plt
import numpy as np
import pandas as pa

shrooms = pa.read_csv('Data/Mushrooms/results.csv')
fig1 = plt.figure()
tr = plt.plot(shrooms['Boost class num'], shrooms['boost train err'])
te = plt.plot(shrooms['Boost class num'], shrooms['boost test err'])
plt.legend(labels= ['Train Error','Test Error'])
plt.xlabel("number of weak classifiers")
plt.ylabel("misclassification in %")
plt.show()

fig2 = plt.figure()
tr = plt.plot(shrooms['tree train err'])
te = plt.plot(shrooms['tree test err'])
plt.legend(labels= ['Train Error','Test Error'])
plt.xlabel('maximum allowed depth')
plt.ylabel('misclassification in %')
plt.show()