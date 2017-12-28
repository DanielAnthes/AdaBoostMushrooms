import numpy as np
import DecisionTree as dt

temp = [[1,2,3],[4,5,6],[7,8,9]]
temp2 = [temp[2], temp[1], temp[0]]
for (t1, t2) in zip(temp, temp2):
    print(
        t1, t2
    )