"""
"do the citing thing"

paper by Shapiro
several tutorials

add short description here



Note:
    Notice that the error is measured with respect to the distribution Dt on which the weak learner
was trained. In practice, the weak learner may be an algorithm that can use the weights Dt on the
training examples. Alternatively, when this is not possible, a subset of the training examples can
be sampled according to Dt, and these (unweighted) resampled examples can be used to train the
weak learner.
"""

#necessary imports
import numpy as np
import DecisionTree as dt

class AdaBoost:

    def __init__(self):
        self.classifiers = None
        self.weights = None
        self.dict = None           #store translation between actual class names and {-1,1}

    #takes data and labels, number of classifiers and the proportion of training examples used in each training run
    def train(self, X, y, numClassifiers = 10, trainingSize = 0.66):
        length = len(X)
        data = np.array(X)
        dataIndices = range(0,length)
        weights = np.full(length, 1/float(length))
        self.classifiers = np.empty(numClassifiers)
        self.classifiers = np.empty(numClassifiers)

        #translate classes
        classNames = np.unique(y)
        classes = [-1 if c == classNames[0] else 1 for c in y]
        self.dict = np.array((classNames[0],-1),(classNames[1],1))


        for i in range(0,numClassifiers):
            #draw training set
            nSamples = int(length*trainingSize)
            sampleIndices = np.random.sample(dataIndices, size = nSamples, p = weights)
            sampleX = data[sampleIndices]
            sampley = classes[sampleIndices]

            #train classifier
            tree = dt.DecisionTree()
            tree.fitTree(sampleX,sampley, max_depth= 1)


