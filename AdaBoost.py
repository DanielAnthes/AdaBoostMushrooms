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

# necessary imports
import numpy as np
import DecisionTree as dt


class AdaBoost:

    def __init__(self):
        self.classifiers = None
        self.weights = None
        self.dict = None  # store translation between actual class names and {-1,1}
        # stuff I added
        self.alphas = None

    # takes data and labels, number of classifiers and the proportion of training examples used in each training run
    def train(self, X, y, numClassifiers=10, trainingSize=0.66):
        length = len(X)
        data = np.array(X)
        dataIndices = range(0, length)
        train_weights = np.full(length, 1 / float(length))
        self.classifiers = [dt.DecisionTree()] * numClassifiers  # I'm surprised this works as well

        # translate classes
        classNames = np.unique(y)
        classes = [-1 if c == classNames[0] else 1 for c in y]
        self.dict = {classNames[0]: -1, classNames[1]: 1}

        for i in range(0, numClassifiers):
            # draw training set
            nSamples = int(length * trainingSize)
            sampleIndices = np.random.choice(dataIndices, size=nSamples, p=train_weights)
            sampleX = data[sampleIndices]
            sampley = classes[sampleIndices]

            # train classifier
            tree = dt.DecisionTree()
            tree.fitTree(sampleX, sampley, max_depth=1)

    def predict(self, element):
        if self.classifiers == None:
            print("train classifier first")
            return None
        elif np.ndim(element) > 1:
            length = len(element)
            results = np.empty(length)
            for i in range(0, length):
                e = element[i]
                results[i] = self.predict(e)
            return results
        else:
            numClassifiers = len(self.classifiers)
            weightedClassifications = [c.predict(element) * w for (c, w) in zip(self.classifiers, self.weights)]
            return (-1 if np.sum(weightedClassifications) < 0 else 1)

    # small functions for ada boost

    def compute_alpha(self, error_rate):
        return 0.5 * np.log((1 - error_rate) / error_rate)

    def update_weights(self, clf_index):
        new_weights = np.array([])
        new_weights = self.test_weights * np.exp(
                -self.alphas[clf_index] * self.y * self.classifiers[clf_index].predict(self.x))
        normalized_weights = new_weights / sum(new_weights)
        self.test_weights = normalized_weights