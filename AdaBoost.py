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
        self.csf_weights = None
        self.dict = None  # store translation between actual class names and {-1,1}

    # takes data and labels, number of classifiers and the proportion of training examples used in each training run
    def train(self, X, y, numClassifiers=10, trainingSize=0.66):
        length = len(X)
        data = np.array(X)
        dataIndices = range(0, length)
        train_weights = np.full(length, 1 / float(length))
        self.classifiers = np.full(numClassifiers, dt.DecisionTree())
        self.csf_weights = np.full(numClassifiers, 0)

        # translate classes
        classNames = self.classNames = np.unique(y)
        classes = np.array([-1 if c == classNames[0] else 1 for c in y])
        self.dict = {classNames[0]: -1, classNames[1]: 1}
        nSamples = int(length * trainingSize)

        for i in range(0, numClassifiers):
            # draw training set
            sampleIndices = np.random.choice(dataIndices, size=nSamples, p=train_weights)
            sampleX = data[sampleIndices]
            sampley = classes[sampleIndices]

            # train classifier
            self.classifiers[i].fitTree(sampleX, sampley, max_depth=1)

            # update weights
            #y_pred = np.array([self.dict[y] for y in self.classifiers[i].predict(data)])
            y_pred = np.array([-1 if y == 0 else 1 for y in self.classifiers[i].predict(data)])
            self.csf_weights[i] = self.compute_alpha(y_pred, classes)
            train_weights = self.compute_train_weights(y_pred, classes, train_weights, i)

        # normalize classifier weights
        self.csf_weights = self.csf_weights / sum(self.csf_weights)

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
            weightedClassifications = [c.predict(element) * w for (c, w) in zip(self.classifiers, self.csf_weights)]
            return (-1 if np.sum(weightedClassifications) < 0 else 1)

    # my own prediction function
    def predict2(self, data):
        length = len(data)
        results = []
        for i in range(length):
            score = 0
            for j in range(len(self.csf_weights)):
                score += self.csf_weights[j] * self.classifiers[j].predict(data[i])
            score = np.sign(score)
            results.append(score)
        return results

    # small functions for ada boost
    def compute_alpha(self, y_pred, y_true):
        error_rate = (sum([0 if pred == true else 1 for (pred, true) in zip(y_pred, y_true)]) / len(y_true))
        return 0.5 * np.log((1 - error_rate) / (error_rate + 0.000001))

    def compute_train_weights(self, y_pred, y_true, train_weights, csf_index):
        csf_weight = self.csf_weights[csf_index]
        y_true = np.array(y_true)
        new_weights = train_weights * np.exp(- csf_weight * y_true * y_pred)
        normalized_weights = new_weights / sum(new_weights)
        return normalized_weights