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

    debugger_working = True

    def __init__(self):
        self.classifiers = None
        self.csf_weights = None
        self.dict = None  # store translation between actual class names and {-1,1}
        self.classNames = None

    # takes data and labels, number of classifiers and the proportion of training examples used in each training run
    def train(self, X, y, numClassifiers=10):
        length = len(X)
        data = np.array(X)
        dataIndices = range(0, length)
        train_weights = np.full(length, 1 / float(length))
        self.classifiers = np.array([])  # np.full(numClassifiers, dt.DecisionTree())
        self.csf_weights = np.full(numClassifiers, 0.0)

        # translate classes
        classNames = self.classNames = np.unique(y)
        classes = np.array([-1 if c == classNames[0] else 1 for c in y])
        self.dict = {-1 : classNames[0], 1 : classNames[1]}

        for i in range(0, numClassifiers):
            # draw training set
            sampleIndices = np.random.choice(dataIndices, size=length, p=train_weights, replace= True)
            sampleX = data[sampleIndices]
            sampley = classes[sampleIndices]

            # train classifier
            tree = dt.DecisionTree()
            tree.fitTree(sampleX, sampley, max_depth=1)

            # update weights
            y_pred = tree.predict(data)
            alpha = self.compute_alpha(y_pred, classes)
            self.csf_weights[i] = alpha
            train_weights = self.compute_train_weights(y_pred, classes, train_weights, i)
            self.classifiers = np.append(self.classifiers, tree)

            # debugging workaround
            if not self.debugger_working:
                self.print_vals(alpha, sampleIndices, sampley, y_pred, train_weights)
                if i == 4:
                    break

    def predict(self, element):
        if len(self.classifiers) == 0:
            print("train classifier first")
            return None
        elif np.ndim(element) > 1:
            length = len(element)
            results = np.empty(length,dtype=str)
            for i in range(0, length):
                e = element[i]
                results[i] = self.predict(e)
            return results
        else:
            numClassifiers = len(self.classifiers)
            weightedClassifications = [c.predict(element) * w for (c, w) in zip(self.classifiers, self.csf_weights)]
            return self.convert((-1 if np.sum(weightedClassifications) < 0 else 1))

    def convert(self,element):
        if element not in self.dict.keys():
            print('invalid key')
            return None
        else:
            return self.dict[element]

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
        error_rate = (sum([0 if pred == true else 1 for (pred, true) in zip(y_pred, y_true)]) / float(len(y_true)))
        # debugger workaround
        if not self.debugger_working:
            print("error rate: ", error_rate)
        return 0.5 * np.log((1 - error_rate) / (error_rate + 0.000001)) #add very small constant to denominator to avoid division by 0

    def compute_train_weights(self, y_pred, y_true, train_weights, csf_index):
        csf_weight = self.csf_weights[csf_index]
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        new_weights = train_weights * np.exp(- float(csf_weight) * y_true * y_pred)
        normalized_weights = new_weights / float(sum(new_weights))
        return normalized_weights

    def print_vals(self, alpha, sample_idx, sample_y, y_pred, train_weights):
        print("csf weight: ", alpha)
        print("train weights: ", train_weights)
        print("sample idx: ", sample_idx)
        print("saple y: ", sample_y)
        print("y pred: ", y_pred)
        print("")