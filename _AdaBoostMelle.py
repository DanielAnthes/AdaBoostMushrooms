import DecisionTree as dt
import numpy as np


class AdaBoostBinary:
    def __init__(self, purity_measure='gini'):
        self.purity_measure = purity_measure
        self.alphas = []
        self.classifiers = []
        self.test_weights = []
        self.x = []
        self.y = []

    def fit(self, x, y, n_classifiers=30):
        self.x = x
        self.y = y
        self.test_weights = [1 / len(y)] * len(y)
        self.classifiers = [dt.DecisionTree(purityMeasure=self.purity_measure)] * n_classifiers

        

    def compute_alpha(self, error_rate):
        return 0.5 * np.log((1 - error_rate) / error_rate)

    def update_weights(self, clf_index):
        new_weights = np.array([])
        new_weights = self.test_weights * np.exp(
            -self.alphas[clf_index] * self.y * self.classifiers[clf_index].predict(self.x))
        normalized_weights = new_weights / sum(new_weights)
        self.test_weights = normalized_weights