import numpy as np
import DecisionTree as dt
import pandas as pa
import math
from collections import Counter

class Boost:
    def __init__(self):
        self.classifiers = list()
        self.csf_weights = list()
        self.classes = None #use to store class names for conversion to -1,1

    def train(self,X,y,cNum,verbose=False):
        #TODO add early stopping if no more improvement is achieved for more classifiers
        self.classes = np.unique(y)
        if verbose:
            print('Classes: ', self.classes)
        length = len(X)
        indices = range(length)
        w = np.ones(length)/float(length)

        for i in range(cNum):
            if verbose:
                print("Classifier: ",i)
            #get sample data
            sampleIndices = np.random.choice(indices, size=length, p=w, replace=True)
            sampleX = X[sampleIndices]
            sampley = y[sampleIndices]
            #train classifier
            #cl = dt.DecisionTree()
            #cl.fitTree(sampleX,sampley,1)
            cl = Stump(sampleX,sampley)
            #predictions on whole data set
            pred = cl.predict(X)
            #calculate error on entire data, weigh each error by weight of its datapoint
            err = [(0 if p==t else 1) for p, t in zip(pred, y)]
            err_weighted = err*w
            err_rate = sum(err_weighted) / float(sum(w))
            if verbose:
                print('Error Rate: ', err_rate)
            #calculate stage value ln((1-error)/error)
            stage = np.log((1-err_rate)) #add correction term to avoid division by 0
            if verbose:
                print('Stage value: ', stage)
            #update training weights with w = w*exp(stage*error)
            w = [weight*math.exp(stage*error) for weight,error in zip(w,err)]
            #normalize weights to sum to 1
            w = w/sum(w)
            #store classifier
            self.classifiers.append(cl)
            self.csf_weights.append(stage)

    def predict(self,X):
        if len(self.classifiers) == 0:
            print("train classifier first")
            return None
        elif np.ndim(X) > 1:
            length = len(X)
            results = list()
            for i in range(0, length):
                e = X[i]
                results.append(self.predict(e))
            return results
        else:
            #collect predictions from each classifier
            p = [c.predict(X) for c in self.classifiers]
            #convert to -1,1 format
            if(len(self.classes) != 2):
                print("Data should contain exactly two possible classes")
                return None
            else:
                p_con = [-1 if pred == self.classes[0] else 1 for pred in p]
                p_sum = sum(p_con)
                p_class = (self.classes[0] if p_sum < 0 else self.classes[1])
                return p_class

class Stump:
    def __init__(self,X,y):
        self.root = Node()

        y_unique = np.unique(y)
        if len(y_unique) == 1:
            self.root.classification = y_unique[0]

        else:
            self.root.splitCriteria = self.generateSplit(X, y)
            classes_left = list()
            classes_right = list()

            # generate binary split on condition: True is left child, False is right child
            for i in range(0, len(X)):
                element = X[i]
                if self.root.condition(element):
                    classes_left.append(y[i])
                else:
                    classes_right.append(y[i])
            left_counts = Counter(classes_left)
            self.root.left = Node()
            self.root.left.classification = left_counts.most_common(1)[0][0]

            right_counts = Counter(classes_right)
            self.root.right = Node()
            self.root.right.classification = right_counts.most_common(1)[0][0]


    def gini(self, y):
        length = float(len(y))
        unique, counts = np.unique(y, return_counts=True)
        helper = 0
        for i in range(0, len(counts)):
            helper += (counts[i] / length) ** 2
        return 1 - helper

    def generateSplit(self, X, y):
        # Splits will always be binary, for each attribute try all possible splits and return rule of the best one
        num_entries, num_attr = X.shape
        best_split = (None, None, 1)  # holds column, value to be split on and current lowest Gini
        # loop over all attributes
        for i in range(0, num_attr):
            currentBest = (None, None, 1)  # holds column, value to be split on and gini
            attr = X[:, i]  # single out relevant column
            unique = np.unique(attr)  # get unique values of attribute, split on each
            for u in unique:
                # get indices on condition
                left_indices = list()
                right_indices = list()
                for j in range(0, num_entries):
                    if attr[j] == u:
                        left_indices.append(j)
                    else:
                        right_indices.append(j)
                # use indices to split y
                y1 = y[left_indices]
                y2 = y[right_indices]
                # compute gini
                gini = (len(left_indices) / float(num_entries)) * self.gini(y1) + (
                        len(right_indices) / float(num_entries)) * self.gini(y2)
                if currentBest[2] > gini:
                    # update best condition
                    currentBest = (i, u, gini)
            # update overall best split
            if best_split[2] > currentBest[2]:
                best_split = currentBest

        return (best_split[0], best_split[1])  # returns column and value to be split on

    def predict(self, element):
        #add support for classification of multiple elements at once
        if np.ndim(element) > 1:
            length = len(element)
            results = []
            for i in range(0,length):
                e = element[i]
                results.append(self.predict(e))
            return np.array(results)

        else:
            current = self.root
            while current.classification == None:
                if current.condition(element):
                    current = current.left
                else:
                    current = current.right
            return current.classification



class Node():
    def __init__(self):
        self.classification = None
        self.splitCriteria = None  # column and value to split on
        self.left = None
        self.right = None

    def condition(self, X):
        # takes as input one datapoint
        if self.splitCriteria == None:
            print("no split Criteria set")
            return None
        else:
            col = self.splitCriteria[0]
            con = self.splitCriteria[1]
            return X[col] == con