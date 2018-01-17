import numpy as np
import DecisionTree as dt
import pandas as pa
import math

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
            cl = dt.DecisionTree()
            cl.fitTree(sampleX,sampley,1)
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
