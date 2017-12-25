#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 15:37:20 2017

@author: daniel
loosely based on pseudocode from book 
"do the citing thing"

Takes data X as numpy array and y as list or array, currently the only purity measure implemented is the gini coefficient

"""

#%%
from collections import Counter

class DecisionTree:
    def __init__(self,purityMeasure='gini'):
        self.purity_measure = purityMeasure
        self.depth = 1
        self.root = None
        
    def classify(self,y)  :
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def generateSplit(self,X,y):
        #Splits will always be binary, for each attribute try all possible splits and return rule of the best one
        #TODO improve by introducing early stopping
        #TODO improve by returning split indices
        num_entries, num_attr = X.shape
        best_split = (None, None,1) #holds column, value to be split on and current lowest Gini
        #loop over all attributes
        for i in range(0,num_attr):
            currentBest = (None,None,1) #holds column, value to be split on and gini
            attr = X[:,i] #single out relevant column
            unique = np.unique(attr) # get unique values of attribute, split on each
            for u in unique:
                #get indices on condition
                left_indices =  list()
                right_indices = list()
                for j in range(0, num_entries):
                    if attr[j] == u:
                        left_indices.append(j)
                    else:
                        right_indices.append(j)
                #use indices to split y
                y1 = y[left_indices]
                y2 = y[right_indices]
                #compute gini
                gini = (len(left_indices)/float(num_entries)) * gini(y1) + (len(right_indices)/float(num_entries)) * gini(y2) 
                if currentBest[1] > gini:
                    #update best condition
                    currentBest = (i,u,gini)
            #update overall best split
            if best_split[2] > currentBest[2]:
                best_split = currentBest
        
        
        return (bestSplit[0],bestSplit[1]) #returns column and value to be split on
        
    def gini(self,y):
        length = float(len(y))
        unique,counts = np.unique(y, return_counts = True)
        helper = 0
        for i in range(0,len(counts)):
            helper += (counts[i]/length)**2
        return 1 - helper
            
        
    def stop(self, X,max_depth,minNodeSize):
        #also stop if there is no more gain?
        size = len(X)
        if self.depth >= max_depth:
            return True
        
        elif size < minNodeSize:
            return True
        
        elif size < 2:
            return True
        
        else:
            return False
        
    def fitTree(self,X,y,max_depth = None,minNodeSize=None):
        newNode = Node()
        if self.root == None: #in first step set root node
            self.root = newNode
        if stop(max_depth):
            #create leaf node, assign classification
            leaf = newNode
            leaf.classification = classify(X)
            
        else:
            #create node, generate test condition
            self.depth += 1
            node = newNode
            node.splitCriteria = generateSplit(X,y)
           
            indices_left  = list()
            indices_right = list()
            
            #generate binary split on condition: True is left child, False is right child
            for i in range(0,len(X)):
                if node.condition(X[i]):
                    indices_left.append(i)
                else:
                    indices_right.append(i)
                    
            #recursively call fitTree for child nodes
            node.left  = fitTree( X[indices_left], y[indices_left], max_depth, minNodeSize)
            node.right = fitTree(X[indices_right], y[indices_right], max_depth, minNodeSize)
    
    def predict(self, element):
        if self.root == None:
            print("call fitTree first to generate tree")
            return None
        else:
            current = self.root
            while current.classification == None:
                if current.condition(element):
                    current = current.left
                else:
                    current = current.right
            return current.classification
            
            
#Define Node objects for DecisionTree            
class Node():
    def __init__(self):
        self.classification = None
        self.splitCriteria = None #column and value to split on
        self.left = None
        self.right = None
    
    def condition(self,X):
        #takes as input one datapoint
        if self.splitCriteria == None:
            print("no split Criteria set")
            return None
        else:
            col = self.splitCriteria[0]
            con = self.splitCriteria[1]
            return X[col] == con
        
    