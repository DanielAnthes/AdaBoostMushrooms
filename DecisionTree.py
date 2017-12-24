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
    def __init__(self,X,y,purityMeasure='gini'):
        self.X = X
        self.y = y
        self.purity_measure = purityMeasure
        self.depth = 1
        self.root = None
        
    def generateSplit(self,e,f):
        #TODO 
        return 0
        
    def gini(self,e,attr):
        #attr should be index or name of column to be split on
        #Calculate Prerequisites
        c = e[:,attr]
        length = float(len(c))
        unique,counts = np.unique(c, return_counts = True)
        helper = 0
        for i in range(0,len(counts)):
            helper += (counts[i]/length)**2
        return 1 - helper
            
        
    def stop(self, e,max_depth,minNodeSize):
        #also stop if there is no more gain?
        size = len(e)
        if self.depth >= max_depth:
            return True
        
        elif size < minNodeSize:
            return True
        
        elif size < 2:
            return True
        
        else:
            return False
        
    def fitTree(self,e,f,max_depth = None,minNodeSize=None):
        if stop(max_depth):
            #create leaf node, assign classification
            leaf = Node()
            leaf.classification = classify(e)
            
        else:
            #create node, generate test condition
            self.depth += 1
            node = Node()
            node.condition = generateSplit(e,f)
           
            e1 = list()
            e2 = list()
            
            #generate binary split on condition: True is left child, False is right child
            for element in e:
                if node.condition(e):
                    e1.append(element)
                else:
                    e2.append(element)
                    
            #recursively call fitTree for child nodes
            node.left  = fitTree( e1, f, max_depth, minNodeSize)
            node.right = fitTree(e2, f, max_depth, minNodeSize)
    
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
        self.condition = None
        self.left = None
        self.right = None
    