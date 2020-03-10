#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: perceptron.py
Description: 
Author: Barry Chow
Date: 2020/3/9 6:22 PM
Version: 0.1
"""
from base import BaseModel
import numpy as np
from numpy import shape,inf
from utils import sign
from utils import accuracy_score

class Perceptron(BaseModel):
    '''
    Implementation of Perceptron
    '''
    def __init__(self,max_iterations=100,esplion=1e-3,learning_rate=0.1):
        assert max_iterations>0
        assert 1>esplion>0
        assert 0<learning_rate<=1

        self.max_iterations = max_iterations
        self.esplion = esplion
        self.learning_rate = learning_rate

    def fit(self, X, y):
        '''
        fit process of perceptron
        '''
        self.X = X
        self.y = y
        self._check_params()

        n_samples,n_features = shape(self.X)
        #Gram matrix
        gram_matrix = np.dot(self.X,self.X.T)
        self.alpha = np.zeros(n_samples)
        self.b = 0
        for iter in range(self.max_iterations):
            for ind in range(n_samples):
                #misclassification point
                if self.y[ind]*sum(self.alpha*(gram_matrix[:,ind].T*self.y))<=0:
                    self.alpha[ind]=self.alpha[ind]+self.learning_rate
                    self.b = self.b+self.y[ind]*self.learning_rate

            #compare accuracy
            if self.score(X,y)>0.9:
                break


    def score(self,X,y):
        return accuracy_score(y, self.predict(X))

    def _check_params(self):
        '''
        check params
        '''
        #assert type(self.X).__name__=='ndarray'
        #assert type(self.y).__name__=='ndarray'
        assert shape(self.X)[0]==len(self.y)

    def _predict_sample(self,sample):
        return sign(sum((sum((self.alpha*self.X.T*self.y).T))*sample)+self.b)

    def predict(self, X):
        return np.array([self._predict_sample(sample) for sample in X])



