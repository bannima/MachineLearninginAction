#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: naive_bayes.py
Description: implementation of naive bayes
Author: Barry Chow
Date: 2020/3/27 9:15 PM
Version: 0.1
"""

from numpy import array,mat,concatenate,unique,shape,nonzero
from base import BaseModel
from collections import Counter

class NaiveBayes(BaseModel):
    '''
    implementation of naive bayes model
    '''

    def __init__(self,smoothing=True):
        '''

        Parameters
        ----------
        smoothing: using laplace smoothing

        '''
        self.smoothing=smoothing
        super(NaiveBayes,self).__init__()

    def fit(self, X, y):
        '''
        fit the naive bayes model

        Parameters
        ----------
        X: array_like, input data

        y: array_like, input label

        Returns
        -------
        None

        '''
        n_samples,n_features = shape(X)
        assert n_samples==shape(y)[0]

        self.cond_prob = {}
        self.label_prob = {}

        for label in unique(y):

            self.cond_prob[label]={}
            label_index = nonzero(y==label)

            remain_data,remain_labels = X[label_index],y[label_index]

            n_remain_data = shape(remain_data)[0]

            self.label_prob[label] = float(n_remain_data)/n_samples

            #for each feature
            for fea_ind in range(n_features):
                if self.smoothing:
                    counter = Counter(unique(X[:,fea_ind]))
                else:
                    counter = Counter()

                counter += Counter(remain_data[:,fea_ind])
                #get conditional prob
                for key in counter.keys():
                    counter[key]=counter[key]/float(n_remain_data)
                self.cond_prob[label][fea_ind] = counter

    def predict(self, X):
        '''
        predictions of naive bayes

        Parameters
        ----------
        X: array_like, input data

        Returns
        -------
        preds: array_like, predict results according to naive bayes

        '''
        preds = []
        for sample in X:
            prediction = {}
            for label in self.cond_prob.keys():
                label_probs = self.label_prob[label]
                for fea_ind,fea_val in enumerate(sample):
                     label_probs *= self.cond_prob[label][fea_ind][fea_val]
                prediction[label] = label_probs
            preds.append(sorted(prediction.items(),key = lambda asv:asv[1],reverse=True)[0][0])

        return array(preds)

