#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: test_maximum_entropy.py
Description: 
Author: Barry Chow
Date: 2020/5/22 9:08 PM
Version: 0.1
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from linear_model import MaxEntropy, FeatureFunction


class DataBinWrapper(object):
    '''
    data discretization class
    '''

    def __init__(self, max_bins=10):
        '''

        Parameters
        ----------
        max_bins: the maximum bin nums
        '''
        self.max_bins = max_bins

        self.xrange_map = None

    def fit(self, X):
        '''

        Parameters
        ----------
        X: array_like, input data

        Returns
        -------

        '''
        n_samples, n_features = X.shape
        self.xrange_map = [[] for _ in range(0, n_features)]
        for index in range(0, n_features):
            # the index feature in X
            tmp = X[:, index]
            for percent in range(1, self.max_bins):
                percent_value = np.percentile(tmp, (1.0 * percent / self.max_bins) * 100 // 1)
                self.xrange_map[index].append(percent_value)

    def transform(self, x):
        '''

        Parameters
        ----------
        x: array_like,input

        Returns
        -------

        '''
        if x.ndim == 1:
            return np.asarray([np.digitize(x[i], self.xrange_map[i]) for i in range(0, x.size)])
        else:
            return np.asarray([np.digitize(x[:, i], self.xrange_map[i]) for i in range(0, x.shape[1])]).T


class TestMaxEntropy(object):

    def test_max_entropy(self):
        iris = load_iris()
        data = iris['data']
        target = iris['target']
        X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.3, random_state=0)

        # data transformation
        data_bin_wrapper = DataBinWrapper(max_bins=10)
        data_bin_wrapper.fit(X_train)
        X_train = data_bin_wrapper.transform(X_train)
        X_test = data_bin_wrapper.transform(X_test)

        featurefunction = FeatureFunction()
        featurefunction.build_features(X_train, Y_train)

        max_entropy = MaxEntropy(features=featurefunction)
        max_entropy.fit(X_train, Y_train)

        y = max_entropy.predict(X_test)
        print('f1:', f1_score(Y_test, y, average='macro'))
