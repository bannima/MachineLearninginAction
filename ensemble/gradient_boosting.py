#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: gradient_boosting.py
Description: implementation of gradient boosting tree
Author: Barry Chow
Date: 2020/3/1 4:52 PM
Version: 0.1
"""
from abc import ABCMeta

from numpy import *

from base import BaseModel
from tree import CARTRegressor
from tree import CRITERION
from utils import LOSS_FUNCTIONS
from utils import one_hot


class GradientBoostingModel(BaseModel, metaclass=ABCMeta):
    '''
    Basic Gradient Boosting Tree

    '''

    def __init__(self, n_estimators, loss, learning_rate, max_depth, max_leafs, min_sample_splits, epsilon, impurity):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leafs = max_leafs
        self.min_sample_splits = min_sample_splits
        self.epsilon = epsilon

        self.loss = loss
        assert self.loss in LOSS_FUNCTIONS

        self.loss_ = LOSS_FUNCTIONS[self.loss]()

        assert impurity in CRITERION
        self.impurity = impurity

        self._check_params()

    def _check_params(self):
        '''
        check params for gradient boosting model

        Note that the params of  basic tree model should be checked in Tree itself.
        Don't repeat those params checking in gradient boosting model.

        '''
        assert isinstance(self.n_estimators, int)
        assert self.n_estimators > 0
        assert 0 < self.learning_rate < 1


class GradientBoostingRegressor(GradientBoostingModel):
    '''
        Gradient Boosting Regressor model
    '''

    def __init__(self, n_estimators=100, loss='mse', learning_rate=1e-2, max_depth=10, max_leafs=10,
                 min_sample_splits=10, epsilon=1e-4, impurity='mse'):
        super(GradientBoostingRegressor, self).__init__(
            n_estimators,
            loss,
            learning_rate,
            max_depth,
            max_leafs,
            min_sample_splits,
            epsilon,
            impurity)

    def fit(self, X, y):
        gradient = copy(y)
        self.trees = []
        raw_predictions = zeros(shape(X)[0])
        for index in range(self.n_estimators):
            tree_ = CARTRegressor(max_depth=10, max_leafs=10, min_sample_splits=10, epsilon=1e-4, impurity='mse')
            tree_.fit(X, gradient)
            self.trees.append(tree_)
            raw_predictions += tree_.predict(X)
            # raw_predictions = self.predict(X)
            gradient = self.loss_.negative_gradient(raw_predictions, y)

    def predict(self, X):
        predictions = zeros(X.shape[0])
        for tree in self.trees:
            # predictions += np.multiply(self.learning_rate,tree.predict(X))
            predictions += tree.predict(X)
        return predictions


class GradientBoostingClassifier(GradientBoostingModel):
    '''
        Gradient Boosting Classifier model
    '''

    def __init__(self, n_estimators=100, loss='mse', learning_rate=1e-2, max_depth=10, max_leafs=10,
                 min_sample_splits=10, epsilon=1e-4, impurity='mse'):
        super(GradientBoostingClassifier, self).__init__(
            n_estimators,
            loss,
            learning_rate,
            max_depth,
            max_leafs,
            min_sample_splits,
            epsilon,
            impurity)

    def fit(self, X, y):
        '''
        The key idea for train gbdt classifier is to train K GBDT regressor,
        which corresponding to one class probability.

        '''
        self.gbregressors = []
        y_prob = one_hot(y.tolist()[0])
        n_samples, n_class = shape(y_prob)
        for index in range(n_class):
            gbregressor_ = GradientBoostingRegressor(self.n_estimators,
                                                     self.loss,
                                                     self.learning_rate,
                                                     self.max_depth,
                                                     self.max_leafs,
                                                     self.min_sample_splits,
                                                     self.epsilon,
                                                     self.impurity)
            gbregressor_.fit(X, y_prob[:, index])
            self.gbregressors.append(gbregressor_)

    def predict(self, X):
        '''
        choose the maximum probablity for each class

        '''
        res = array([])
        for gbregressor_ in self.gbregressors:
            class_pred = gbregressor_.predict(X)
            res = concatenate((res, class_pred), axis=0)
        res = res.reshape(-1, shape(X)[0])
        return res.argmax(axis=0)
