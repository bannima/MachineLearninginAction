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
from base import BaseModel
from utils import LOSS_FUNCTIONS
from tree import CARTRegressor
from tree import CRITERION
from numpy import *
import numpy as np


class GradientBoostingModel(BaseModel,metaclass=ABCMeta):
    '''
    Basic Gradient Boosting Tree

    '''

    def __init__(self, n_estimators, loss,learning_rate, max_depth, max_leafs, min_sample_splits, epsilon, impurity):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leafs = max_leafs
        self.min_sample_splits = min_sample_splits
        self.epsilon = epsilon

        assert loss in LOSS_FUNCTIONS
        self.loss = LOSS_FUNCTIONS[loss]()

        assert impurity in CRITERION
        self.impurity = impurity

        self._check_params()

    def _check_params(self):
        '''
        check params for gradient boosting model

        Note that the params of  basic tree model should be checked in Tree itself.
        Don't repeat those params checking in gradient boosting model.

        '''
        assert isinstance(self.n_estimators,int)
        assert self.n_estimators>0
        assert 0<self.learning_rate<1


class GBRegressor(GradientBoostingModel):
    '''
    Gradient Boosting Regressor model
    '''

    def __init__(self, n_estimators=100, loss='mse', learning_rate = 1e-2,max_depth=10, max_leafs=10, min_sample_splits=10, epsilon=1e-4, impurity='mse'):
        super(GBRegressor,self).__init__(
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
            tree_ = CARTRegressor( max_depth=10, max_leafs=10, min_sample_splits=10, epsilon=1e-4, impurity='mse')
            tree_.fit(X,gradient)
            self.trees.append(tree_)
            raw_predictions+=tree_.predict(X)
            #raw_predictions = self.predict(X)
            gradient = self.loss.negative_gradient(raw_predictions,y)

    def predict(self, X):
        predictions = zeros(X.shape[0])
        for tree in self.trees:
            #predictions += np.multiply(self.learning_rate,tree.predict(X))
            predictions += tree.predict(X)
        return predictions


