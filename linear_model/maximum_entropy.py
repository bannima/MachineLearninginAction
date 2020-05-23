#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: maximum_entropy.py
Description: implementation of maximum entropy model
Author: Barry Chow
Date: 2020/5/22 9:05 PM
Version: 0.1
"""

import numpy as np

from base import Classifier
from utils import softmax


class FeatureFunction(object):
    '''
    feature function class
    '''

    def __init__(self):
        self.features = set()

    def build_features(self, X, y):
        '''
        build feature functions

        Parameters
        ----------
        X:array_like
        y:array_like

        Returns
        -------

        '''
        n_samples, n_features = X.shape
        for index in range(n_samples):
            x = X[index, :].tolist()
            for feat_index, feat_value in enumerate(x):
                self.features.add(tuple([feat_index, feat_value, y[feat_index]]))

    def get_feature_nums(self):
        '''
        get all feature nums

        Returns
        -------

        '''
        return len(self.features)

    def match_features_indices(self, x, y):
        '''
        match features and return the shot one
        Parameters
        ----------
        x
        y

        Returns
        -------

        '''
        match_indices = []
        index = 0
        for feat_index, feat_value, feat_y in self.features:
            if feat_y == y and x[feat_index] == feat_value:
                match_indices.append(index)
            index += 1
        return match_indices


class MaxEntropy(Classifier):
    '''
    maximum entropy model

    '''

    def __init__(self, features, epochs=5, eta=1e-2):
        '''

        Parameters
        ----------
        features: feature functions set
        epochs: the maximum train epochs
        eta: learning rate

        '''
        self.features = features
        self.epochs = epochs
        self.eta = eta
        self.class_num = None
        # the joint distribution of x and y
        self.Pxy = {}
        # the marginal distribution of x
        self.Px = {}

    def _init_params(self, X, y):
        '''
        initialize all parameters

        Parameters
        ----------
        X: array_like
        y: array_like

        Returns
        -------

        '''
        n_samples, n_features = X.shape
        self.class_num = np.max(y) + 1

        # initialize all distributions and features
        for index in range(n_samples):
            range_indices = X[index, :].tolist()

            if self.Px.get(tuple(range_indices)) is None:
                self.Px[tuple(range_indices)] = 1
            else:
                self.Px[tuple(range_indices)] += 1

            if self.Pxy.get(tuple(range_indices + [y[index]])) is None:
                self.Pxy[tuple(range_indices + [y[index]])] = 1
            else:
                self.Pxy[tuple(range_indices + [y[index]])] = 1

        for key, value in self.Pxy.items():
            self.Pxy[key] = 1.0 * self.Pxy[key] / n_samples
        for key, value in self.Px.items():
            self.Px[key] = 1.0 * self.Px[key] / n_samples

        # initialize w
        self.w = np.zeros(self.features.get_feature_nums())

    def _sum_exp_w_on_all_y(self, x):
        sum_w = 0
        for y in range(0, self.class_num):
            tmp_w = self._sum_exp_w_on_y(x, y)
            sum_w += np.exp(tmp_w)
        return sum_w

    def _sum_exp_w_on_y(self, x, y):
        tmp_w = 0
        match_feature_func_indices = self.features.match_features_indices(x, y)
        for match_feature_func_index in match_feature_func_indices:
            tmp_w += self.w[match_feature_func_index]
        return tmp_w

    def fit(self, X, y):
        self.eta = max(1.0 / np.sqrt(X.shape[0]), self.eta)
        self._init_params(X, y)
        x_y = np.c_[X, y]

        for epoch in range(self.epochs):
            count = 0
            np.random.shuffle(x_y)

            for index in range(x_y.shape[0]):
                count += 1
                x_point = x_y[index, :-1]
                y_point = x_y[index, -1:][0]

                # get joint distribution
                p_xy = self.Pxy.get(tuple(x_point.tolist() + [y_point]))
                # get marginal distribution
                p_x = self.Px.get(tuple(x_point))
                # delta w
                dw = np.zeros(shape=self.w.shape)
                match_feature_func_indices = self.features.match_features_indices(x_point, y_point)

                if len(match_feature_func_indices) == 0:
                    continue
                if p_xy is not None:
                    for match_feature_func_index in match_feature_func_indices:
                        dw[match_feature_func_index] = p_xy
                if p_x is not None:
                    sum_w = self._sum_exp_w_on_all_y(x_point)
                    for match_feature_func_index in match_feature_func_indices:
                        dw[match_feature_func_index] -= p_x * np.exp(self._sum_exp_w_on_y(x_point, y_point)) / (
                                    1e-7 + sum_w)

                # update w
                self.w += self.eta * dw

                # print
                if count % (X.shape[0] // 4) == 0:
                    print("processing:\tepoch:" + str(epoch + 1) + "/" + str(self.epochs) + ",percent:" + str(
                        count) + "/" + str(X.shape[0]))

    def _predict_prob(self, x):
        y = []
        for x_point in x:
            y_tmp = []
            for y_index in range(0, self.class_num):
                match_feature_func_indices = self.features.match_features_indices(x_point, y_index)
                tmp = 0
                for match_feature_func_index in match_feature_func_indices:
                    tmp += self.w[match_feature_func_index]
                y_tmp.append(tmp)
            y.append(y_tmp)
        return softmax(np.asarray(y))

    def predict(self, x):
        return np.argmax(self._predict_prob(x), axis=1)
