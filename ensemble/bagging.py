#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: bagging.py
Description: implementation of random forest
Author: Barry Chow
Date: 2021/3/10 4:08 PM
Version: 0.1
"""
from abc import ABCMeta
from numpy import random
from numpy import array,mat,mean
from numpy import argmax,bincount

from base import Classifier
from tree import CARTRegressor
from tree import CARTClassifier


class ForestModel(Classifier,metaclass=ABCMeta):
    '''
    Basic Random Forest Model
    '''

    def __init__(self, n_estimators, sample_scale, feature_scale):
        '''

        Parameters
        ----------
        n_estimators: 基本决策树的数目
        sample_scale: 每个决策树采样样本的比例
        feature_scale: 每个样本的特征进行采样的比例
        '''
        self.seed_val = 2021
        random.seed(self.seed_val)

        self.n_estimators = n_estimators

        self.sample_scale = sample_scale

        self.feature_scale = feature_scale

        self._check_params()

    def _check_params(self):
        '''
        check params for gradient boosting model

        Note that the params of  basic tree model should be checked in Tree itself.
        Don't repeat those params checking in gradient boosting model.

        '''
        assert isinstance(self.n_estimators, int)
        assert self.n_estimators > 0
        assert 0 < self.sample_scale < 1
        assert 0 < self.feature_scale < 1


class RandomForestRegressor(ForestModel):
    '''
    Random Forest Regressor
    '''

    def __init__(self, n_estimators, sample_scale, feature_scale):
        super(RandomForestRegressor, self).__init__(
            n_estimators,
            sample_scale,
            feature_scale)

    def fit(self, X, y):
        # M个训练样本，N个特征
        M, N = X.shape
        self.forest = []

        for i in range(self.n_estimators):
            # 生成行索引和列索引
            sample_indices = random.choice(M, int(self.sample_scale * M), replace=False)
            feature_indices = random.choice(N, int(self.feature_scale * N), replace=False)
            # 生成数据
            data = X[[sample_indices]]
            data = data[:, feature_indices]
            target = y[[sample_indices]]
            # 进行cart回归树训练
            basic_tree = CARTRegressor()
            basic_tree.fit(data, target)

            self.forest.append((basic_tree, feature_indices))

    def predict(self, X):
        res = []
        for tree, feature_indices in self.forest:
            data = X[:, feature_indices]
            pred = tree.predict(data)
            res.append(pred)

        res = mat(res).getA()

        return mean(res,axis=0)


class RandomForestClassifier(ForestModel):
    '''
    Random Forest Classifier
    '''

    def __init__(self, n_estimators, sample_scale, feature_scale):
        super(RandomForestClassifier, self).__init__(
            n_estimators,
            sample_scale,
            feature_scale)


    def fit(self, X, y):
        '''
        随机森林的训练就是训练n个基础cart决策分类树，再对结果进行统计，求投票表决最大者
        '''
        #M个训练样本，N个特征
        M,N = X.shape
        self.forest = []

        for i in range(self.n_estimators):
            #生成行索引和列索引
            sample_indices = random.choice(M,int(self.sample_scale*M),replace=False)
            feature_indices = random.choice(N,int(self.feature_scale*N),replace=False)
            #生成数据
            data = X[[sample_indices]]
            data = data[:,feature_indices]
            target = y[[sample_indices]]
            #进行cart分类树训练
            basic_tree = CARTClassifier()
            basic_tree.fit(data,target)

            self.forest.append((basic_tree,feature_indices))


    def predict(self, X):
        res = []
        for tree,feature_indices in self.forest:
            data = X[:,feature_indices]
            pred = tree.predict(data)
            res.append(pred)

        res = mat(res).getA()
        preds= []
        for i in range(len(X)):
            column = res[:,i]
            preds.append(argmax(bincount(column)))
        return preds

