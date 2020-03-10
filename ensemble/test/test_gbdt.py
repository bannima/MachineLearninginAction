#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: test_gbdt.py
Description: 
Author: Barry Chow
Date: 2020/3/3 4:15 PM
Version: 0.1
"""
from ensemble import GradientBoostingRegressor
from sklearn.datasets import load_iris
from numpy import *
from utils import accuracy_score,mean_square_error
from ensemble import GradientBoostingClassifier
iris = load_iris()


class Test_GBDT(object):
    def test_gbregressor(self):
        gbregressor = GradientBoostingRegressor()
        gbregressor.fit(mat(iris.data),mat(iris.target))
        preds = gbregressor.predict(mat(iris.data))
        assert mean_square_error(preds,iris.target)<5e-2


    def test_gbclassifier(self):
        gbclassifier = GradientBoostingClassifier()
        gbclassifier.fit(mat(iris.data),mat(iris.target))
        preds = gbclassifier.predict(mat(iris.data))
        assert accuracy_score(preds,iris.target)>0.95



