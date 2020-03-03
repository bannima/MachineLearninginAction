#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: test_gbdt.py
Description: 
Author: Barry Chow
Date: 2020/3/3 4:15 PM
Version: 0.1
"""
from ensemble import GBRegressor
from sklearn.datasets import load_iris
from numpy import *
from utils import accuracy_score,mean_square_error

iris = load_iris()


class Test_GBRegressor(object):
    def test_gbrt(self):
        gbdt = GBRegressor()
        gbdt.fit(mat(iris.data),mat(iris.target))
        preds = gbdt.predict(mat(iris.data))
        assert mean_square_error(preds,iris.target)<5e-2




