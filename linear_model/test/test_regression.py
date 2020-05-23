#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: test_logistic_regression.py
Description: 
Author: Barry Chow
Date: 2020/3/25 8:21 PM
Version: 0.1
"""
from sklearn.datasets import load_breast_cancer

from linear_model import LogisticRegression, LinearRegression
from utils import accuracy_score, split_train_test

cancer = load_breast_cancer()
data = cancer['data']
labels = cancer['target']


class Test_Regression(object):

    def test_binary_Logistic_Regression(self):
        lr = LogisticRegression(learning_rate=1e-6, max_iter=1000, threshold=1e-4)
        train_X, train_y, test_X, test_y = split_train_test(data, labels, scale=0.7, is_random=True)
        lr.fit(train_X, train_y)
        preds = lr.predict(test_X)
        print(accuracy_score(preds, test_y))
        assert accuracy_score(preds, test_y) > 0.8

    def test_linear_regression(self):
        lr = LinearRegression(learning_rate=1e-6, max_iter=1000, threshold=1e-4)
        train_X, train_y, test_X, test_y = split_train_test(data, labels, scale=0.7, is_random=True)
        lr.fit(train_X, train_y)
        preds = lr.predict(test_X)
        print(accuracy_score(preds, test_y))
        assert accuracy_score(preds, test_y) > 0.8
