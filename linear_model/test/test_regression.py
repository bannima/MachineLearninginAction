#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: test_logistic_regression.py
Description: 
Author: Barry Chow
Date: 2020/3/25 8:21 PM
Version: 0.1
"""
from linear_model import LogisticRegression,LinearRegression
from sklearn.datasets import load_breast_cancer
from utils import accuracy_score

cancer = load_breast_cancer()
data = cancer['data']
labels = cancer['target']

class Test_Regression(object):

    def test_binary_Logistic_Regression(self):
        lr = LogisticRegression(learning_rate=1e-6,max_iter=1000,threshold=1e-4)
        lr.fit(data,labels)
        preds = lr.predict(data)
        assert accuracy_score(preds,labels)>0.8

    def test_linear_regression(self):
        lr = LinearRegression(learning_rate=1e-6, max_iter=1000, threshold=1e-4)
        lr.fit(data, labels)
        preds = lr.predict(data)
        assert accuracy_score(preds, labels) > 0.8