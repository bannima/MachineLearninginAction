#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: test_rf.py
Description: 
Author: Barry Chow
Date: 2021/3/10 4:10 PM
Version: 0.1
"""

from sklearn.datasets import load_iris
from numpy import mat
from ensemble import RandomForestClassifier,RandomForestRegressor
from utils import accuracy_score,mean_square_error

iris = load_iris()
data = iris['data']
target = iris['target']

class TestRandomForest():

    def test_randomforest_classifier(self):

        rf = RandomForestClassifier(n_estimators=30,sample_scale=0.67,feature_scale=0.6)
        rf.fit(mat(data),target)
        preds = rf.predict(mat(data))

        assert accuracy_score(preds,target)>0.95

    def test_randomforest_regressor(self):

        rf = RandomForestRegressor(n_estimators=30,sample_scale=0.67,feature_scale=0.6)
        rf.fit(mat(data),target)
        preds = rf.predict(mat(data))
        print(mean_square_error(preds,target))
        assert mean_square_error(preds,target)<2e-2
