#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: test_adaboost.py
Description: 
Author: Barry Chow
Date: 2021/3/12 12:21 PM
Version: 0.1
"""
from datasets import loadHorseColic
from ensemble import AdaBoostClassifier
from utils import accuracy_score

class TestAdaBoost:
    def test_adaboost(self):
        train_X,train_y,test_X,test_y = loadHorseColic()
        adaboost = AdaBoostClassifier()
        adaboost.fit(train_X,train_y)
        preds = adaboost.predict(test_X)
        print(accuracy_score(preds,test_y))
        assert accuracy_score(preds,test_y)>0.7




