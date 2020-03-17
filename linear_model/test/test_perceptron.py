#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: test_perceptron.py
Description: 
Author: Barry Chow
Date: 2020/3/9 7:53 PM
Version: 0.1
"""
from linear_model import Perceptron
import numpy as np

class TestPerceptron(object):
    def test_perceptron(self):
        clf = Perceptron(learning_rate=1)
        X,y = np.array([[3,3],[4,3],[1,1]]),np.array([1,1,-1])
        clf.fit(X,y)
        assert clf.score(X,y)>0.9



