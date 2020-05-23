#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: test_naive_bayes.py
Description: 
Author: Barry Chow
Date: 2020/3/27 9:15 PM
Version: 0.1
"""
from numpy import array
from sklearn.datasets import load_iris

from bayes import NaiveBayes
from utils import accuracy_score


class Test_Naive_Bayes(object):

    def test_naive_bayes(self):
        data = array([
            [1, 'S'],
            [1, 'M'],
            [1, 'M'],
            [1, 'S'],
            [1, 'S'],
            [2, 'S'],
            [2, 'M'],
            [2, 'M'],
            [2, 'L'],
            [2, 'L'],
            [3, 'L'],
            [3, 'M'],
            [3, 'M'],
            [3, 'L'],
            [3, 'L'],
        ])
        labels = array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])

        nb = NaiveBayes()
        nb.fit(data, labels)
        preds = nb.predict(data)
        assert accuracy_score(preds, labels) > 0.7

    def test_nb_using_iris(self):
        iris = load_iris()
        data = iris['data']
        target = iris['target']
        nb = NaiveBayes()
        nb.fit(data, target)
        preds = nb.predict(data)
        assert accuracy_score(preds, target) > 0.9
