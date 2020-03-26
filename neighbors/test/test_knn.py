#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: test_knn.py
Description: 
Author: Barry Chow
Date: 2020/3/18 9:02 PM
Version: 0.1
"""
from numpy import array, mat
from sklearn.datasets import load_iris

from neighbors import KNN
from utils import accuracy_score


class Test_KNN():

    def test_knn(self):
        iris = load_iris()
        data = iris['data']
        target = iris['target']
        knn = KNN(k=3, tree='kdtree', distance='euclidean')
        knn.fit(data, array(mat(target).T))
        preds = knn.predict(data)
        print(accuracy_score(preds, target))
        assert accuracy_score(preds, target) > 0.9
