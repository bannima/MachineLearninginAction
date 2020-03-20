#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: test_kdtree.py
Description: 
Author: Barry Chow
Date: 2020/3/20 9:40 PM
Version: 0.1
"""
from numpy import mat, array
from sklearn.datasets import load_iris

from knn import KDTree


class Test_KDTree():

    def test_kdtree(self):
        iris = load_iris()
        iris_data = iris['data']
        iris_target = iris['target']
        tree = KDTree()
        root = tree.build(iris_data, array(mat(iris_target).T))
        print(root.value, root.label)

    