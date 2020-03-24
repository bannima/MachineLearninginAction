#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: knn.py
Description:
Author: Barry Chow
Date: 2020/3/18 7:52 PM
Version: 0.1
"""
from base import BaseModel
from .tree import TREE


class KNN(BaseModel):
    def __init__(self, k=5, tree='kdtree', distance='euclidean'):
        assert isinstance(k, int)
        self.k = k

        # tree type
        assert tree in TREE
        self.tree = TREE[tree](distance)

        super(KNN, self).__init__()

    def fit(self, X, y):
        '''
        train the specic k-dimension tree
        '''
        self.tree.build(X,y)

    def predict(self, X):
        pass
