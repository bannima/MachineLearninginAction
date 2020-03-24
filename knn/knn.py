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
from collections import Counter


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
        '''
        predict the points in X
        using average voting strategy

        '''
        return [self._predict_sample(point) for point in X]

    def _predict_sample(self, point):
        '''
        predict single point

        '''
        k_nearest_dist, k_nearest_nodes = self.tree.k_nearest_neighbor(self.k, point)
        labels = [node.label for node in k_nearest_nodes]
        return Counter(labels).most_common(1)[0][0]

