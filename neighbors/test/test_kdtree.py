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
from numpy.linalg import norm
from sklearn.datasets import load_iris

from neighbors import KDTree

iris = load_iris()
iris_data = iris['data']
iris_target = iris['target']


class Test_KDTree():

    def test_build_kdtree(self):
        tree = KDTree()
        root = tree.build(iris_data, array(mat(iris_target).T))
        print(root.value, root.label)

    def test_min_1_point(self):
        tree = KDTree()
        tree.build(iris_data, array(mat(iris_target).T))

        points = [[4.3, 2.3, 4.6, 3.1], [1.3, 4.6, 3.6, 1.1], [8.1, 3.3, 2.1, 1.9], [1.7, 2.9, 4.3, 2.2]]
        for point in points:
            min_dist, min_node = tree.k_nearest_neighbor(1, point)

            print(min_dist[0], min_node[0].value, point)

            # calc all distances with given point
            distances = array([norm(point - vec) for vec in iris_data])
            min = distances[distances.argmin()]

            assert min == min_dist[0]

    def test_min_k_point(self):
        tree = KDTree()
        tree.build(iris_data, array(mat(iris_target).T))
        points = [[4.3, 2.3, 4.6, 3.1], [1.3, 4.6, 3.6, 1.1], [8.1, 3.3, 2.1, 1.9], [1.7, 2.9, 4.3, 2.2]]
        k = 5
        for point in points:
            k_nearesr_dist, k_nearesr_nodes = tree.k_nearest_neighbor(k, point)

            # calc all distances with given point
            distances = array([norm(point - vec) for vec in iris_data])
            k_min_dist = distances[distances.argsort()][:k]
            for (a, b) in zip(k_nearesr_dist, k_min_dist):
                assert a == b
