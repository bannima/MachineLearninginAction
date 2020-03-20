#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: tree.py
Description: tree data structure used in K Nearest Neighbor
Author: Barry Chow
Date: 2020/3/20 8:27 PM
Version: 0.1
"""

from abc import ABCMeta, abstractmethod

from numpy import concatenate, shape

from utils import DISTANCE


class Node(object):
    '''
    defination of node in tree

    '''

    def __init__(self, value, label, left, right, depth, parent):
        self.value = value
        self.label = label
        self.left = left
        self.right = right
        self.parent = parent
        self.depth = depth


class Tree(metaclass=ABCMeta):
    '''
    abstract tree for KNN algorithms

    '''

    def __init__(self, distance):
        '''

        Parameters
        ----------

        distance: specific distance

        '''
        assert distance in DISTANCE
        self.distance = DISTANCE[distance]()

    @abstractmethod
    def build(self, X, y):
        '''
        build tree for input data X and y

        Parameters
        ----------
        X: array_like

        y: array_like

        Returns
        -------
        None

        '''
        pass

    @abstractmethod
    def search_k_nearest_neighbor(self, k, point):
        '''
        search k nearest neighbor for given data point

        Parameters
        ----------
        k: int

        point: array_like

        Returns
        -------
        k nearest points

        '''
        pass


class KDTree(Tree):

    def __init__(self, distance):
        super(KDTree, self).__init__(distance)

    def build(self, X, y):
        data = concatenate((X, y), axis=1)

    def _build(self, data, dimension, depth, parent):
        '''
        build tree for given dataset

        Parameters
        ----------
        data: specific data

        dimension: cutting dimension

        depth: node depth in tree

        parent: node parent

        Returns
        -------
        the builded kd tree

        '''

        n_samples, n_dimension = shape(data)

        if len(n_samples) <= 1:
            return Node(data[0][:-1], data[0][-1], None, None, parent, depth + 1)

    def search_k_nearest_neighbor(self, k, point):
        pass


class BallTree(Tree):
    '''
    implementation of ball tree
    '''

    def __init__(self, distance):
        super(BallTree, self).__init__(distance)

    def build(self, X, y):
        pass

    def search_k_nearest_neighbor(self, k, point):
        pass


TREE = {
    'kdtree': KDTree,
    'balltree': BallTree
}
