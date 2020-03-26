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
from heapq import heappop, nsmallest, heappush

from numpy import concatenate, shape, inf

from utils import DISTANCE


class Node(object):
    '''
    defination of node in tree

    '''

    def __init__(self, value=0, label=0, axis=0, left=0, right=0, parent=0, depth=0):
        self.value = value
        self.label = label
        self.axis = axis
        self.left = left
        self.right = right
        self.parent = parent
        self.depth = depth

    @property
    def brother(self):
        if not self.parent:
            return None
        elif self.parent.left == self:
            return self.parent.right
        else:
            return self.parent.left

    # for heap operation
    def __lt__(self, other):
        return self.value[0] > other.value[0]


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
    def k_nearest_neighbor(self, k, point):
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

    def __init__(self, distance='euclidean'):
        super(KDTree, self).__init__(distance)
        self.root = None

    def build(self, X, y):
        '''
        build K-Dimension Tree
        '''
        self.n_samples, self.n_dimension = shape(X)
        data = concatenate((X, y), axis=1)
        self.root = self._build_tree(data, 0, None, 0)
        return self.root

    def _build_tree(self, data, axis, parent, depth):
        '''
        build tree for given dataset

        Parameters
        ----------
        data: specific data

        axis: cutting dimension

        depth: node depth in tree

        parent: node parent

        Returns
        -------
        the builded kd tree

        '''

        n_samples, n_dimension = shape(data)

        if n_samples == 0:
            return None

        if n_samples == 1:
            return Node(data[0][:-1], data[0][-1], axis, None, None, parent, depth)

        # sort the given dimension and get the middle number
        data = data[data[:, axis].argsort()]
        middle_index = n_samples // 2

        # current node
        node = Node(data[middle_index][:-1], data[middle_index][-1], axis, None, None, parent, depth)

        # left and right child
        next_axis = self._next_axis(axis)

        node.left = self._build_tree(data[:middle_index], next_axis, node, depth + 1, )
        node.right = self._build_tree(data[middle_index + 1:], next_axis, node, depth + 1)

        return node

    def _next_axis(self, axis):
        '''
        get next dimension
        '''
        return (axis + 1) % self.n_dimension

    def k_nearest_neighbor(self, k, point):
        '''
        search the k nearest neighbor in kd tree
        '''

        assert isinstance(k, int)
        assert 1 <= k <= self.n_samples
        if not self.root:
            raise RuntimeError("tree should be build first")

        self.heap = [(-inf, None) for _ in range(k)]
        self._nearest_k_point(self.root, point)
        k_nearest = [heappop(self.heap) for _ in range(k)]
        k_nearest.reverse()
        k_nearest_dist = [value[0] * -1 for value in k_nearest]
        k_nearest_nodes = [value[-1] for value in k_nearest]
        return k_nearest_dist, k_nearest_nodes

    def _nearest_k_point(self, root, point):

        '''
        given the point and starting root node
        search the nearest point in the kd tree starting from root node.

        Parameters
        ----------
        root: starting tree node, any inner kd tree node, may not be thr tree root.

        point: array_like

        Returns
        -------
        the nearest point in the tree

        '''

        path = self._nearest_leafnode(root, point)
        while path:
            cur_node = path.pop()
            dist = self.distance(cur_node.value, point)
            '''
            if dist<min_dist:
                min_dist = dist
                min_node = cur_node
            '''
            if nsmallest(1, self.heap)[0][0] < -1 * dist:
                # heappushpop(self.heap,(-1*dist,cur_node))
                heappush(self.heap, (-1 * dist, cur_node))
                heappop(self.heap)

            # skip the root node case, to avoid left and right brother cycle.
            if cur_node.brother and (len(path) != 0):
                axis = cur_node.parent.axis
                if abs(cur_node.parent.value[axis] - point[axis]) < -1 * nsmallest(1, self.heap)[0][0]:
                    self._nearest_k_point(cur_node.brother, point)

    def _nearest_leafnode(self, root, point):
        '''
        given starting root node and point,
        search the minimum distance leafnode in the tree.

        Note that the starting root may be inner node in the kd tree,
        not just the tree root.
        '''
        path = []
        cur_node = root
        # not include the root node
        path.append(cur_node)
        # get the nearest leaf node in the tree
        while cur_node.left or cur_node.right:
            if not cur_node.left:
                cur_node = cur_node.right
            elif not cur_node.right:
                cur_node = cur_node.left
            elif point[cur_node.axis] < cur_node.value[cur_node.axis]:
                cur_node = cur_node.left
            else:
                cur_node = cur_node.right
            path.append(cur_node)
        return path


TREE = {
    'kdtree': KDTree
}
