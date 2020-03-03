#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: _plotter.py
Description:
Author: Barry Chow
Date: 2020/2/14 7:14 PM
Version: 0.1
"""

from sklearn import tree

classifier = tree.DecisionTreeClassifier(criterion='gini')


# return the leaf nums
def getNumLeafs(tree):
    numLeafs = 0
    rootNode = list(tree.keys())[0]
    leftTree = tree[rootNode]
    # single node
    if type(leftTree).__name__ != 'dict':
        return 1
    for key in leftTree.keys():
        if type(leftTree[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(leftTree[key])
        else:
            numLeafs += 1
    return numLeafs


# return the tree depths
def getTreeDepth(tree):
    maxDepth = 0
    rootNode = list(tree.keys())[0]
    leftTree = tree[rootNode]
    # single node
    if type(leftTree).__name__ != 'dict':
        return 1
    for key in leftTree:
        if type(leftTree[key]).__name__ == 'dict':
            if getTreeDepth(leftTree) + 1 > maxDepth:
                maxDepth = getTreeDepth(leftTree) + 1
    return maxDepth


if __name__ == '__main__':
    tree = {'有自己的房子': {'是': '是', '否': {'有工作': {'是': '是', '否': '否'}}}}
    print(getNumLeafs(tree))
    print(getTreeDepth(tree))
