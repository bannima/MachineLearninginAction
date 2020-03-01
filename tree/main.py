#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: main.py
Description: 
Author: Barry Chow
Date: 2020/2/10 6:46 PM
Version: 0.1
"""

from tree import ID3,CARTClassifier,CARTRegressor
from numpy import *
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor


#load dataset
def loadDataset(filename):
    res = []
    for line in open(filename,'r'):
        line = line.strip(' \n').split('\t')
        res.append(line)
    dataset = res[1:];labels = res[0]
    return mat(dataset),mat(labels)

def loadRegDataset(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return mat(dataMat)

if __name__ == '__main__':

    '''lenses = [line.strip().split('\t') for line in open('./dataset/lenses.txt').readlines()]
    lensesLables = ['age','prescript','astigmatic','tearRate']
    model = ID3Tree(lenses,lensesLables)
    lensesTree = model.createTree()
    print(lensesTree)
    
    dataset,labels = loadDataset('./dataset/loads.txt')
    model = ID3Tree(dataset,labels)
    tree = model.createTree()
    print(tree)

    cart = CART()
    testMat = loadRegDataset('./dataset/ex00.txt')
    tree = cart.createTree(testMat)
    print(tree)



    cart = CART()
    myMat = loadRegDataset('./dataset/ex2.txt')
    tree = cart.createTree(myMat)
    myTestMat = loadRegDataset('./dataset/ex2test.txt')
    cart.prune(tree,myTestMat)
    print(tree)
    '''

    id3 = ID3(max_depth=100, max_leafs=100, epsilon = 0.00001)
    dataset,labels = loadDataset('./dataset/loads.txt')
    id3.set_feature_labels(labels.tolist()[0])
    id3.fit(dataset[:, 0:-1], dataset[:, -1])
    print(id3.predict(dataset[:,0:-1]))
    print(dataset[:,-1])
    print("----------")


    from sklearn.datasets import load_iris
    iris = load_iris()

    cartclassifier = CARTClassifier(max_depth=100)
    cartclassifier.fit(mat(iris.data), mat(iris.target).T)

    cartregressor = CARTRegressor(max_depth=10)
    cartregressor.fit(mat(iris.data),mat(iris.target).T)

    print(cartregressor.predict(mat(iris.data)))
    print(iris.target)


    '''    
    classifier = C45(max_depth=10,max_leafs=10,epsilon =0.001)
    dataset,labels = loadDataset('./dataset/loads.txt')
    classifier.set_feature_labels(labels.tolist()[0])
    classifier.fit(dataset[:,0:-1],dataset[:,-1])
    
    from sklearn.datasets import load_iris
    from sklearn import tree

    iris = load_iris()  # 加载Iris数据集
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(iris.data, iris.target)

    res = clf.predict(iris.data)
    from sklearn.tree import plot_tree
    plot_tree(clf)
    print(res)
    '''






