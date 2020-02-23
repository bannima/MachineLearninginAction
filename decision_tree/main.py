#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: main.py
Description: 
Author: Barry Chow
Date: 2020/2/10 6:46 PM
Version: 0.1
"""

from model.cart import CART
from model.tree import ID3Tree
from model.tree import ID3,C45
from numpy import *
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
    print(tree)'''

    classifier = ID3(max_depth=10,max_leafs=10,epsilon =0.001)
    dataset,labels = loadDataset('./dataset/loads.txt')
    classifier.fit(dataset[:,0:-1],dataset[:,-1],labels.tolist()[0])

    '''
    classifier = C45(max_depth=10,max_leafs=10,epsilon =0.001)
    dataset,labels = loadDataset('./dataset/loads.txt')
    classifier.fit(dataset[:,0:-1],dataset[:,-1],labels.tolist()[0])
    '''




