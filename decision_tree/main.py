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
from model.Tree import ID3Tree
from numpy import *
#load dataset
def loadDataset(filename):
    res = []
    for line in open(filename,'r'):
        line = line.strip(' \n').split('\t')
        res.append(line)
    dataset = res[1:];labels = res[0]
    return dataset,labels


def loadRegDataset(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return mat(dataMat)

if __name__ == '__main__':
    '''dataset,labels = loadDataset('./dataset/loads.txt')
    model = ID3Tree(dataset,labels)
    tree = model.createTree()
    print(tree)

    cart = CART()
    testMat = loadRegDataset('./dataset/ex00.txt')
    tree = cart.createTree(testMat)
    print(tree)'''

    cart = CART()
    testMat = loadRegDataset('./dataset/ex0.txt')
    tree = cart.createTree(testMat)
    print(tree)






