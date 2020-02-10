#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: main.py
Description: 
Author: Barry Chow
Date: 2020/2/10 6:46 PM
Version: 0.1
"""

from model.Tree import ID3Tree
#load dataset
def loadDataset(filename):
    res = []
    for line in open(filename,'r'):
        line = line.strip(' \n').split('\t')
        res.append(line)
    dataset = res[1:];labels = res[0]
    return dataset,labels

if __name__ == '__main__':
    dataset,labels = loadDataset('./dataset/loads.txt')
    model = ID3Tree(dataset,labels)
    tree = model.createTree()
    print(tree)




