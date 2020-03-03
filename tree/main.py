#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: main.py
Description: 
Author: Barry Chow
Date: 2020/2/10 6:46 PM
Version: 0.1
"""

from numpy import *
from tree import ID3, CARTClassifier, CARTRegressor
from sklearn.datasets import load_iris





def loadRegDataset(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return mat(dataMat)
