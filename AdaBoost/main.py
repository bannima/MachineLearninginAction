#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: main.py
Description: 
Author: Barry Chow
Date: 2020/2/17 3:10 PM
Version: 0.1
"""
from numpy import matrix
from model.adaboost import AdaBoost

def loadSimpData():
    dataMat = matrix([
        [1,2.1],
        [2,1.1],
        [1.3,1],
        [1,1],
        [2,1]])
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabels

if __name__ == '__main__':
    dataMat, classLabels = loadSimpData()
    adaboost = AdaBoost()
    classifierArray = adaboost.adaboostTrain(dataMat,classLabels,10)
