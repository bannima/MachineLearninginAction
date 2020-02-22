#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: main.py
Description: adaboost 算法实现，参考机器学习实战第七章
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

def loadHorseColicDataset(filename):
    numFeat = len(open(filename).readline().split('\t'))
    dataMat = [];labelMat=[]
    fr = open(filename)
    for line in open(filename):
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

if __name__ == '__main__':
    dataMat, classLabels = loadSimpData()
    adaboost = AdaBoost()
    classifierArray = adaboost.adaboostTrain(dataMat,classLabels,10)
    predClass = adaboost.adaClassify([[5,5],[0,0]])
    print(predClass)

    #horse colic experiment
    dataArr,labelArr = loadHorseColicDataset('./dataset/horseColicTraining2.txt')
    adaboost = AdaBoost()
    classifierArray = adaboost.adaboostTrain(dataArr,labelArr,10)

    testArr,testLabelArr = loadHorseColicDataset('./dataset/horseColicTest2.txt')
    errorRate = adaboost.pred(testArr,testLabelArr)
    print("\n\nhorse colic testArrRate: ",errorRate)
