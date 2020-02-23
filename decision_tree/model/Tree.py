#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: Tree.py
Description: 
Author: Barry Chow
Date: 2020/2/10 6:47 PM
Version: 0.1
"""
import math
from numpy import *

class ID3Tree:
    def __init__(self,dataset,labels,epsilon = 0.00001):
        self.epsilon = epsilon
        self.dataset = dataset
        self.featLabels  = labels

    #calc Shannon Entropy
    def __calcShannonEntropy(self,dataset):
        numEntries = len(dataset)
        labelCounts  = {}
        for featVec in dataset:
            currentLabel = featVec[-1]
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel]=0
            labelCounts[currentLabel]+=1
        shannonEntropy = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key])/numEntries
            shannonEntropy-=prob*math.log(prob,2)
        return shannonEntropy

    #given feature axis and value ,return the selected split dataset
    def __splitDatsetByAaixValue(self,dataset,axis,value):
        reducedDataset = []
        for featVec in dataset:
            if featVec[axis]==value:
                reduceFeatVec = featVec[:axis]
                reduceFeatVec.extend(featVec[axis:])
                reducedDataset.append(reduceFeatVec)
        return reducedDataset

    #calc Emperical Conditional Entropy,given feature axis
    def __calcEmpConEnt(self, dataset, axis):
        empConEnt = 0.0;featCount = {}
        for featVec in dataset:
            featValue = featVec[axis]
            if featValue not in featCount:
                featCount[featValue]=0
            featCount[featValue]+=1
        for featValue in featCount.keys():
            splitDataset = self.__splitDatsetByAaixValue(dataset,axis,featValue)
            prob = float(featCount[featValue])/len(dataset)
            empConEnt += prob*self.__calcShannonEntropy(splitDataset)
        return empConEnt

    #given feature axis, return information gain
    def __calcInformationGain(self,dataset,axis):
        return self.__calcShannonEntropy(dataset)-self.__calcEmpConEnt(dataset,axis)

    #choose the one feature by information gain, return -1 if information gain is less than epsilon
    def __chooseFeat(self,dataset):
        entropy = self.__calcShannonEntropy(dataset)
        greatestInformationGain  = -1.0
        theFeatInd = -1;
        for featVec in dataset:
            for featIndex in range(len(featVec[:-1])):
                if self.__calcInformationGain(dataset,featIndex)>greatestInformationGain:
                    greatestInformationGain = self.__calcInformationGain(dataset,featIndex)
                    theFeatInd = featIndex

        if greatestInformationGain >=self.epsilon:
            return theFeatInd
        return -1

    #create Tree
    def __createTree(self,dataset):
        while True:
            flag,currentLabel = self.__countLabels(dataset)
            # no features left or one label left or little information gain,
            #  it's left node and return the max label
            if len(dataset[0])==1 or flag==True:
                return currentLabel

            featInd = self.__chooseFeat(dataset)
            if featInd ==-1:
                return currentLabel
            tree = {}
            for featVal in self.__countFeatVal(dataset,featInd):
                splitDataset = self.__splitDatsetByAaixValue(dataset,featInd,featVal)
                tree[featVal] = self.__createTree(splitDataset)
            return {self.featLabels[featInd]:tree}

    #create tree
    def createTree(self):
        return self.__createTree(self.dataset)


    #count feature values
    def __countFeatVal(self,dataset,axis):
        featCount = {}
        for featVec in dataset:
            if featVec[axis] not in featCount:
                featCount[featVec[axis]]=0
            featCount[featVec[axis]]+=1
        return featCount.keys()


    #count the labels and return true if only one label left
    def __countLabels(self,dataset):
        labelCount = {}
        for featVec in dataset:
            label = featVec[-1]
            if label not in labelCount:
                labelCount[label]=0
            labelCount[label]+=1
        if len(labelCount)==1:
            return True,list(labelCount.keys())[0]
        else:
            return False,sorted(labelCount.items(),key=lambda x:x[1])[0][0]


