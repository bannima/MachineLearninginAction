#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: _cart.py
Description: classification and regression tree implementation
Author: Barry Chow
Date: 2020/2/23 10:56 AM
Version: 0.1
"""
from numpy import *


'''
abandoned 
'''
class CART(object):
    def __init__(self):
        pass

    def _binarySplitDataset(self, dataset, feature, value):
        mat0 = dataset[nonzero(dataset[:,feature]>value)[0],:]
        mat1 = dataset[nonzero(dataset[:,feature]<=value)[0],:]
        return mat0,mat1

    def _regLeaf(self, dataSet):
        return mean(dataSet[:,-1])

    def _regErr(self, dataSet):
        return var(dataSet[:,-1])*shape(dataSet)[0]

    def _chooseBestSplit(self, dataSet, ops=(1, 4)):
        tolS = ops[0];tolN=ops[1]
        if len(set(dataSet[:,-1].T.tolist()[0]))==1:
            return None,self._regLeaf(dataSet)
        m,n=shape(dataSet)
        S = self._regErr(dataSet)
        bestS = inf;bestIndex = 0;bestValue=0
        for featIndex in range(n-1):
            for splitVal in set(dataSet[:,featIndex].T.A[0].tolist()):
                mat0,mat1 = self._binarySplitDataset(dataSet, featIndex, splitVal)
                if(shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN):
                    continue
                newS = self._regErr(mat0) + self._regErr(mat1)
                if newS<bestS:
                    bestIndex = featIndex
                    bestValue = splitVal
                    bestS = newS
        if(S-bestS)<tolS:
            return None,self._regLeaf(dataSet)
        mat0,mat1 = self._binarySplitDataset(dataSet, bestIndex, bestValue)
        if(shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN):
            return None,self._regLeaf(dataSet)

        return bestIndex,bestValue

    def createTree(self, dataSet, leafType=_regLeaf, errType=_regErr, ops=(1, 4)):
        feat,val = self._chooseBestSplit(dataSet, ops)
        if feat==None:
            return val
        retTree = {}
        retTree['spInd']=feat
        retTree['spVal']=val
        lSet,rSet = self._binarySplitDataset(dataSet, feat, val)
        retTree['left']=self.createTree(lSet,leafType,errType,ops)
        retTree['right']=self.createTree(rSet,leafType,errType,ops)
        return retTree

    def _isTree(self,obj):
        return(type(obj).__name__=='dict')

    def _getMean(self,tree):
        if self._isTree(tree['right']):
            tree['right']=self._getMean(tree['right'])
        if self._isTree(tree['left']):
            tree['left']=self._getMean(tree['left'])
        return (tree['left']+tree['right'])/2.0

    def prune(self,tree,testData):
        if shape(testData)[0]==0:
            return self._getMean(tree)
        if self._isTree(tree['right']) or self._isTree(tree['left']):
            lSet,rSet = self._binarySplitDataset(testData,tree['spInd'],tree['spVal'])
        if self._isTree(tree['left']):
            tree['left']=self.prune(tree['left'],lSet)
        if self._isTree(tree['right']):
            tree['right']=self.prune(tree['right'],rSet)
        if not self._isTree(tree['left']) and not self._isTree(tree['right']):
            lSet,rSet = self._binarySplitDataset(testData,tree['spInd'],tree['spVal'])
            errorNoMessage = sum(power(lSet[:,-1]-tree['left'],2))+\
                sum(power(rSet[:,-1]-tree['right'],2))
            treeMean = (tree['left']+tree['right'])/2.0
            errorMessage = sum(power(testData[:,-1]-treeMean,2))
            if errorMessage<errorNoMessage:
                print("merging")
                return treeMean
            else:
                return tree
        else:
            return tree



