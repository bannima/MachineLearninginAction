#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: cart.py
Description: classification and regression tree implementation
Author: Barry Chow
Date: 2020/2/23 10:56 AM
Version: 0.1
"""
from numpy import *

class CART(object):
    def __init__(self):
        pass

    def binarySplitDataset(self,dataset,feature,value):
        mat0 = dataset[nonzero(dataset[:,feature]>value)[0],:]
        mat1 = dataset[nonzero(dataset[:,feature]<=value)[0],:]
        return mat0,mat1

    def regLeaf(self,dataSet):
        return mean(dataSet[:,-1])

    def regErr(self,dataSet):
        return var(dataSet[:,-1])*shape(dataSet)[0]

    def chooseBestSplit(self,dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
        tolS = ops[0];tolN=ops[1]
        if len(set(dataSet[:,-1].T.tolist()[0]))==1:
            return None,self.regLeaf(dataSet)
        m,n=shape(dataSet)
        S = self.regErr(dataSet)
        bestS = inf;bestIndex = 0;bestValue=0
        for featIndex in range(n-1):
            for splitVal in set(dataSet[:,featIndex].T.A[0].tolist()):
                mat0,mat1 = self.binarySplitDataset(dataSet,featIndex,splitVal)
                if(shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN):
                    continue
                newS = self.regErr(mat0)+self.regErr(mat1)
                if newS<bestS:
                    bestIndex = featIndex
                    bestValue = splitVal
                    bestS = newS
        if(S-bestS)<tolS:
            return None,self.regLeaf(dataSet)
        mat0,mat1 = self.binarySplitDataset(dataSet,bestIndex,bestValue)
        if(shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN):
            return None,self.regLeaf(dataSet)

        return bestIndex,bestValue

    def createTree(self,dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
        feat,val = self.chooseBestSplit(dataSet,leafType,errType,ops)
        if feat==None:
            return val
        retTree = {}
        retTree['spInd']=feat
        retTree['spVal']=val
        lSet,rSet = self.binarySplitDataset(dataSet,feat,val)
        retTree['left']=self.createTree(lSet,leafType,errType,ops)
        retTree['right']=self.createTree(rSet,leafType,errType,ops)
        return retTree

