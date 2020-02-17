#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: adaboost.py
Description:
Author: Barry Chow
Date: 2020/2/17 3:26 PM
Version: 0.1
"""

import numpy as np
from numpy import inf
from numpy import matrix,mat,array,multiply
from numpy import ones,shape,zeros
from math import log
from numpy import exp,sign

class AdaBoost:
    def __init__(self):
        pass

    #return the classification results, given the dimen and thresVal
    def __stumpClassify(self,dataMatrix,dimen,threshVal,threshIneq):
        retArray = ones((shape(dataMatrix)[0],1))
        #less than
        if threshIneq=='lt':
            retArray[dataMatrix[:,dimen]>=threshVal]=-1.0
        else:
            retArray[dataMatrix[:,dimen]<=threshVal]=-1.0
        return retArray

    #build each decision stump according to weights D
    def __buildStump(self,dataArr,classLabels,D,numSteps = 10):
        dataArr = mat(dataArr)
        #num of features, num of simples
        numSimps,numFeats = shape(dataArr)
        minError = inf; bestStump = {}
        bestStumpResult = mat(ones((numSimps, 1)))
        for featInd in range(numFeats):
            minFeatVal = dataArr[:,featInd].min();maxFeatVal = dataArr[:,featInd].max()
            stepSize = (maxFeatVal-minFeatVal)/numSteps
            for stepInd in range(-1,int(numSteps)+1):
                featVal = minFeatVal+float(stepInd)*stepSize
                #less than or greater than
                for threIneq in ['lt','gt']:
                    predVals = self.__stumpClassify(dataArr,featInd,featVal,threIneq)
                    #calc error rate
                    errArr = mat(ones((numSimps,1)))
                    errArr[predVals==mat(classLabels).T]=0
                    error = (D.T*errArr)[0,0]

                    if error< minError:
                        bestStump['dimen']= featInd
                        bestStump['featVal']=featVal
                        bestStump['thresIneq']=threIneq
                        bestStumpResult = predVals.copy()
                        minError =  error

        return minError,bestStumpResult,bestStump

    #train adaboost
    def adaboostTrain(self,dataArr,classLabels,trainSteps=10):
        m,n = shape(dataArr)
        #init weights
        D = mat(ones((m,1))/m)
        #weak classifier list
        weakClassList = []

        #aggretrate
        aggClassResult = mat(zeros((m,1)))

        for i in range(trainSteps):
            #choose the best decision stump
            minError, bestStumpResult, bestStump = self.__buildStump(dataArr,classLabels,D)
            alpha = float(0.5*log((1.0-minError)/max(minError,1e-16)))
            bestStump['alpha'] = alpha
            weakClassList.append(bestStump)

            #adjust weight D
            expon = multiply(-1.0*alpha*mat(classLabels).T,bestStumpResult)
            D = multiply(D,exp(expon))
            D = D/D.sum()

            aggClassResult+=alpha*bestStumpResult
            aggErrors = multiply(sign(aggClassResult)!=mat(classLabels).T,ones((m,1)))
            aggErrorRate = aggErrors.sum()/m

            print("\n=====")
            print("D: ",D.T)
            print("agg preds: ",aggClassResult.T)
            print("true labels: ",classLabels)
            print("total error rate: ",aggErrorRate)

            if aggErrorRate==0.0:
                break
        self.classifierArray = weakClassList
        return weakClassList

    def adaClassify(self,dataToClass):
        dataMatrix = mat(dataToClass)
        m = shape(dataMatrix)[0]
        aggClassResult = mat(zeros((m,1)))
        for i in range(len(self.classifierArray)):
            predVals = self.__stumpClassify(dataMatrix,\
                                self.classifierArray[i]['dimen'],\
                                self.classifierArray[i]['featVal'],\
                                self.classifierArray[i]['thresIneq'])
            aggClassResult+=self.classifierArray[i]['alpha']*predVals
        return sign(aggClassResult)


    def pred(self,dataToClass,classLabels):
        predClass = self.adaClassify(dataToClass)
        errArr = mat(ones((shape(dataToClass)[0],1)))
        return float(errArr[predClass!=mat(classLabels).T].sum())/shape(dataToClass)[0]















