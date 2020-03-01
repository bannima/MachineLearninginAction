#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: criterion.py
Description: class for all kinds of criterion
Author: Barry Chow
Date: 2020/2/27 9:12 PM
Version: 0.1
"""
from abc import ABCMeta
from math import log
from numpy import *
from abc import abstractmethod
from utils.tools import calc_entropy,calc_conditional_entropy,calc_conditional_mse,calc_conditional_gini

class Criterion(metaclass=ABCMeta):

    @abstractmethod
    def criterion(self,dataset,labels,feat_ind,feat_val):
        """different feature selection criterion for different trees

        for example:

        information gain for ID3 model
        which is entropy(labels)-conditionalentropy(labels,feat_ind)

        information gain rate for C4.5

        gini for CART classification tree

        mean square error,MSE, for CARE regression tree

        Parameters
        ----------
        dataset:array_like, shape = (n_samples,n_features)

        labels: array_like, shape = (n_samples,1)

        feat_ind: feature index, int

        Returns
        -------
        information gain, float

        """
        pass

class ClassificationCriterion(Criterion):
    _criterion_type_ = "classification"
    def __init__(self):
        super(ClassificationCriterion,self).__init__()

class RegressionCriterion(Criterion):
    _criterion_type_ = "regression"

    def __init__(self):
        super(RegressionCriterion,self).__init__()


class Gini(ClassificationCriterion):

    def __init__(self):
        super(Gini,self).__init__()

    def criterion(self,dataset,labels,feat_ind,feat_val):
        """calculate the gini of the specific feature on given dataset

        Parameters
        ----------
        labels : array_like, shape = (n_samples,1), discrete features for classification problems

        Returns
        -------
        float, information gain
        """

        return 1-calc_conditional_gini(dataset, labels, feat_ind,feat_val)


class EntropyGain(ClassificationCriterion):

    def __init__(self):
        super(EntropyGain,self).__init__()

    def criterion(self,dataset,labels,feat_ind,feat_val=None):
        """calculate the information gain of the specific feature on given dataset

        Parameters
        ----------
        labels : array_like, shape = (n_samples,1), discrete features for classification problems

        Returns
        -------
        float, information gain
        """
        return calc_entropy(labels) - calc_conditional_entropy(dataset, labels, feat_ind,feat_val)


class MSE(RegressionCriterion):

    def __init__(self):
        super(MSE,self).__init__()

    def criterion(self,dataset,labels,feat_ind,feat_val):
        """calculate the mean square error of the specific feature on given dataset

        Parameters
        ----------
        labels : array_like, shape = (n_samples,1), discrete features for classification problems

        Returns
        -------
        float, mean square error
        """
        return 1-calc_conditional_mse(dataset,labels,feat_ind,feat_val)

_CRITERION= {'entropy':EntropyGain,'gini':Gini,'mse':MSE}

