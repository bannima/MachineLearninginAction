#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: loss.py
Description: 
Author: Barry Chow
Date: 2020/2/22 7:37 PM
Version: 0.1
"""
from abc import ABCMeta, abstractmethod
from numpy import *


class BaseLoss(metaclass=ABCMeta):
    '''
    abstract loss function class

    '''

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, preds, labels):
        '''
        given true labels and predictions ,return the calculated the loss

        Parameters
        ----------
        pred: array_like

        labels: array_like

        Returns
        -------
        the defined loss: array_like of float

        '''

    @abstractmethod
    def negative_gradient(self, preds, labels):
        ''''
        calculate the negative gredient

        Parameters
        ----------
        pred: array_like

        labels: array_like

        Returns
        -------
        the defined negative gradient

        '''


class MeanSquareLoss(BaseLoss):
    '''
    mean square loss function

    formula: loss = (label-pred)**2
             negative gradient = 2*(label-pred)
    '''

    def __init__(self):
        super(MeanSquareLoss, self).__init__()

    def __call__(self, preds, labels):
        #return (array(labels) - array(preds)) ** 2
        return (labels-preds)**2

    def negative_gradient(self, preds, labels):
        #return 2 * (array(labels) - array(preds))
        return 2*(labels-preds)


class ExponetialLoss(BaseLoss):
    '''
    exponetial loss function

    formula: loss = exp(-label*pred)
             negative gradient = -label*exp(-label*pred)
    '''

    def __init__(self):
        super(ExponetialLoss, self).__init__()

    def __call__(self, preds, labels):
        return exp(-1 * array(labels) * array(preds))

    def negative_gradient(self, preds, labels):
        return -1 * array(labels) * exp(-1 * array(labels) * array(preds))


LOSS_FUNCTIONS = {
    'mse': MeanSquareLoss,
    'el': ExponetialLoss
}
