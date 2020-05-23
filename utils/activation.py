#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: activation.py
Description: activation functions
Author: Barry Chow
Date: 2020/3/30 4:15 PM
Version: 0.1
"""
from abc import ABCMeta, abstractmethod

from numpy import exp, where


class Activation(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, x):
        '''
        calculate the activation results in point x according to activation function

        Parameters
        ----------
        x: given point

        Returns
        -------
        the activation results in point x

        '''
        raise NotImplementedError()

    @abstractmethod
    def differential(self, x):
        '''
        return the differential in point x according to the activation function

        Parameters
        ----------
        x: given point

        Returns
        -------
        the differential in point x

        '''
        raise NotImplementedError()


class Sigmod(Activation):

    def __call__(self, x):
        return exp(x) / (1 + exp(x))

    def differential(self, x):
        return x * (1 - x)


class ReLU(Activation):

    def __call__(self, x):
        return where(x > 0, x, 0)

    def differential(self, x):
        return where(x > 0, 1, 0)


ACTIVATIONS = {
    'sigmod': Sigmod,
    'ReLU': ReLU
}
