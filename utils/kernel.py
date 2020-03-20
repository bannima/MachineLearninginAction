#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: kernel.py
Description: kernel functions
Author: Barry Chow
Date: 2020/3/17 7:45 PM
Version: 0.1
"""
from abc import ABCMeta, abstractmethod


class Kernel(metaclass=ABCMeta):
    '''
    abstract class for kernel functions
    '''

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, x, y):
        '''
        given vector x and y ,return the specific kernel product

        Parameters
        ----------
        x: array_like

        y: array_like

        Returns
        -------
        the defined kernel product: float

        '''
        pass


class LinearKernel(Kernel):

    def __init__(self, b=0):
        self.b = b
        super(LinearKernel, self).__init__()

    def __call__(self, x, y):
        '''
        given vector x and y ,return the linear kernel product

        Parameters
        ----------
        x: array_like of numpy matrix

        y: array_like of numpy matrix

        Returns
        -------
        the defined kernel product: float

        '''
        return float(x * y.T) + self.b


KERNEL = {
    'linear': LinearKernel
}
