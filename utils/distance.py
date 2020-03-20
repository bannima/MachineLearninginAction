#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: distance.py
Description: 
Author: Barry Chow
Date: 2020/3/18 8:50 PM
Version: 0.1
"""
from abc import ABCMeta,abstractmethod
from numpy import array,sqrt
from math import pow

class Distance(metaclass=ABCMeta):
    '''
    abstract class for distance
    '''
    def __init__(self):
        pass

    def __call__(self,x,y):
        raise NotImplementedError()


class EuclideanDistance():
    def __init__(self):
        super(EuclideanDistance,self).__init__()

    def __call__(self,x,y):
        '''
        given vector x and y ,return the euclidean distance

        Parameters
        ----------
        x: array_like of numpy matrix

        y: array_like of numpy matrix

        Returns
        -------
        the defined euclidean distance

        '''
        assert len(x)==len(y)
        return sqrt(sum(pow(array(x)-array(y),2)))



DISTANCE = {
    'euclidean':EuclideanDistance
}