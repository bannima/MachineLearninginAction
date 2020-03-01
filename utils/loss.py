#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: loss.py
Description: 
Author: Barry Chow
Date: 2020/2/22 7:37 PM
Version: 0.1
"""
from abc import ABCMeta,abstractmethod

class BaseLoss(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def negative_gradient(self):
        ''''
        calc the negative gredient
        '''
        pass

class MeanSquareLoss(BaseLoss):
    def __init__(self):
        super(MeanSquareLoss,self).__init__()

    def negative_gradient(self):
        pass
