#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: base.py
Description: 
Author: Barry Chow
Date: 2020/3/28 11:39 PM
Version: 0.1
"""
from abc import ABCMeta


class Layer(metaclass=ABCMeta):
    def __init__(self, dimension, bias=None):
        assert isinstance(dimension, int)
        self.dimension = dimension


class Linear(Layer):
    '''
    full connected layer
    '''
