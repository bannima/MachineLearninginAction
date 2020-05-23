#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: base.py
Description: 
Author: Barry Chow
Date: 2020/4/6 9:31 AM
Version: 0.1
"""
from abc import ABCMeta


class Sampling(metaclass=ABCMeta):
    def __init__(self):
        pass

    def sampling(self):
        '''
        returns one sample with sampling policy

        Returns
        -------

        '''
        raise NotImplementedError()
