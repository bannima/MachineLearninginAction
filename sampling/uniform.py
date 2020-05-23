#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: uniform.py
Description: implementation of uniform sampling
Author: Barry Chow
Date: 2020/4/6 9:30 AM
Version: 0.1
"""
from .base import Sampling


class Uniform(Sampling):
    '''
    implement uniform sampling using Linear congruential generator
    '''

    def __init__(self):
        super(Uniform, self).__init__()

    def sampling(self):
        pass
