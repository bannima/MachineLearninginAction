#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: normal.py
Description: implementation of normal distribution
Author: Barry Chow
Date: 2020/4/9 8:06 PM
Version: 0.1
"""
from .base import Sampling


class Normal(Sampling):
    def __init__(self, mean, standard_deviation):
        super(Normal, self).__init__()

        assert isinstance(mean, float)
        self.mean = mean

        assert isinstance(standard_deviation, float)
        self.scale = standard_deviation

    def sampling(self):
        pass
