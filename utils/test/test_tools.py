#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: test_tools.py
Description: unittest for tools
Author: Barry Chow
Date: 2020/2/29 8:39 PM
Version: 0.1
"""
from numpy import mat
from sklearn.datasets import load_iris

from ..tools import calc_entropy

iris = load_iris()


class TestTools(object):
    def test_calc_entropy(self):
        entropy = calc_entropy((mat(iris.target).T))
        assert entropy == 1.584962500721156
