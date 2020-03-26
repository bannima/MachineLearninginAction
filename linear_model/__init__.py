#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: __init__.py.py
Description: 
Author: Barry Chow
Date: 2020/3/9 6:22 PM
Version: 0.1
"""
from .perceptron import Perceptron
from .logistic_regression import LogisticRegression

__all__ = [
    'Perceptron',
    'LogisticRegression'
]
