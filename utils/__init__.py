#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: __init__.py.py
Description: 
Author: Barry Chow
Date: 2020/2/22 7:37 PM
Version: 0.1
"""
from .activation import ACTIVATIONS, Activation
from .criterion import CRITERION
from .distance import DISTANCE
from .evaluator import accuracy_score, mean_square_error
from .function import sign, sigmod, softmax
from .kernel import KERNEL, Kernel
from .loss import LOSS_FUNCTIONS
from .preprocessing import one_hot, split_train_test

__all__ = [
    'LOSS_FUNCTIONS',
    'accuracy_score',
    'mean_square_error',
    'one_hot',
    'CRITERION',
    'KERNEL',
    'Kernel',
    'sign',
    'sigmod',
    'softmax',
    'DISTANCE',
    'Activation',
    'ACTIVATIONS'
]
