#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: preprocessing.py
Description: 
Author: Barry Chow
Date: 2020/3/4 7:41 PM
Version: 0.1
"""
from numpy import *


def one_hot(X):
    '''
    transform input array to one hot vector

    Parameters:
    -----------
    X: array_like

    Returns:
    --------
    one hot vector matrix

    '''
    values = array(X)
    n_values = max(values) + 1
    return eye(n_values)[values]
