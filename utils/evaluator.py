#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: evaluator.py
Description: 
Author: Barry Chow
Date: 2020/3/3 6:40 PM
Version: 0.1
"""
from numpy import array,mean

def accuracy_score(preds,labels):
    """
    calculate the classification accuracy score

    """
    assert len(preds)==len(labels)
    return (array(preds)==array(labels)).sum()/float(len(labels))


def mean_square_error(preds,labels):
    '''
    calculate the mean square error for array

    '''
    assert len(preds)==len(labels)
    return mean((array(preds)-array(labels))**2)
