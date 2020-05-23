#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: function.py
Description: 
Author: Barry Chow
Date: 2020/3/9 8:16 PM
Version: 0.1
"""
from numpy import exp

def sign(x,positive=1,middle=0,negative=-1):
    return positive if x >= middle else negative

def sigmod(x):
    return float(exp(x))/(1+exp(x))

def softmax(x):
    if x.ndim==1:
        return exp(x)/exp(x).sum()
    else:
        return exp(x)/exp(x).sum(axis=1,keepdims=True)
