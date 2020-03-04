#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: test_preprocessing.py
Description: 
Author: Barry Chow
Date: 2020/3/4 7:52 PM
Version: 0.1
"""
from utils import one_hot
from numpy import *

def test_preprocessing():
    a = [0,1,2]
    x = one_hot(a)
    assert (x==eye(3)).all()