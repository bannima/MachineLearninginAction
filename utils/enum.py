#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: enum.py
Description: 
Author: Barry Chow
Date: 2020/3/1 4:40 PM
Version: 0.1
"""
from enum import Enum, unique


@unique
class SYMBOL(Enum):
    '''
    character symbol, less than and no less than
    '''
    LT = '<'
    NLT = '>='
