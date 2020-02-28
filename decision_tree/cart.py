#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: cart.py
Description: 
Author: Barry Chow
Date: 2020/2/27 9:00 PM
Version: 0.1
"""
from abc import ABCMeta
from utils.base import BaseModel

class CART(BaseModel,metaclass=ABCMeta):
    def __init__(self,max_depth,max_leafs,min_splits):
        self.max_depth = max_depth
        self.max_leafs = max_leafs
        self.min_splits = min_splits

    def fit(self,X,y):
        pass


    def predict(self,X):
        pass





