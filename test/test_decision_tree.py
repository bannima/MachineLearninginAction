#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: test_decision_tree.py
Description: 
Author: Barry Chow
Date: 2020/2/22 8:51 PM
Version: 0.1
"""

import pytest
from decision_tree.model.Tree import CART
import numpy as np
from numpy import eye

def test_cart():
    cart = CART()
    testMat = np.mat(eye(4))
    mat0,mat1 = cart.binarySplitDataset(eye,1,0.5)


