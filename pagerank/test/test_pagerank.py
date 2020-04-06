#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: test_pagerank.py
Description: 
Author: Barry Chow
Date: 2020/4/6 4:18 PM
Version: 0.1
"""
from pagerank import PageRank
from numpy import mat,array,mean

class Test_PageRank(object):

    def test_pagerank(self):
        M = mat([[0,1/2,0,0],
                 [1/3,0,0,1/2],
                 [1/3,0,1,1/2],
                 [1/3,1/2,0,0,]])
        R = array([1/4,1/4,1/4,1/4]).reshape(-1,1)
        pr = PageRank(M,R,damping=0.8,max_iter=100)
        R = pr.fit()
        R_true = array([15/148,19/148,95/148,19/148]).reshape(-1,1)
        assert mean(R-R_true)<1e-3
