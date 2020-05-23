#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: page_rank.py
Description: 
Author: Barry Chow
Date: 2020/4/6 4:17 PM
Version: 0.1
"""
from copy import copy

from numpy import shape, ones, linalg


class PageRank(object):
    def __init__(self, M, R, damping=0.5, max_iter=100, threshold=1e-5):
        '''

        Parameters
        ----------
        M: stochastic matrix
        damping: damping factor
        R: initial distribution
        max_iter: the maximum iteration times
        threshold: stop calculation when the page rank value change is less than the threshold
        '''
        # assert isinstance(M,mat)
        self.M = M

        # assert isinstance(R,mat)
        self.R = R

        assert isinstance(damping, float)
        assert damping > 0
        self.d = damping

        assert isinstance(max_iter, int)
        assert max_iter > 0
        self.max_iter = max_iter

        self.threshold = threshold

    def fit(self):
        '''
        calculate the page rank value using iterative algorithm.

        Returns
        -------
        final page rank value: array_like
        '''
        n = shape(self.M)[0]
        assert n == len(self.R)

        R = copy(self.R)
        for iter in range(self.max_iter):
            R_next = (self.d * self.M) * R + ((1 - self.d) / n) * ones(n).reshape(-1, 1)
            if self._calc_diff(R, R_next) < self.threshold:
                break
            R = R_next
            print(R)

        return R

    def _calc_diff(self, R1, R2):
        '''

        Parameters
        ----------

        Returns
        -------
        '''
        return linalg.norm(R1 - R2)
