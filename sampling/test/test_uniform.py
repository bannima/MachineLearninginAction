#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: test_uniform.py
Description: 
Author: Barry Chow
Date: 2020/4/6 9:37 AM
Version: 0.1
"""

import matplotlib.pyplot as plt
from numpy import random, exp, sqrt, pi, arange
from scipy import stats


class Test_Uniform(object):

    def test_standord_uniform(self):
        mean = 0
        scale = 1
        num_samples = 100000
        x = arange(-10, 10, 0.1)

        norm_samples = random.normal(mean, scale, num_samples)
        # random.standard_normal(10000)
        # print(norm_samples)
        # plt.hist(norm_samples,bins=100,histtype='step')

        # normal function picture
        # y = norm_function(x,mean,scale)
        y = stats.norm.pdf(x)
        plt.plot(x, y)
        plt.show()


def norm_function(x, mu, sigma):
    pdf = exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * sqrt(2 * pi))
    return pdf
