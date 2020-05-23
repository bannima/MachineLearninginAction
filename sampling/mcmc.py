#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: mcmc.py
Description: implementation of Monte Carlo Markov Chain, in detail, which is Metropolis-Hasting and Gibbs sampling
Author: Barry Chow
Date: 2020/4/9 8:10 PM
Version: 0.1
"""
from abc import ABCMeta, abstractmethod

from .base import Sampling


class MCMC(metaclass=ABCMeta, Sampling):
    def __init__(self):
        super(MCMC, self).__init__()

    @abstractmethod
    def mcmc_sampling(self):
        pass


class M_H(MCMC):
    def __init__(self, num_samples=1e3, num_burn=1e4):
        super(M_H, self).__init__()

        assert num_samples > 0
        self.num_samples = num_samples

        # the examples before markov chain become stable
        assert num_burn > 0
        self.num_burn = num_burn

    def mcmc_sampling(self):
        for i in range(self.num_burn + self.num_samples):
            pass


class Gibbs(MCMC):
    def __init__(self):
        super(Gibbs, self).__init__()
