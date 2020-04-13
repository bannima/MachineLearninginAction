#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: test_mcmc.py
Description: 
Author: Barry Chow
Date: 2020/4/9 8:15 PM
Version: 0.1
"""
from numpy import random
from scipy import stats
from random import uniform
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal


class Test_MCMC(object):

    def test_M_H(self):
        '''
        implementation of M-H sampling examples in https://www.cnblogs.com/pinard/p/6638955.html

        Object probability distribution: X-Norm(3,2)

        Transform function, P(i,j): the probablity of point j in J-Norm(i,1)
        '''

        n1 = 10000
        n2 = 100000
        x = random.uniform(0, 1)
        samples=  []

        for i in range(n1+n2):
            #generate a sample using conditional probability
            x_next = random.normal(x,1,1)[0]

            u = uniform(0,1)
            alpha = min(normal_prob(x_next,10,2)/normal_prob(x,10,2),1)
            if u<alpha:
                x = x_next

            if i>n1:
                samples.append(x)

        show_distribution(samples)


    def test_gibbs(self):
        n1 = 1000
        n2 = 1000000
        x1 = uniform(0,1)
        x2 = uniform(0,1)
        x1_samples = []
        x2_samples = []
        for i in range(n1+n2):
            x2 = random.normal(-1+(1/(x1-5)),0.75*4,1)[0]
            x1 = random.normal(5+0.5/(2*(x2+1)),0.75,1)[0]

            if too_large(x1) or too_large(x2):
                continue

            if i>n1:
                x1_samples.append(x1)
                x2_samples.append(x2)

        show_distribution(x1_samples)
        show_distribution(x2_samples)

        fig = plt.figure()
        samplesource = multivariate_normal(mean=[5, -1], cov=[[1, 1], [1, 4]])
        z_samples =[samplesource.pdf([x1,x2]) for x1,x2 in zip(x1_samples,x2_samples)]
        ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
        ax.scatter(x1_samples, x2_samples, z_samples, marker='o')
        plt.show()


def too_large(x):
    return x>10 or x<-10







def show_distribution(samples):
    plt.hist(samples,bins=100,histtype='step',normed=1)
    plt.show()




def normal_prob(x,mean,scale):
    '''
    the probability of point x in X-Norm(mean,scale)

    Parameters
    ----------
    x: point
    mean: mean value of normal distribution
    scale: standard devitation of normal distribution

    Returns
    -------
    probability

    '''
    return stats.norm.pdf(x,mean,scale)

