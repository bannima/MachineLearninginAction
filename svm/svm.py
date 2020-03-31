#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: svm.py
Description: 
Author: Barry Chow
Date: 2020/3/10 7:40 PM
Version: 0.1
"""
from numpy import shape, zeros, array, mat, multiply
from base import Classifier
from utils import KERNEL, Kernel
from utils import sign


class BinarySVM(Classifier):
    '''
    Support Vector Machines for Binary Classification Tasks.

    '''

    def __init__(self, C=0.5, kernel='linear', max_iter=100, toler=1e-3):
        '''

        :param C: penalty confficient
        :param kernel: kernel function types
        :param max_iter: maximum iteration times
        :param toler: allowable tolerance
        '''
        self.C = C
        if kernel in KERNEL:
            self.kernel = KERNEL[kernel]()
        elif isinstance(kernel, Kernel):
            self.kernel = kernel
        else:
            raise ValueError("wrong kernel type %s".format(str(kernel)))
        self.max_iter = max_iter
        self.toler = toler
        super(BinarySVM, self).__init__()

    def fit(self, X, y):
        n_samples, n_features = shape(X)
        assert n_samples == len(y)
        self.y = mat(y)
        self.alphas = zeros((n_samples, 1))
        self.b = 0

        # kernel products
        # self.K = mat(X)*mat(X).T
        self.K = self._kernel_product(X)

        self.X = X

        # maximum iteration times
        for iter in range(self.max_iter):

            for i in range(n_samples):

                # update E once the aphas has changed
                self.E = array([self._E(i) for i in range(n_samples)])

                # choose first alpha
                if (self.alphas[i] < self.C and (self.y[i] * self.E[i] < 0) or
                        self.alphas[i] > 0 and (self.y[i] * self.E[i] > 0)):

                    # choose the second alpha
                    j = abs(self.E[i] - self.E).argmax()
                    # calc selected alphas
                    new_alpha_i, new_alpha_j = self._update_alpha(i, j)

                    # check update
                    if abs(self.alphas[j] - new_alpha_j) < self.toler:
                        # alpha j not moving enough, next i
                        continue
                    # update new alpha i and j
                    self.alphas[i] = new_alpha_i
                    self.alphas[j] = new_alpha_j

                    # update b
                    self._update_b(i, j, new_alpha_i, new_alpha_j)

    def _kernel_product(self, X):
        '''
        generate kernel product for vector in X
        '''
        n_samples = shape(X)[0]
        K = mat(zeros((n_samples, n_samples)))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])
        return K

    def _update_b(self, i, j, new_alpha_i, new_alpha_j):
        '''
        update b
        '''
        new_b_i = -1 * self.E[i] - self.y[i] * self.K[i, i] * (new_alpha_i - self.alphas[i]) - \
                  self.y[j] * self.K[j, i] * (new_alpha_j - self.alphas[j]) + self.b
        new_b_j = -1 * self.E[j] - self.y[i] * self.K[i, j] * (new_alpha_i - self.alphas[i]) - \
                  self.y[j] * self.K[j, j] * (new_alpha_j - self.alphas[j]) + self.b
        if (0 < self.alphas[i]) and (self.alphas[i] < self.C):
            self.b = new_b_i
        elif (0 < self.alphas[j]) and (self.alphas[j] < self.C):
            self.b = new_b_j
        else:
            self.b = (new_b_i + new_b_j) / 2.0

    def _update_alpha(self, i, j):
        '''
        return the new cliped alpha according to the select (i,j)
        '''
        eta = float(self.K[i, i] + self.K[j, j] - 2 * self.K[i, j])
        unc_alpha_j = self.alphas[j] + self.y[j] * (self.E[i] - self.E[j]) / eta
        if self.y[i] == self.y[j]:
            L = max(0, self.alphas[j] + self.alphas[i] - self.C)
            H = min(self.C, self.alphas[j] + self.alphas[i])
        else:
            L = max(0, self.alphas[j] - self.alphas[i])
            H = min(self.C, self.C + self.alphas[j] - self.alphas[i])

        new_alpha_j = self._clip(unc_alpha_j, L, H)
        new_alpha_i = self.alphas[i] + float(self.y[i] * self.y[j]) * (self.alphas[j] - new_alpha_j)
        return new_alpha_i, new_alpha_j

    def _clip(self, aj, L, H):
        if aj < L:
            return L
        elif aj > H:
            return H
        else:
            return aj

    def _E(self, i):
        '''
        implementation of function Ei
        '''
        return self._gX(i) - self.y[i]

    def _gX(self, i):
        '''
        implementation of function gX(i)
        '''
        return float(self.K[i] * multiply(self.y, self.alphas)) + self.b

    def predict(self, X):
        '''
        given samples return the predict results
        '''
        return array([self._predict_sample(xi) for xi in X])

    def _predict_sample(self, sample):
        '''
        predict sample instance
        '''
        K_sample = array([self.kernel(mat(sample), xi) for xi in self.X])
        return sign(float(mat(K_sample) * multiply(self.y, self.alphas)) + self.b)
