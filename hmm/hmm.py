#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: hmm.py
Description: hmm Model
Author: Barry Chow
Date: 2020/2/3 11:24 PM
Version: 0.1
"""
from numpy import zeros,array,argmax

class HiddenMarkovModel(object):
    def __init__(self, A, B, PI, O):
        '''
        init hmm model

        Parameters
        ----------
        A: matrix
        B: matrix
        PI: matrix
        O: matrix, observation
        '''

        # laod init params
        self.A = A
        self.B = B
        self.PI = PI
        self.O = O

        # num of hidden states
        self.N = self.A.shape[0]

        # num of steps
        self.T = len(O)

    def forward_backword(self):
        '''
        calc probability using forward and backword pass

        Returns
        -------
        probability

        '''
        self.alpha = zeros((self.N, self.T))
        for i in range(self.T):
            for j in range(self.N):
                # init case
                if i == 0:
                    self.alpha[j, i] = self.PI[j] * self.B[j, self.O[i]]
                    continue
                # else
                self.alpha[j, i] = (self.alpha[:, i - 1].transpose() * self.A[:, j]) * self.B[j, self.O[i]]
        return sum(self.alpha[:, -1])

    def viterbi(self):
        '''
        viterbi algorithm for decode the hidden state path with max probability.

        Returns
        -------
        prob: maximum probability

        path: hidden state path

        '''
        self.delta = zeros((self.N, self.T))
        self.path = zeros((self.N, self.T))
        for i in range(self.T):
            for j in range(self.N):
                if i == 0:
                    self.delta[j, i] = self.PI[j] * self.B[j, self.O[i]]
                    continue

                probs = (self.delta[:, i - 1].T * self.A[:, j].A.T).tolist()[0]
                self.delta[j, i] = max(probs) * self.B[j, self.O[i]]
                self.path[j, i - 1] = argmax(probs) + 1

        # traceback for the maximum path
        max_path = []
        current = argmax(self.delta[:, -1]) + 1
        max_path.append(current)
        for i in reversed(range(self.T - 1)):
            current = int(self.path[current - 1, i])
            max_path.append(current)

        max_path.reverse()
        #return max(self.delta[:, -1]), reversed(max_path)
        return max(self.delta[:, -1]), max_path
