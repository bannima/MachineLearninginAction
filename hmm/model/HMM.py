#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: hmm.py
Description: hmm Model
Author: Barry Chow
Date: 2020/2/3 11:24 PM
Version: 0.1
"""
import numpy as np

class HiddenMarkovModel(object):
    #init hmm model
    def __init__(self,A,B,PI,O):
        #params check(ignored)

        #laod init params
        self.A = A
        self.B = B
        self.PI = PI
        self.O = O

        #num of hidden states
        self.N = self.A.shape[0]

        #num of steps
        self.T = len(O)

    #calc probability
    def forward_backword(self):
        self.alpha = np.zeros((self.N,self.T))
        for i in range(self.T):
            for j in range(self.N):
                # init case
                if i == 0:
                    self.alpha[j, i] = self.PI[j] * self.B[j, self.O[i]]
                    continue
                #else
                self.alpha[j,i] = (self.alpha[:,i-1].transpose()*self.A[:,j])*self.B[j,self.O[i]]
        return sum(self.alpha[:,-1])

    #viterbi
    def viterbi(self):
        self.delta = np.zeros((self.N,self.T))
        self.path = np.zeros((self.N,self.T))
        for i in range(self.T):
            for j in range(self.N):
                if i==0:
                    self.delta[j,i] = self.PI[j]*self.B[j,self.O[i]]
                    continue

                probs = (self.delta[:,i-1].T*self.A[:,j].A.T).tolist()[0]
                self.delta[j,i] = max(probs)*self.B[j,self.O[i]]
                self.path[j,i-1] = np.argmax(probs)+1

        #traceback for the maximum path
        max_path = []
        current = np.argmax(self.delta[:, -1])+1
        max_path.append(current)
        for i in reversed(range(self.T-1)):
            current = int(self.path[current-1,i])
            max_path.append(current)

        return max(self.delta[:,-1]),reversed(max_path)

