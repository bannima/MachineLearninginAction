#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: test_hmm.py
Description: A simple implementation of Hidden Markov Model
Author: Barry Chow
Date: 2020/2/3 11:24 PM
Version: 0.1
"""
import numpy as np

from hmm import HiddenMarkovModel


class Test_HMM(object):
    def test_hmm(self):
        '''A = np.matrix([[0,1,0,0],
                           [0.4,0,0.6,0],
                           [0,0.4,0,0.6],
                           [0,0,0.5,0.5]])
            B = np.matrix([[0.5,0.5],
                           [0.3,0.7],
                           [0.6,0.4],
                           [0.8,0.2]])
            PI = np.matrix([0.25,0.25,0.25,0.25]).transpose()

            #observation result
            O = [0,0,1,1,0]'''

        A = np.matrix([[0.5, 0.2, 0.3],
                       [0.3, 0.5, 0.2],
                       [0.2, 0.3, 0.5]])

        B = np.matrix([[0.5, 0.5],
                       [0.4, 0.6],
                       [0.7, 0.3]])
        PI = np.matrix([0.2, 0.4, 0.4]).transpose()
        O = [0, 1, 0]

        hmm = HiddenMarkovModel(A, B, PI, O)
        prob, path = hmm.viterbi()

        print("The probability for Observation States", str(O), " is ", hmm.forward_backword())
        assert hmm.forward_backword() == 0.130218

        print("The max probs for Observation States", str(O), " is ", prob, " and the hidden state path is ",
              '-'.join(['%s' % id for id in path]))
        assert prob == 0.014699999999999998
        assert path == [3, 3, 3]
