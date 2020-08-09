#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: crf.py
Description: implementation of linear chain conditional random field
Author: Barry Chow
Date: 2020/5/31 10:30 PM
Version: 0.1
"""
from itertools import product

import numpy as np

from base import Classifier


class LinearChainCRF(Classifier):
    '''
    implementation of linear chain CRF model
    '''

    def __init__(self, feature_builder, epochs=100, learning_rate=1e-3):
        self.feature_builder = feature_builder
        self.epochs = epochs
        self.lr = learning_rate

    def _init_params(self, X, y):

        self.tag_set = list(self.feature_builder.tag_set)

        self.tag_dict = {}
        for tag in self.tag_set:
            self.tag_dict[len(self.tag_dict)] = tag

        self.n_samples, self.feature_nums = len(X), self.feature_builder.feature_nums

        # initialize weight vectors
        self.w = np.zeros(self.feature_nums)

        # margin distribution of x
        self.P_x = {}
        # joint distribution of x and y
        self.P_xy = {}

        # calculate the margin and joint distribution
        for sentence, tag in zip(X, y):
            # count x and y frequency
            if self.P_x.get(sentence) is None:
                self.P_x[sentence] = 1
            else:
                self.P_x[sentence] += 1

            if self.P_xy.get(sentence + tag) is None:
                self.P_xy[sentence + tag] = 1
            else:
                self.P_xy[sentence + tag] += 1

        for key, value in self.P_xy.items():
            self.P_xy[key] = 1.0 * self.P_xy[key] / self.n_samples
        for key, value in self.P_x.items():
            self.P_x[key] = 1.0 * self.P_x[key] / self.n_samples

    def fit(self, X, y):
        '''

        Parameters
        ----------
        X: input sequence dataset
        y: tags or hidden state with corresponding feature vectors
        feature_vectors: the specific feature function id for each input sequence

        Returns
        -------

        '''
        self._init_params(X, y)
        self.feature_vectors = {}

        for epoch in range(self.epochs):
            for sentence, tag in zip(X, y):
                delta_w = np.zeros(self.feature_nums)
                # match the specific feature function list for given sentence and tag
                # calculate in the first time and save in feature_vectors
                if (sentence + tag) in self.feature_vectors:
                    feature_vector = self.feature_vectors[sentence + tag]
                else:
                    feature_vector = self.feature_builder.match(sentence, tag)

                # feature_vector = self.feature_builder.match(sentence, tag)

                if len(feature_vector) == 0:
                    continue

                # calculate the delta w
                P_y_of_x = self.P_y_of_x(sentence, tag)

                for feature_function_num in feature_vector:
                    delta_w[feature_function_num] += self.P_xy[sentence + tag]
                    delta_w[feature_function_num] -= self.P_x[sentence] * P_y_of_x

                self.w += self.lr * delta_w

            if (epoch % 10 == 0):
                print("training epoch: " + str(epoch))

    def P_y_of_x(self, sentence, tag):
        '''
        calcualate the probability of  P(y|x), where sentence equals x and y means tag

        Parameters
        ----------
        sentence
        tag

        Returns
        -------

        '''

        return self._exp_prob_of_x_y(sentence, tag) / self._exp_prob_of_x(sentence)

    def _exp_prob_of_x_y(self, sentence, tag):
        '''
        the prob of tag(y) given sentence(x)

        Parameters
        ----------
        sentence
        tag

        Returns
        -------

        '''

        tmp_w = 0.0
        match_functions = self.feature_builder.match(sentence, tag)
        for feature_function_num in match_functions:
            tmp_w += self.w[feature_function_num]
        return np.exp(tmp_w)

    def _exp_prob_of_x(self, sentence):
        '''
        the sum prob of all tag(y) given sentence(x)
        using itertools.product to produce all possible tags

        Parameters
        ----------
        sentence

        Returns
        -------

        '''
        n = len(sentence)
        sum_w = 0.0
        for tag in product(self.tag_set, repeat=n):
            tag = ''.join(tag)
            sum_w += self._exp_prob_of_x_y(sentence, tag)
        return sum_w

    def predict(self, X):
        '''
        predict sentence dataset using Viterbi Algorithm

        Parameters
        ----------
        X

        Returns
        -------

        '''
        predicts = []
        for sentence in X:
            predicts.append(self._predict_sentence(sentence))
        return predicts

    def _transfer_prob(self, sentence, prior, current, index):
        '''
        calc the transfer probability from prior state to current state in the position index for senetence

        Parameters
        ----------
        sentence
        prior
        current
        index

        Returns
        -------

        '''

        match_feature_vectors = self.feature_builder.match_with_index(sentence, prior, current, index)
        return sum([self.w[feature_num] for feature_num in match_feature_vectors])

    def _predict_sentence(self, sentence):
        '''
        predict single sentence using Viterbi Algorithm

        Parameters
        ----------
        sentence

        Returns
        -------

        '''
        n, m = len(sentence), len(self.tag_set)
        # y0=start

        self.delta = np.zeros((m, n))
        self.path = np.zeros((m, n))
        for i in range(n):
            for j in range(m):
                # start position
                if i == 0:
                    self.delta[j][i] = self._transfer_prob(sentence, '#', self.tag_dict[j], i)
                    continue
                probs = []
                for k in range(m):
                    probs.append(
                        self._transfer_prob(sentence, self.tag_dict[k], self.tag_dict[j], i) + self.delta[k][i - 1])
                self.delta[j][i] = max(probs)
                self.path[j][i - 1] = np.argmax(probs)

        # traceback for maximum probability path
        max_path = []
        current = np.argmax(self.delta[:, -1])
        max_path.append(current)
        for i in reversed(range(n - 1)):
            current = int(self.path[current, i])
            max_path.append(current)

        max_path.reverse()
        return ''.join([self.tag_dict[path] for path in max_path])

        '''for index in range(n):
            for state in self.tag_set:

                # start position
                j = self.tag_dict[state]
                if index == 0:
                    prior_state = '#'
                    self.delta[index][j] = self._transfer_prob(sentence,'#',state,index)
                    continue

                probs = []
                for state2 in self.tag_set:
                    k = self.tag_set[state2]
                    probs.append(self._transfer_prob(sentence, state, state2, index)+self.delta[index][j])
        '''


class Node(object):
    def __init__(self, step, state, prob, prior):
        '''

        Parameters
        ----------
        step
        state
        prob
        prior
        '''
        self.step = step
        self.state = state
        self.prob = prob
        self.prior = prior
