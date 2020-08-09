#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: corpus.py
Description: Corpus for Chinese Word Segmentation Problem
Author: Barry Chow
Date: 2020/7/29 10:18 PM
Version: 0.1
"""
from datasets import load_msr

DATASET_FUNCTIONS = {
    'MSR': load_msr
}


class Corpus(object):

    def __init__(self, dataset='MSR', max_samples=1000, max_length=1000, min_length=5):
        self.max_samples = max_samples
        self.max_length = max_length
        self.min_length = min_length
        self.data_type = dataset
        self._preprocess()

    def _tag_word(self, word):
        '''
        tagging single word

        Parameters
        ----------
        word

        Returns
        -------

        '''
        return "S" if len(word) == 1 else "B" + "M" * (len(word) - 2) + "E"

    def _tag_sequence(self, words):
        '''
        tagging word sequence

        Parameters
        ----------
        words

        Returns
        -------

        '''
        return "".join([self._tag_word(word) for word in words])

    def _preprocess(self):
        '''
        given input word segmentation dataset, return 4-tag{B,M,E,S} tag sequence.

        Returns
        -------

        '''
        data = DATASET_FUNCTIONS[self.data_type](self.max_samples, self.max_length, self.min_length)
        self.tags, self.dataset = [], []
        for line in data:
            self.tags.append(self._tag_sequence(line))
            self.dataset.append("".join(line))
