#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: test_svm.py
Description: 
Author: Barry Chow
Date: 2020/3/16 9:34 PM
Version: 0.1
"""
from numpy import array, mat

from datasets import load_svm_data
from svm import BinarySVM
from utils import accuracy_score
from utils import split_train_test


class TestSVM(object):
    def test_simple_svm(self):
        dataset, labels = load_svm_data()
        svm = BinarySVM(C=0.5, max_iter=40)
        train_X, train_y, test_X, test_y = split_train_test(array(dataset), array(labels), scale=0.7, is_random=True)

        svm.fit(mat(train_X), mat(train_y))
        preds = svm.predict(mat(test_X))
        accuracy = accuracy_score(preds, test_y)

        '''
        svm.fit(dataset, labels)
        preds = svm.predict(dataset)
        accuracy = accuracy_score(preds, array(labels.T.tolist()[0]))
        '''

        assert accuracy > 0.8
