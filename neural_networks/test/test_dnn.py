#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: test_dnn.py
Description: 
Author: Barry Chow
Date: 2020/3/30 6:46 PM
Version: 0.1
"""
from numpy import array
from sklearn.datasets import load_digits

from neural_networks import DNN
from utils import split_train_test, one_hot, accuracy_score


class Test_DNN(object):

    def test_XOR(self):
        # data = array([[1,1],[1,0],[0,1],[0,0]])
        data = array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
        labels = array([[0, 1], [1, 0], [1, 0], [0, 1]])
        res_labels = array([0, 1, 1, 0])

        classifier = DNN(layers=[2, 3, 2], learning_rate=0.3, activation='sigmod', Epochs=10000, threhold=0.1)
        classifier.fit(data, labels)

        preds = classifier.predict(data)
        res_preds = preds.argmax(axis=1)
        print(preds)
        assert accuracy_score(res_preds, res_labels) > 0.8

    def test_digits(self):
        digits = load_digits(n_class=10)
        data = digits['data']
        labels = one_hot(digits['target'])

        train_X, train_y, test_X, test_y = split_train_test(data, labels)
        # classifier = DNN(layers= [64,50,10],learning_rate=0.3,activation='sigmod',Epochs=10,threhold=0.1)
        classifier = DNN(layers=[64, 50, 50, 10], learning_rate=0.1, activation='sigmod', Epochs=100, threhold=0.1)

        classifier.fit(train_X, train_y)
        preds = classifier.predict(test_X)
        res_test_y = test_y.argmax(axis=1)
        pred_test_y = preds.argmax(axis=1)
        print(accuracy_score(pred_test_y, res_test_y))
        assert accuracy_score(pred_test_y, res_test_y) > 0.7
