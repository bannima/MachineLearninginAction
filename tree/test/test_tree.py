#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: test_tree.py
Description: 
Author: Barry Chow
Date: 2020/2/23 7:21 PM
Version: 0.1
"""
from tree import ID3
from numpy import *
from utils import accuracy_score
from datasets import load_loans
from sklearn.datasets import load_iris
from tree import CARTRegressor,CARTClassifier

iris = load_iris()


class TestTree(object):

    def test_id3(self):
        id3 = ID3(max_depth=100, max_leafs=100, epsilon=0.00001)
        dataset, labels, feat_labels = load_loans()
        id3.set_feature_labels(feat_labels)
        id3.fit(dataset,labels)
        preds = id3.predict(dataset)
        score = accuracy_score(preds,labels)
        assert score>0.99

    def test_cart_classifier(self):
        cart_classifier = CARTClassifier(max_depth=100)
        cart_classifier.fit(mat(iris.data), iris.target)
        preds = cart_classifier.predict(mat(iris.data))
        score = accuracy_score(preds,iris.target)
        assert score>0.99

    def test_cart_regressor(self):
        cart_regressor = CARTRegressor()
        cart_regressor.fit(mat(iris.data),iris.target)
        preds = cart_regressor.predict(mat(iris.data))






