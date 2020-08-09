#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: test_features.py
Description: 
Author: Barry Chow
Date: 2020/8/6 11:53 PM
Version: 0.1
"""
from crf import Corpus, TemplateBuilder, FeatureBuilder
from sklearn.model_selection import train_test_split


class TestGenerateFeatures(object):
    def test_gen_features(self):
        # template of feature functions
        template_builder = TemplateBuilder("feature.template")

        # input dataset and tags
        msr_corpus = Corpus('MSR',max_length=100, max_samples=1000)

        dataset, tags = msr_corpus.dataset, msr_corpus.tags

        # split train and test dataset
        X_train, X_test, Y_train, Y_test = train_test_split(dataset, tags, test_size=0.3, random_state=0)

        # build feature functions according to input dataset and feature templates
        feature_builder = FeatureBuilder(X_train,Y_train,template_builder)

