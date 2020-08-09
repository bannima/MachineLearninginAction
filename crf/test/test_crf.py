#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: test_crf.py
Description: 
Author: Barry Chow
Date: 2020/5/31 10:33 PM
Version: 0.1
"""
import datetime

from sklearn.model_selection import train_test_split

from crf import Corpus, TemplateBuilder, FeatureBuilder, LinearChainCRF


class TestCRF(object):
    def test_CRF_with_Word_Segmentation(self):
        starttime = datetime.datetime.now()

        # template of feature functions
        template_builder = TemplateBuilder("feature.template")

        # input dataset and tags
        msr_corpus = Corpus('MSR', max_length=15, min_length=5, max_samples=10000)

        dataset, tags = msr_corpus.dataset, msr_corpus.tags
        print("dataset nums: " + str(len(dataset)))

        # split train and test dataset
        X_train, X_test, Y_train, Y_test = train_test_split(dataset, tags, test_size=0.3, random_state=0)

        # build feature functions according to input dataset and feature templates
        feature_builder = FeatureBuilder(X_train, Y_train, template_builder)

        # CRF model
        crf = LinearChainCRF(feature_builder, epochs=30, learning_rate=1e-2)
        crf.fit(X_train, Y_train)

        endtime = datetime.datetime.now()
        print("training time: " + str((endtime - starttime).seconds) + " seconds")

        # predict
        predicts = crf.predict(X_test)

        # evaluation
        total, correct = 0, 0
        for pred, tag in zip(predicts, Y_test):
            for i in range(len(pred)):
                if pred[i] == tag[i]:
                    correct += 1
                total += 1
        accuracy = float(correct) / total
        print("accuracy: ", accuracy)
