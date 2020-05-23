#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: base.py
Description: 
Author: Barry Chow
Date: 2020/3/3 7:01 PM
Version: 0.1
"""

import os

from numpy import *


def load_loans():
    '''
    load loans dataset

    return
    ------

    dataset,labels,feature_labels
    '''
    res = []
    base_dir = os.path.dirname(__file__) + '/data'
    pathfile = os.path.join(base_dir, 'loads.txt')
    for line in open(pathfile, 'r'):
        line = line.strip(' \n').split('\t')
        res.append(line)
    dataset = res[1:]
    feat_labels = res[0]
    return mat(dataset)[:, 0:-1], mat(dataset)[:, -1].T.tolist()[0], feat_labels


def load_svm_data():
    '''

    :return:
    '''
    res = []
    base_dir = os.path.dirname(__file__) + '/data'
    pathfile = os.path.join(base_dir, 'svmTestSet.txt')
    for line in open(pathfile, 'r'):
        line = line.strip(' \n').split('\t')
        res.append([float(v) for v in line])
    return mat(res)[:, 0:-1], mat(res)[:, -1]
