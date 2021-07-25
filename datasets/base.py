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


def load_msr(max_line=1000,max_length=1000,min_length=5):
    '''

    Returns
    -------

    '''
    res =[]
    base_dir = os.path.dirname(__file__) + '/data'
    pathfile = os.path.join(base_dir, 'msr_training.utf8')
    count=0
    for line in open(pathfile, 'r'):
        if len(line)>max_length or len(line)<min_length:
            continue
        if count>max_line:
            break
        line = [word for word in line.strip(' \r\n').split(" ") if len(word) > 0]
        res.append(line)
        count+=1
    return res

def loadHorseColicDataset(filename):
    numFeat = len(open(filename).readline().split('\t'))
    dataMat = [];
    labelMat = []
    fr = open(filename)
    for line in open(filename):
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def loadHorseColic():
    base_dir = os.path.dirname(__file__) + '/data'
    train_file = os.path.join(base_dir, 'horseColicTraining2.txt')
    test_file = os.path.join(base_dir, 'horseColicTest2.txt')

    train_X, train_y = loadHorseColicDataset(train_file)
    test_X, test_y = loadHorseColicDataset(test_file)
    return train_X,train_y,test_X,test_y

def loadAGNewsTestCorpus():
    base_dir = os.path.dirname(__file__) + '/data'
    file = os.path.join(base_dir, 'agnews_test_texts.txt')
    corpus = []
    with open(file,'r',encoding='utf-8') as fread:
        for line in fread.readlines():
            corpus.append(line)
    return corpus


