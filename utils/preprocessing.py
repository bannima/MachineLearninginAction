#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: preprocessing.py
Description: 
Author: Barry Chow
Date: 2020/3/4 7:41 PM
Version: 0.1
"""
from numpy import *
from numpy import random


def one_hot(X):
    '''
    transform input array to one hot vector

    Parameters:
    -----------
    X: array_like

    Returns:
    --------
    one hot vector matrix

    '''
    values = array(X)
    n_values = max(values) + 1
    return eye(n_values)[values]


def split_train_test(X, y, scale=0.7, is_random=True):
    '''
    return train data, train label and test data ,test label

    Parameters
    ----------
    X: array_like
    y: array_like
    scale: the rate of train and test dataset
    is_random: whether reshuffle the data

    Returns
    -------
    train_x,train_y,test_x,test_y

    '''
    n_samples,n_features = shape(X)
    index = array(range(n_samples))
    break_point = int(n_samples*scale)
    if is_random:
        train_index = random.choice(n_samples, break_point)
        test_index = array(list(set(index)-set(train_index)))
    else:
        train_index = index[:break_point]
        test_index = index[break_point:]
    return X[train_index],y[train_index],X[test_index],y[test_index]
