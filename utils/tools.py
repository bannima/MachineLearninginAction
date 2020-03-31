#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: tools.py
Description: tool functions for machine learning
Author: Barry Chow
Date: 2020/2/27 9:57 PM
Version: 0.1
"""
from numpy import *

from .enum import SYMBOL


def calc_entropy(labels):
    """calculate the shannon entropy of a given dataset labels

    Parameters
    ----------
    labels : array_like, shape = (n_samples,1), discrete features for classification problems

    Returns
    -------
    float, shannon entropy
    """
    shannon_entropy = 0.0
    label_list = set(labels.T.tolist()[0])
    for label in label_list:
        p = labels[labels == label].T.shape[0] / float(labels.shape[0])
        shannon_entropy -= p * math.log(p, 2)
    return shannon_entropy


def calc_conditional_entropy(dataset, labels, feat_ind, feat_val):
    """calculate the emperical conditional entropy according to given feature index and feat_val

    Parameters
    ----------
    dataset: array_like, shape = (n_samples,n_features)


    feat_ind: the specific feature index, int

    Returns
    -------
    corresponding conditional entropy of the given feature

    """
    m, n = shape(dataset);
    conditional_entropy = 0.0
    feat_value_list = set(dataset[:, feat_ind].T.tolist()[0])
    for feat_val in feat_value_list:
        filtered_dataset, filtered_labels = filter_cate_feat_data(dataset, labels, feat_ind, feat_val)
        conditional_entropy += float(shape(filtered_dataset)[0]) / m * \
                               calc_entropy(filtered_labels)
    return conditional_entropy


def calc_conditional_gini(dataset, labels, feat_ind, feat_val):
    """calculate the emperical conditional gini according to given feature index and feat_val

    Parameters
    ----------
    dataset: array_like, shape = (n_samples,n_features)


    feat_ind: the specific feature index, int

    Returns
    -------
    corresponding conditional entropy of the given feature

    """
    m, _ = shape(dataset);
    conditional_gini = 0.0
    for symbol in [SYMBOL.LT, SYMBOL.NLT]:
        filtered_dataset, filtered_labels = filter_cont_feat_data(dataset, labels, feat_ind, feat_val, symbol)
        conditional_gini += float(shape(filtered_dataset)[0]) / m * \
                            calc_gini(filtered_labels)
    return conditional_gini


def calc_gini(labels):
    """calculate the gini of a given dataset labels

    Parameters
    ----------
    labels : array_like, shape = (n_samples,1), discrete features for classification problems

    Returns
    -------
    float, gini
    """
    label_list = set(labels.T.tolist()[0]);
    gini = 1
    for label in label_list:
        gini -= math.pow(labels[labels == label].T.shape[0] / float(labels.shape[0]), 2)
    return gini


def calc_conditional_mse(dataset, labels, feat_ind, feat_val):
    """calculate the conditional mean square error according to given feature index and feat_val

    Parameters
    ----------
    dataset: array_like, shape = (n_samples,n_features)


    feat_ind: the specific feature index, int

    Returns
    -------
    corresponding conditional entropy of the given feature

    """
    m, _ = shape(dataset);
    conditional_mse = 0.0
    for symbol in [SYMBOL.LT, SYMBOL.NLT]:
        _, filtered_labels = filter_cont_feat_data(dataset, labels, feat_ind, feat_val, symbol)
        conditional_mse += var(filtered_labels.T.tolist()[0])
    return conditional_mse


def filter_cate_feat_data(dataset, labels, feat_ind, feat_val):
    """
    return the filtered dataset according to the feature index and feature value

    Note that it will delete the given feature because categorical feature will
    only be used at one times.

    Parameters
    ----------
    dataset: array_like, shape = (n_samples,n_features)

    labels: array_like, shape = (n_samples,1)

    feat_ind: the specific feature index, int

    feat_val: the specific feature value

    Returns
    -------
    filtered dataset, array_like

    corresponding filtered labels,array_like

    """
    filter_index = nonzero(dataset[:, feat_ind] == feat_val)[0]
    return delete(dataset[filter_index], [feat_ind], axis=1), labels[filter_index]


def filter_cont_feat_data(dataset, labels, feat_ind, feat_val, symbol):
    """
    filter the dataset by continus feature index and value
    Note that it's binary tree with left branch less than the feature value,
    and the selected continus feature is not deleted compared with categorical
    features

    Parameters
    ----------
    dataset: array_like, shape = (n_samples,n_features)

    labels: array_like, shape = (n_samples,1)

    feat_ind: the specific feature index, int

    feat_val: the specific feature value

    symbol: str, lt or nlt corresponding to less than, or no less than

    Returns
    -------
    filtered dataset, array_like

    corresponding filtered labels,array_like

    """
    assert symbol in [SYMBOL.LT, SYMBOL.NLT]
    if symbol == SYMBOL.LT:
        filter_index = nonzero(dataset[:, feat_ind] < feat_val)[0]
    else:
        filter_index = nonzero(dataset[:, feat_ind] >= feat_val)[0]
    return dataset[filter_index], labels[filter_index]
