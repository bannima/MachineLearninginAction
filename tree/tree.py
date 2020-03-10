#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: tree.py
Description: 
Author: Barry Chow
Date: 2020/2/27 9:03 PM
Version: 0.1
"""

from abc import ABCMeta, abstractmethod
from utils import CRITERION
from numpy import *

from base import BaseModel
from utils.enum import SYMBOL
from utils.tools import filter_cate_feat_data
from utils.tools import filter_cont_feat_data


class BaseDecisionTree(BaseModel, metaclass=ABCMeta):
    """Base Decision Tree

       Warning: This class should not be used directly.
       Use derived classes instead.
    """

    def __init__(self, max_depth, max_leafs, min_sample_splits, epsilon, is_classification, impurity):
        self.max_depth = max_depth
        self.max_leafs = max_leafs
        self.min_sample_splits = min_sample_splits
        self.epsilon = epsilon
        if impurity not in CRITERION:
            raise ValueError("impurity {} is not in selected criterion type {}".format(impurity, CRITERION.keys()))
        self.criterion = CRITERION[impurity]()
        if (getattr(self.criterion, "_criterion_type_", None) == "classification") != is_classification:
            raise ValueError("tree type does not matches the criterion type ")
        self.is_classification = is_classification
        self.tree = None

        self._check_params()

        super(BaseDecisionTree, self).__init__()


    def _check_params(self):
        '''
        check the validity of input params

        '''
        assert isinstance(self.max_depth,int)
        assert self.max_depth>0

        assert isinstance(self.max_leafs,int)
        assert self.max_leafs>0

        assert isinstance(self.min_sample_splits,int)
        assert self.min_sample_splits>0

        assert self.epsilon>0

        assert type(self.is_classification).__name__ == 'bool'


    @abstractmethod
    def _choose_best_feat(self, dataset, labels):
        """choose the best feature according to criterion

        Parameters
        ----------
        dataset: array_like, shape = (n_samples,n_features)

        labels: array_like, shape = (n_samples,1)

        Returns
        -------
        the corresponding feature index of maximum criterion and maximum criterion

        """


    @abstractmethod
    def _build_branch(self, dataset, labels, best_feat, current_depth):
        """build branch tree based on the selected best feature index

        Note that, for ID3 and C4.5, it's multiway tree, one branch for one categorical featrue value.
        As for CART, it's binary tree.

        Parameters
        ----------
        dataset: array_like, shape = (n_samples,n_features)

        labels: array_like, shape = (n_samples,1)

        best_feat: the selected best feature, in detail,feature index for ID3, and feature index and value
        for CART

        current_depth: the tree depth, must less than max depth

        Returns
        -------
        the branch tree

        """


    def _build_tree(self, dataset, labels, current_depth):
        """build decision tree based on criterion

        Parameters
        ----------
        dataset: array_like, shape = (n_samples,n_features)

        labels: array_like, shape = (n_samples,1)

        Returns
        -------
        the unpruned tree stored in dict

        """
        n_samples, n_features = shape(dataset)
        num_labels, leaf_label = self._check_labels(labels)

        # depth limitation and min splits limitation
        if current_depth + 1 >= self.max_depth or n_samples <= self.min_sample_splits:
            return leaf_label

        # labels are the same kind, return the specific label
        if num_labels == 1:
            return leaf_label

        # empty feature set or all the same feature combination, return the most occured label
        single_feat_comb = self._check_features(dataset)
        if n_features == 0 or single_feat_comb:
            return leaf_label

        # choose the best feature
        best_feat, maximum_criterion = self._choose_best_feat(dataset, labels)

        # criterion is less than epsilon, return most occuerd label
        if maximum_criterion < self.epsilon:
            return leaf_label

        # build branch
        return {best_feat: self._build_branch(dataset, labels, best_feat, current_depth + 1)}

    def _check_features(self, dataset):
        """
        return the nums of feature and whether contains only one features combination

        Parameters
        ----------
        dataset: array_like, shape = (n_samples,n_features)

        Returns
        -------
        the feature nums, int

        is single feature cmbination, boolean

        """
        return False not in (dataset == dataset[0])

    def _check_labels(self, labels):
        """
        check the dataset label kinds, and return the most occured label(classification) or mean value(regression)
        as the leaf labels.

        Parameters
        ----------
        labels: array_like, shape = (n_samples,1)

        Returns
        -------
        the label kinds, int

        most counted label or mean value, int or float

        """

        label_count = {}
        label_list = set(labels.T.tolist()[0])
        for label in label_list:
            label_count[label] = labels[labels == label].T.shape[0]
        (freq_label, max_count) = sorted(label_count.items(), key=lambda asv: asv[1], reverse=True)[0]
        if self.is_classification:
            return len(label_list), freq_label
        else:
            return len(label_list), mean(labels.T.tolist()[0])

    def set_feature_labels(self, feat_labels):
        """
        set the feature label meanings

        Parameters
        ----------
        feat_labels: array_like, len = n_features

        Returns
        -------
        None

        """
        self.feat_labels = feat_labels

    def fit(self, X, y):
        #transform the label array(length: nsamples) to  matrix(shape: n_sample,1)
        y = mat(y).T

        self.tree = self._build_tree(X, y, 0)
        #print(self.tree)

        # prune strategy

    def predict(self, X):
        """
        predict the given test dataset

        Parameters
        ----------
        X: array_like, shape = (n_samples,n_features)

        Returns
        -------
        predicted labels according to tree, shape=(n_samples,1)

        """
        if self.tree is None:
            raise RuntimeError("ERROR: the tree has not been trained yet.")
        return [self._pred_sample(sample, self.tree) for sample in array(X)]

    def _pred_sample(self, sample, tree):
        """
        predict the single sample using recursive method

        Warning, this method should be implemented in classification and regression tree separately

        Parameters
        ----------
        sample: array_like, shape = (1,n_features)

        Returns
        -------
        predicted labels according to tree

        """
        # lead node
        if type(tree).__name__ != 'dict':
            return tree

        root = list(tree.keys())[0]

        # regression
        if type(root).__name__ == 'tuple':
            feat_ind, feat_val = root
            if sample[feat_ind] < feat_val:
                return self._pred_sample(sample, tree[root][SYMBOL.LT])
            else:
                return self._pred_sample(sample, tree[root][SYMBOL.NLT])

        # classification
        else:
            feat_ind = root
            feat_val = sample[feat_ind]
            return self._pred_sample(sample, tree[feat_ind][feat_val])

    '''
    @abstractmethod
    def plot_tree(self):
        pass
    '''


class ID3(BaseDecisionTree):

    _SUPPORT_CRITERION = ['gini','entropy']

    def __init__(self, max_depth=100, max_leafs=1000, min_sample_splits=2, epsilon=1e-4, is_classification=True,
                 impurity='entropy'):

        super(ID3, self).__init__(
            max_depth=max_depth,
            max_leafs=max_leafs,
            min_sample_splits=min_sample_splits,
            epsilon=epsilon,
            is_classification=is_classification,
            impurity=impurity
        )


    def _check_params(self):
        '''
        customize derived class should check it's own params

        warning: if you override the check params function in order to customize
        derived params checking, you should invoke the param_checking funciton of
        base class explicity. Otherwise, just ingoring this funcion, and base class
        will invoke it's own param_checking function.

        '''

        #customize param_checking function of derived class
        assert self.is_classification==True

        super(ID3,self)._check_params()


    def _choose_best_feat(self, dataset, labels):
        """choose the best feature according to criterion

        Note that, in this implementaion ID3 only accept categorical features
        and split dataset should remove the given feature column according to
        the nature of multiway branch of ID3 algorithms.

        """
        _, n = shape(dataset)
        maximum_criterion = -inf
        best_ind = -1
        for feat_ind in range(n):
            if self.criterion(dataset, labels, feat_ind) > maximum_criterion:
                maximum_criterion = self.criterion(dataset, labels, feat_ind)
                best_ind = feat_ind
        return best_ind, maximum_criterion

    def _build_branch(self, dataset, labels, best_feat, current_depth):
        """build branch tree based on the selected best feature index

        """
        best_ind = best_feat
        tree = {}
        for feat_val in set(dataset[:, best_ind].T.tolist()[0]):
            filtered_dataset, filtered_labels = filter_cate_feat_data(dataset, labels, best_ind, feat_val)
            tree[feat_val] = self._build_tree(filtered_dataset, filtered_labels, current_depth + 1)
        return tree


class CARTClassifier(BaseDecisionTree):

    def __init__(self, max_depth=100, max_leafs=1000, min_sample_splits=2, epsilon=1e-4, impurity='gini'):
        super(CARTClassifier, self).__init__(
            max_depth=max_depth,
            max_leafs=max_leafs,
            min_sample_splits=min_sample_splits,
            epsilon=epsilon,
            is_classification=True,
            impurity=impurity
        )

    def _choose_best_feat(self, dataset, labels):
        """choose the best feature according to criterion

        Note that, in this implementaion CARTClassifier only accept continuous features,
        which means that the selected features may be reused in the branch tree.
        """
        _, n = shape(dataset)
        maximum_criterion = -inf
        best_feat = (-1, inf)
        for feat_ind in range(n):
            for feat_val in sort(list(set(dataset[:, feat_ind].T.tolist()[0])))[1:]:
                if self.criterion(dataset, labels, feat_ind, feat_val) > maximum_criterion:
                    maximum_criterion = self.criterion(dataset, labels, feat_ind, feat_val)
                    best_feat = (feat_ind, feat_val)
        return best_feat, maximum_criterion

    def _build_branch(self, dataset, labels, best_feat, current_depth):
        """build branch tree based on the selected best feature index and value

        Note that the tree will be binary , while left branch means less than the current
        feature value and the right otherwise.
        """
        (best_ind, feat_val) = best_feat
        tree = {}
        for symbol in [SYMBOL.LT, SYMBOL.NLT]:
            filtered_dataset, filtered_labels = filter_cont_feat_data(dataset, labels, best_ind, feat_val, symbol)
            tree[symbol] = self._build_tree(filtered_dataset, filtered_labels, current_depth + 1)
        return tree


class CARTRegressor(BaseDecisionTree):
    def __init__(self, max_depth=100, max_leafs=1000, min_sample_splits=2, epsilon=1e-4, impurity='mse'):
        super(CARTRegressor, self).__init__(
            max_depth=max_depth,
            max_leafs=max_leafs,
            min_sample_splits=min_sample_splits,
            epsilon=epsilon,
            is_classification=False,
            impurity=impurity
        )

    def _choose_best_feat(self, dataset, labels):
        """choose the best feature according to criterion

        Note that, in this implementaion CARTClassifier only accept continuous features,
        which means that the selected features may be reused in the branch tree.
        """
        _, n = shape(dataset)
        maximum_criterion = -inf
        best_feat = (-1, inf)
        for feat_ind in range(n):
            for feat_val in sort(list(set(dataset[:, feat_ind].T.tolist()[0])))[1:]:
                if self.criterion(dataset, labels, feat_ind, feat_val) > maximum_criterion:
                    maximum_criterion = self.criterion(dataset, labels, feat_ind, feat_val)
                    best_feat = (feat_ind, feat_val)
        return best_feat, maximum_criterion

    def _build_branch(self, dataset, labels, best_feat, current_depth):
        """build branch tree based on the selected best feature index and value

        Note that the tree will be binary , while left branch means less than the current
        feature value and the right otherwise.
        """
        (best_ind, feat_val) = best_feat
        tree = {}
        for symbol in [SYMBOL.LT, SYMBOL.NLT]:
            filtered_dataset, filtered_labels = filter_cont_feat_data(dataset, labels, best_ind, feat_val, symbol)
            tree[symbol] = self._build_tree(filtered_dataset, filtered_labels, current_depth + 1)
        return tree
