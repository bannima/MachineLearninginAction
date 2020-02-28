#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: tree.py
Description: 
Author: Barry Chow
Date: 2020/2/27 9:03 PM
Version: 0.1
"""

from numpy import *
from utils.base import BaseModel
from math import log
from abc import ABCMeta,abstractmethod
from criterion import _CRITERION
from utils.tools import filter_cate_feat_data
from utils.tools import  filter_cont_feat_data

class BaseDecisionTree(BaseModel, metaclass=ABCMeta):
    """Base Decision Tree

       Warning: This class should not be used directly.
       Use derived classes instead.
    """
    def __init__(self,max_depth,max_leafs,min_sample_splits,epsilon,impurity):
        self.max_depth = max_depth
        self.max_leafs = max_leafs
        self.min_sample_splits = min_sample_splits
        self.epsilon = epsilon
        self.criterion = _CRITERION[impurity]()
        super(BaseDecisionTree,self).__init__()

    @abstractmethod
    def _choose_best_feat(self,dataset,labels):
        """choose the best feature according to criterion

        Parameters
        ----------
        dataset: array_like, shape = (n_samples,n_features)

        labels: array_like, shape = (n_samples,1)

        Returns
        -------
        the corresponding feature index of maximum criterion and maximum criterion

        """
        pass

    @abstractmethod
    def _build_branch(self,dataset,labels,best_feat,current_depth):
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
        pass

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
        n_samples,_ = shape(dataset)
        num_labels,most_label,max_count = self._check_labels(labels)

        # depth limitation and min splits limitation
        if current_depth + 1 >= self.max_depth or n_samples <= self.min_sample_splits:
            return most_label

        #labels are the same kind, return the specific label
        if num_labels==1:
            return most_label

        #empty feature set or all the same feature combination, return the most occured label
        feat_nums,single_feat_comb = self._check_features(dataset)
        if feat_nums==0 or single_feat_comb:
            return most_label

        #choose the best feature
        best_feat, maximum_criterion = self._choose_best_feat(dataset,labels)

        #criterion is less than epsilon, return most occuerd label
        if maximum_criterion<self.epsilon:
            return most_label

        #build branch
        return {best_feat:self._build_branch(dataset,labels,best_feat,current_depth+1)}

    def _check_features(self,dataset):
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
        return shape(dataset)[1], False not in (dataset==dataset[0])

    def _check_labels(self,labels):
        """
        check the dataset label kinds, and most occured label and max count

        Parameters
        ----------
        labels: array_like, shape = (n_samples,1)

        Returns
        -------
        the label kinds, int

        most counted label,int

        corresponding max count ,int

        """
        label_count = {}
        label_list = set(labels.T.tolist()[0])
        for label in label_list:
            label_count[label] = labels[labels==label].T.shape[0]
        (most_label,max_count) = sorted(label_count.items(),key=lambda asv:asv[1],reverse=True)[0]

        return len(set(labels.T.tolist()[0])),most_label,max_count

    def set_feature_labels(self,feat_labels):
        self.feat_labels = feat_labels

    def fit(self,X,y):
        tree = self._build_tree(X, y,0)
        print(tree)

    def predict(self,X):
        pass

    '''
    @abstractmethod
    def plot_tree(self):
        pass
    '''

class ID3(BaseDecisionTree):
    def __init__(self,max_depth=100,max_leafs=1000,min_sample_splits=10,epsilon=e-4,impurity='entropy'):
        super(ID3,self).__init__(
            max_depth=max_depth,
            max_leafs=max_leafs,
            min_sample_splits=min_sample_splits,
            epsilon=epsilon,
            impurity=impurity
            )

    def _choose_best_feat(self,dataset,labels):
        """choose the best feature according to criterion

        Note that, in this implementaion ID3 only accept categorical features
        and split dataset should remove the given feature column according to
        the nature of multiway branch of ID3 algorithms.

        """
        _,n = shape(dataset);maximum_criterion= -inf;best_ind = -1
        for feat_ind in range(n):
            if self.criterion.criterion(dataset,labels,feat_ind)>maximum_criterion:
                maximum_criterion = self.criterion.criterion(dataset,labels,feat_ind)
                best_ind = feat_ind
        return best_ind,maximum_criterion

    def _build_branch(self,dataset,labels,best_feat,current_depth):
        """build branch tree based on the selected best feature index

        """
        best_ind = best_feat;tree = {}
        for feat_val in set(dataset[:,best_ind].T.tolist()[0]):
            filtered_dataset,filtered_labels = filter_cate_feat_data(dataset,labels,best_ind,feat_val)
            tree[feat_val]=self._build_tree(filtered_dataset, filtered_labels,current_depth+1)
        return tree



class C45(BaseDecisionTree):
    def __init__(self,max_depth,max_leafs,epsilon=0.001):
        super(C45,self).__init__(
            max_depth=max_depth,
            max_leafs=max_leafs,
            epsilon=epsilon)

class CARTClassifier(BaseDecisionTree):
    def __init__(self, max_depth=100, max_leafs=1000, min_sample_splits=10, epsilon=e - 4, impurity='gini'):
        super(CARTClassifier, self).__init__(
            max_depth=max_depth,
            max_leafs=max_leafs,
            min_sample_splits=min_sample_splits,
            epsilon=epsilon,
            impurity=impurity
        )

    def _choose_best_feat(self,dataset,labels):
        """choose the best feature according to criterion

        Note that, in this implementaion CARTClassifier only accept continuous features,
        which means that the selected features may be reused in the branch tree.
        """
        _,n = shape(dataset);maximum_criterion= -inf;best_feat = (-1,inf)
        for feat_ind in range(n):
            for feat_val in sort(list(set(dataset[:,feat_ind].T.tolist()[0])))[1:]:
                if self.criterion.criterion(dataset,labels,feat_ind,feat_val)>maximum_criterion:
                    maximum_criterion = self.criterion.criterion(dataset,labels,feat_ind,feat_val)
                    best_feat = (feat_ind,feat_val)
        return best_feat,maximum_criterion

    def _build_branch(self,dataset,labels,best_feat,current_depth):
        """build branch tree based on the selected best feature index and value

        Note that the tree will be binary , while left branch means less than the current
        feature value and the right otherwise.
        """
        (best_ind,feat_val) = best_feat;tree = {}
        for symbol in ['lt','nlt']:
            filtered_dataset,filtered_labels = filter_cont_feat_data(dataset,labels,best_ind,feat_val,symbol)
            tree[symbol]=self._build_tree(filtered_dataset,filtered_labels,current_depth+1)
        return tree




