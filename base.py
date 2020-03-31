#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: base.py
Description: 
Author: Barry Chow
Date: 2020/2/23 6:07 PM
Version: 0.1
"""
from abc import ABCMeta
from abc import abstractmethod


class Classifier(metaclass=ABCMeta):
    """Base class for all classifiers

        Warning: This class should not be used directly.
        Use derived classes instead.
    """

    @abstractmethod
    def fit(self, X, y):
        """Given train data X and labels y,and feature labels,  fit the classifier

        Parameters
        ----------
        X : array_like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        y : array_like, length = n_samples

        Returns
        -------
        None
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X):
        """Given train data X and labels y, fit the classifier

        Parameters
        ----------
        X : array_like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        predit labels,array_like, length=n_samples
        """
        raise NotImplementedError()
