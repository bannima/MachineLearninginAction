#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: logistic_regression.py
Description: logistic regression
Author: Barry Chow
Date: 2020/3/25 4:59 PM
Version: 0.1
"""
from numpy import shape, zeros, inf, array,log,exp
from base import BaseModel
from utils import sigmod,sign
from utils import accuracy_score
from random import random

class LogisticRegression(BaseModel):
    def __init__(self, learning_rate=1e-2, max_iter=1000, threshold=1e-2):
        """

        Parameters
        ----------
        learning_rate: float, learning rate

        max_iter: int, maximum iterations

        threshold: float, minimum loss gain

        """
        assert 0 < learning_rate <= 1
        self.learning_rate = learning_rate

        assert max_iter > 0
        assert isinstance(max_iter, int)
        self.max_iter = max_iter

        assert 0 < threshold < 1
        self.threshold = threshold

        super(LogisticRegression, self).__init__()

    def fit(self, X, y):
        """
        fit the logistic regression, until loss gain is less than threshold or
        reach the maximum iterations.

        Parameters
        ----------
        X: array_like, training data

        y: array_like, training labels

        """
        n_samples, n_features = shape(X)
        assert n_samples == len(y)

        # init parameters
        self.w = [random() for _ in range(n_features)]
        self.b = 0

        self._fit(X,y)

    def _fit(self,X,y):
        """
        fit the logistic regression model.

        Parameters
        ----------
        X: array_like, training data

        y: array_like, training labels

        """
        total_loss = inf
        for iter in range(self.max_iter):

            #update parameters w and b
            preds = array([self._predict_sample(sample) for sample in X ])

            delta_w = [self.learning_rate*sum((array(y)-preds)*X[:,j]) for j in range(shape(X)[1])]
            self.w+=array(delta_w)

            delta_b = self.learning_rate*sum(array(y)-preds)
            self.b+=delta_b

            loss = self._calc_loss(X,y)
            # stop iteration when loss gain is less than threshold
            if (total_loss - loss) < self.threshold:
                return
            else:
                total_loss = loss

            #report accuracy score
            preds = self.predict(X)
            print(accuracy_score(preds,y))


    def _calc_loss(self,X,y):
        '''
        calc total loss

        Parameters
        ----------
        X: array_like, training data

        y: array_like, training labels

        Returns
        -------
        loss: float, total loss

        '''
        loss = 0
        for ind,sample in enumerate(X):
            w_xi = sum(sample * self.w) + self.b
            loss += y[ind]*w_xi-log(1+exp(w_xi))
        return loss

    def _predict_sample(self, sample):
        """
        predict the single sample value.

        Parameters
        ----------
        sample

        Returns
        -------
        sigmod value for given sample

        """
        return sigmod(sum(sample * self.w) + self.b)

    def predict(self, X):
        """

        Parameters
        ----------
        X: array_like, input data

        Returns
        -------
        labels for given point, 0 or 1 for negative and positive result

        """
        return [sign(self._predict_sample(sample),1,0.5,0) for sample in X]
