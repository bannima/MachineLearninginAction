#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: dnn.py
Description: 
Author: Barry Chow
Date: 2020/3/27 9:29 PM
Version: 0.1
"""
from copy import copy

from numpy import random, zeros, shape, dot, inf, matmul, array

from base import Classifier
from utils import ACTIVATIONS, LOSS_FUNCTIONS


class DNN(Classifier):

    def __init__(self, layers, learning_rate=1e-2, optimizer='SGD', loss='mse', activation='sigmod', Epochs=1000,
                 threhold=1e-3):
        super(DNN, self).__init__()
        assert isinstance(layers, list)

        assert isinstance(learning_rate, float)
        self.learning_rate = learning_rate

        self.optimizer = optimizer

        assert loss in LOSS_FUNCTIONS
        self.loss = LOSS_FUNCTIONS[loss]()

        assert activation in ACTIVATIONS
        self.activation = ACTIVATIONS[activation]()

        assert isinstance(Epochs, int)
        assert Epochs > 0
        self.Epochs = Epochs

        assert isinstance(threhold, float)
        assert threhold > 0
        self.threshold = threhold

        self._init_nn(layers)

    def _init_nn(self, layers):
        '''
        initialize the neural network

        Parameters
        ----------
        layers: array_like, the hidden neurons of each layer

        Returns
        -------
        None

        '''
        # weight matrixs
        self.W = []
        # bias
        self.B = []
        for index in range(len(layers) - 1):
            self.W.append(self._init_weights(layers[index], layers[index + 1]))
            self.B.append(random.rand(layers[index + 1]))

    def _init_weights(self, input_dimen, output_dimen, is_random=True):
        '''
        random initialzie weights

        Parameters
        ----------
        input_dimen: input dimension
        output_dimen: output dimension
        random: whether random initializer

        Returns
        -------
        weights with shape (input_dimen,output_dimen)
        '''
        if is_random:
            return random.rand(input_dimen, output_dimen) - 0.5
        else:
            return zeros((input_dimen, output_dimen))

    def forward(self, input):
        '''
        forward pass of current network model

        Returns
        -------
        predict results

        '''
        assert len(input) == shape(self.W[0])[0]
        preds = copy(input)
        # the neuron value in forward process
        neuron_values = [preds]
        for ind, weight in enumerate(self.W):
            preds = self.activation(dot(preds, weight) - self.B[ind])
            neuron_values.append(preds)

        return neuron_values

    def back_propagation(self, label, neuron_values):
        '''
        train the network weights using back propagation

        Returns
        -------

        '''
        # initial g
        g = 0.5 * self.loss.negative_gradient(neuron_values[-1], label) * self.activation.differential(
            neuron_values[-1])
        reverse_ind = list(range(len(self.W)))[::-1]
        for ind in reverse_ind:
            # calc delta W and B
            delta_W = self.learning_rate * matmul(neuron_values[ind].reshape(-1, 1), g.reshape(1, -1))
            delta_B = -1 * self.learning_rate * g

            # update g
            g = matmul(self.W[ind], g.reshape(-1, 1)).T[0]
            # g = matmul(g.reshape(1,-1),self.W[ind].T)

            # g = self.activation.differential(neuron_values[ind]).reshape(-1, 1) * g.reshape(1, -1)
            g = self.activation.differential(neuron_values[ind]) * g

            # update W and B
            self.W[ind] += delta_W
            self.B[ind] += delta_B

    def fit(self, X, y):
        last_loss = inf
        for _ in range(self.Epochs):
            epoch_loss = 0
            for sample, label in zip(X, y):
                # forward pass
                neuron_values = self.forward(sample)

                # back error propagation
                self.back_propagation(label, neuron_values)

                epoch_loss += sum(self.loss(neuron_values[-1], label))

            # stop when the loss gain is less than defined threshold
            # if abs(last_loss - epoch_loss) < self.threshold:
            #    break

            if last_loss > epoch_loss:
                last_loss = epoch_loss

    def _predict_sample(self, input):
        preds = copy(input)
        # the neuron value in forward process
        for ind, weight in enumerate(self.W):
            preds = self.activation(dot(preds, weight) + self.B[ind])
        return preds

    def predict(self, X):

        return array([self._predict_sample(sample) for sample in X])
