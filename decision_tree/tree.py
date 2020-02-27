#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: tree.py
Description: 
Author: Barry Chow
Date: 2020/2/23 6:47 PM
Version: 0.1
"""
from numpy import *
from utils.base import BaseClassifier
from math import log
from abc import ABCMeta,abstractmethod

class BaseDecisionTree(BaseClassifier,metaclass=ABCMeta):
    """Base Decision Tree

       Warning: This class should not be used directly.
       Use derived classes instead.
    """
    def __init__(self,max_depth,max_leafs,epsilon):
        self.max_depth = max_depth
        self.max_leafs = max_leafs
        self.epsilon = epsilon
        super(BaseDecisionTree,self).__init__()

    def _calc_shannon_entropy(self,labels):
        """calculate the shannon entropy of a given dataset labels

        Parameters
        ----------
        labels : array_like, shape = (n_samples,1), discrete features for classification problems

        Returns
        -------
        float, shannon entropy
        """
        shannon_entropy = 0.0;label_list = set(labels.T.tolist()[0])
        for label in label_list:
            p = labels[labels==label].T.shape[0]/float(labels.shape[0])
            shannon_entropy-=p*log(p,2)
        return shannon_entropy

    def _filter_dataset_by_feat_val(self, dataset, labels, feat_ind, feat_val):
        """
        return the filtered dataset according to the feature index and feature value

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
        filtered_dataset = dataset[nonzero(dataset[:, feat_ind] == feat_val)[0]]
        filtered_labels = labels[nonzero(dataset[:, feat_ind] == feat_val)[0]]
        return delete(filtered_dataset,[feat_ind],axis=1),filtered_labels

    def _calc_conditional_entropy(self,dataset,labels,feat_ind):
        """calculate the emperical conditional entropy according to given feature index

        Parameters
        ----------
        dataset: array_like, shape = (n_samples,n_features)


        feat_ind: the specific feature index, int

        Returns
        -------
        corresponding conditional entropy of the given feature

        """
        m,n = shape(dataset);conditional_entropy = 0.0
        feat_value_list = set(dataset[:,feat_ind].T.tolist()[0])
        for feat_val in feat_value_list:
            filtered_dataset,filtered_labels = self._filter_dataset_by_feat_val(dataset,labels,feat_ind,feat_val)
            conditional_entropy += float(shape(filtered_dataset)[0])/m*\
                self._calc_shannon_entropy(filtered_labels)
        return conditional_entropy

    @abstractmethod
    def _criterion(self,dataset,labels,feat_ind):
        """
        the criterion for decision tree
        for example, ID3 is entropy gain and entropy gain rate for C4.5

        Returns
        -------
        """
        pass

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
        _,n = shape(dataset);maximum_criterion= -inf;best_ind = -1
        for feat_ind in range(n):
            if self._criterion(dataset,labels,feat_ind)>maximum_criterion:
                maximum_criterion = self._criterion(dataset,labels,feat_ind)
                best_ind = feat_ind

        return best_ind,maximum_criterion

    def _create_tree(self,dataset,labels):
        """create decision tree based on criterion

        Parameters
        ----------
        dataset: array_like, shape = (n_samples,n_features)

        labels: array_like, shape = (n_samples,1)

        Returns
        -------
        the unpruned tree stored in dict

        """
        tree = {}
        num_labels,most_label,max_count = self._check_labels(labels)
        #labels are the same kind, return the specific label
        if num_labels==1:
            return most_label
        #empty feature set or all the same feature combination, return the most occured label
        feat_nums,single_feat_comb = self._check_features(dataset)
        if feat_nums==0 or single_feat_comb:
            return most_label
        #choose the best feature
        best_ind, maximum_criterion = self._choose_best_feat(dataset,labels)
        #criterion is less than epsilon, return most occuerd label
        if maximum_criterion<self.epsilon:
            return most_label
        for feat_val in set(dataset[:,best_ind].T.tolist()[0]):
            filtered_dataset,filtered_labels = self._filter_dataset_by_feat_val(dataset,labels,best_ind,feat_val)
            tree[feat_val]=self._create_tree(filtered_dataset,filtered_labels)

        return {self.feat_labels[best_ind]:tree}


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
        tree = self._create_tree(X,y)
        print(tree)

    def predict(self,X):
        pass

    '''
    @abstractmethod
    def plot_tree(self):
        pass
    '''



class ID3(BaseDecisionTree):
    def __init__(self,max_depth,max_leafs,epsilon=0.001):
        super(ID3,self).__init__(
            max_depth=max_depth,
            max_leafs=max_leafs,
            epsilon=epsilon)

    def _criterion(self,dataset,labels,feat_ind):
        """information gain for ID3 model
        which is entropy(labels)-conditionalentropy(labels,feat_ind)

        Parameters
        ----------
        dataset:array_like, shape = (n_samples,n_features)

        labels: array_like, shape = (n_samples,1)

        feat_ind: feature index, int

        Returns
        -------
        information gain, float

        """
        return super()._calc_shannon_entropy(labels)-super()._calc_conditional_entropy(dataset,labels,feat_ind)



class C45(BaseDecisionTree):
    def __init__(self,max_depth,max_leafs,epsilon=0.001):
        super(C45,self).__init__(
            max_depth=max_depth,
            max_leafs=max_leafs,
            epsilon=epsilon)

    def _criterion(self,dataset,labels,feat_ind):
        """information gain rate for C4.5 model
        which is (entropy(labels)-conditionalentropy(labels,feat_ind))/entropy(feat_ind)
        note that we skip this feature by return inf when the feature contains the same feature value

        Parameters
        ----------
        dataset:array_like, shape = (n_samples,n_features)

        labels: array_like, shape = (n_samples,1)

        feat_ind: feature index, int

        Returns
        -------
        information gain, float

        """
        if float(super()._calc_shannon_entropy(dataset[:, feat_ind]))==0.0:
            return inf
        return (super()._calc_shannon_entropy(labels)-super()._calc_conditional_entropy(dataset,labels,feat_ind))\
               /float(super()._calc_shannon_entropy(dataset[:,feat_ind]))




"""
warining: abandoned 
"""
class ID3Tree:
    def __init__(self,dataset,labels,epsilon = 0.00001):
        self.epsilon = epsilon
        self.dataset = dataset
        self.featLabels  = labels

    #calc Shannon Entropy
    def __calcShannonEntropy(self,dataset):
        numEntries = len(dataset)
        labelCounts  = {}
        for featVec in dataset:
            currentLabel = featVec[-1]
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel]=0
            labelCounts[currentLabel]+=1
        shannonEntropy = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key])/numEntries
            shannonEntropy-=prob*math.log(prob,2)
        return shannonEntropy

    #given feature axis and value ,return the selected split dataset
    def __splitDatsetByAaixValue(self,dataset,axis,value):
        reducedDataset = []
        for featVec in dataset:
            if featVec[axis]==value:
                reduceFeatVec = featVec[:axis]
                reduceFeatVec.extend(featVec[axis:])
                reducedDataset.append(reduceFeatVec)
        return reducedDataset

    #calc Emperical Conditional Entropy,given feature axis
    def __calcEmpConEnt(self, dataset, axis):
        empConEnt = 0.0;featCount = {}
        for featVec in dataset:
            featValue = featVec[axis]
            if featValue not in featCount:
                featCount[featValue]=0
            featCount[featValue]+=1
        for featValue in featCount.keys():
            splitDataset = self.__splitDatsetByAaixValue(dataset,axis,featValue)
            prob = float(featCount[featValue])/len(dataset)
            empConEnt += prob*self.__calcShannonEntropy(splitDataset)
        return empConEnt

    #given feature axis, return information gain
    def __calcInformationGain(self,dataset,axis):
        return self.__calcShannonEntropy(dataset)-self.__calcEmpConEnt(dataset,axis)

    #choose the one feature by information gain, return -1 if information gain is less than epsilon
    def __chooseFeat(self,dataset):
        entropy = self.__calcShannonEntropy(dataset)
        greatestInformationGain  = -1.0
        theFeatInd = -1;
        for featVec in dataset:
            for featIndex in range(len(featVec[:-1])):
                if self.__calcInformationGain(dataset,featIndex)>greatestInformationGain:
                    greatestInformationGain = self.__calcInformationGain(dataset,featIndex)
                    theFeatInd = featIndex

        if greatestInformationGain >=self.epsilon:
            return theFeatInd
        return -1

    #create Tree
    def __createTree(self,dataset):
        while True:
            flag,currentLabel = self.__countLabels(dataset)
            # no features left or one label left or little information gain,
            #  it's left node and return the max label
            if len(dataset[0])==1 or flag==True:
                return currentLabel

            featInd = self.__chooseFeat(dataset)
            if featInd ==-1:
                return currentLabel
            tree = {}
            for featVal in self.__countFeatVal(dataset,featInd):
                splitDataset = self.__splitDatsetByAaixValue(dataset,featInd,featVal)
                tree[featVal] = self.__createTree(splitDataset)
            return {self.featLabels[featInd]:tree}

    #create tree
    def createTree(self):
        return self.__createTree(self.dataset)


    #count feature values
    def __countFeatVal(self,dataset,axis):
        featCount = {}
        for featVec in dataset:
            if featVec[axis] not in featCount:
                featCount[featVec[axis]]=0
            featCount[featVec[axis]]+=1
        return featCount.keys()


    #count the labels and return true if only one label left
    def __countLabels(self,dataset):
        labelCount = {}
        for featVec in dataset:
            label = featVec[-1]
            if label not in labelCount:
                labelCount[label]=0
            labelCount[label]+=1
        if len(labelCount)==1:
            return True,list(labelCount.keys())[0]
        else:
            return False,sorted(labelCount.items(),key=lambda x:x[1])[0][0]



