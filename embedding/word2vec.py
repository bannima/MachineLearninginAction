#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: word2vec.py
Description: implementation of word2vec
Author: Barry Chow
Date: 2020/7/26 9:11 AM
Version: 0.1
"""
import numpy as np
from tqdm import tqdm
import pickle

from .huffman import HuffmanTree
from numpy import random
from utils import sigmod

class Word2Vec():

    def __init__(self,corpus=None,model='CBOW',type='HS',hidden_size = 100,learning_rate=0.05):
        self.type = type
        self.hidden_size = hidden_size #word vectors size
        self.learning_rate = learning_rate

        self.model =model

        if corpus is None:
            raise RuntimeError("#ERROR not corpus to train")
        self.corpus = corpus
        self.vocab_size = corpus.size

        # init parameters
        self._wordvectors = random.rand(self.vocab_size, self.hidden_size) - 0.5

        #hierarchical softmax
        if type=='HS':
            self.model = HuffmanTree(corpus.id2freq,hidden_size=hidden_size)

    def train(self,epoch=100):

        #train
        for i in tqdm(range(epoch),desc='#Training Epoch：'):
            if self.type == 'HS':
                self._train_hs()

    def _train_hs(self):
        '''CBOW,w为单个词，context为周边上下文词组'''
        for w,context in next(self.corpus):
            #w 为list
            w = w[0]

            # forward
            #w 's huffman path
            w_paths = self.model.paths(w)
            context_vectors = [0]*self.vocab_size
            for c in context:
                context_vectors[c] = 1
            # 1xV * VxH = 1xH
            context_vectors = np.matmul(np.array(context_vectors),self._wordvectors)
            e = np.zeros(self.hidden_size)
            cur_node = self.model.root
            for p in w_paths:
                if p==1:
                    cur_node = cur_node.lchild
                elif p==0:
                    cur_node = cur_node.rchild
                q = sigmod(np.dot(cur_node.params,context_vectors))
                g = self.learning_rate*(1-p-q)
                e = e+g*cur_node.params
                cur_node.params = cur_node.params +g*context_vectors

            #backward
            for c in context:
                self._wordvectors[c] = self._wordvectors[c]+e

    def most_similar(self,word,top=3):
        def similarity(vec1,vec2):
            return np.dot(vec1,vec2)
        if word not in self.corpus.word2id:
            print("# word {} not in list".format(word))
            return
        id = self.corpus.word2id[word]
        vector = self._wordvectors[id]
        similar_map = dict()
        for cnt_ in range(len(self._wordvectors)):
            vec_ = self._wordvectors[cnt_]
            similar_map[cnt_] = similarity(vector,vec_)

        similar_map = sorted(similar_map.items(),key=lambda asv:asv[1],reverse=True)
        for i in range(top):
            cnt,sim = similar_map[i]
            print('#word: {}---match: {}, sim: {}'.format(self.corpus.id2word[id],self.corpus.id2word[cnt],sim))


