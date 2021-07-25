#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: test_word2vec.py
Description: 
Author: Barry Chow
Date: 2021/3/16 8:45 PM
Version: 0.1
"""
# 引入 word2vec
from gensim.models import word2vec

# 引入日志配置
import logging

class TestWordVectros():
    def test_gensim_word2vec(self):


        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        # 引入数据集
        raw_sentences = ["the quick brown fox jumps over the lazy dogs", "yoyoyo you go home now to sleep"]

        # 切分词汇
        sentences = [s.split() for s in raw_sentences]
        #sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
        # 构建模型
        model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=3, size=100)

        model.save('model')
        wordvectors = model.wv

        # 进行相关性比较
        say_vector = model['the']  # get vector for word


    def test_myself_word2vec(self):
        from ..word2vec import Word2Vec
        from ..corpus import Corpus
        from datasets.base import loadAGNewsTestCorpus
        data = loadAGNewsTestCorpus()[:1000]
        corpus = Corpus(data,model='CBOW')
        word2vec = Word2Vec(corpus=corpus, type='HS', hidden_size=100)
        word2vec.train(epoch=10)
        for word in ['the','good','bad','a','text']:
            print('\n##############')
            word2vec.most_similar(word,top=3)