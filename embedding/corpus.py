
#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
FileName: corpus.py
Description:
Author: Barry Chow
Date: 2021/3/16 5:14 PM
Version: 0.1
"""
import os
import json
from tqdm import tqdm
from collections import Counter

class Preprocessor():
    def __call__(self, text):
        raise NotImplementedError()

class NormalPreprocessor(Preprocessor):
    def __call__(self, text):
        assert isinstance(text,str)
        return text.lower().replace('\n','').replace('\r','')

class Corpus():
    def __init__(self,data,model='CBOW',window_size=5,preprocessor=None):

        self._id2word = None
        self._size = None
        if preprocessor is None:
            self.preprocessor = NormalPreprocessor()
        else:
            self.preprocessor = preprocessor
        #word2id,id2freq,id2word,corpus
        self.word2id,self.id2freq,self.id2word,self.corpus = self._count_freq(data)

        self.samples = self._construct_samples(self.corpus, window_size=window_size, model=model)


    #构建训练预料，w:context ,pair
    def _construct_samples(self, corpus, window_size=5, model='CBOW'):
        samples = [] #generate w:context pair
        width = int(window_size//2)
        for text in corpus:
            for i in range(len(text)):
                surrending_words = [self.word2id[text[j]] for j in range(i-width,i+width+1) if j in list(range(len(text))) and j!=i]
                center_word = [self.word2id[text[i]]]
                #generate pair w:context(w)
                if model=='CBOW':
                    samples.append([center_word,surrending_words])
                elif model=='SG':
                    samples.append([surrending_words,center_word])
                else:
                    raise RuntimeError("not recognized model {},should be CBOW or SG".format(model))
        return samples

    #统计单词的词频,建立id
    def _count_freq(self,data):
        self._word2counter = Counter()
        corpus = []
        for line in tqdm(data,desc="Loading corpus"):
            line = [self.preprocessor(t) for t in line.split(' ') if t]
            corpus.append(line)
            self._word2counter.update(line)

        word2id = dict()
        id2freq = dict()
        id2word = dict()
        cnt = 0
        for word in self._word2counter:
            freq = self._word2counter[word]
            word2id[word] = cnt
            id2word[cnt] = word
            id2freq[cnt] = freq
            cnt +=1

        return word2id,id2freq,id2word,corpus

    def __next__(self):
        for i in range(len(self.samples)):
            yield self.samples[i]

    @property
    def size(self):
        if self._size is None:
            self._size = len(self.word2id)
        return self._size


