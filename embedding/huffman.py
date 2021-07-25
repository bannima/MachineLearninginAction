#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: huffman.py
Description: huffman tree
Author: Barry Chow
Date: 2021/3/16 5:14 PM
Version: 0.1
"""
from numpy import random
from copy import copy


class HuffmanTree():

    class Node():
        def __init__(self,id=None,freq=None,vector_size=100):
            self.id = id
            self.freq = freq
            self.params = random.rand(1,vector_size)-0.5

            self.parent = None
            self.lchild = None
            self.rchild = None

    def __init__(self,id2freq,hidden_size = 100):
        self.hidden_size = hidden_size

        self._root = self._construct_tree(id2freq)

        self._mappings = dict()
        self._pre_order_traverse(self.root,path=[],mappings=self._mappings)
        #print(self._mappings)


    def _construct_tree(self, id2freq):

        node_sets = [HuffmanTree.Node(id=id,freq=freq) for id,freq in id2freq.items()]
        while(len(node_sets)>1):
            node_sets = sorted(node_sets,key=lambda asv:asv.freq,reverse=True)
            mini_1 = node_sets[-1]
            mini_2 = node_sets[-2]
            node = HuffmanTree.Node(id=None,freq=mini_1.freq+mini_2.freq)
            if mini_1.freq>mini_2.freq:
                node.lchild = mini_1
                node.rchild = mini_2
            else:
                node.lchild = mini_2
                node.rchild = mini_1

            mini_1.parent = node
            mini_2.parent = node

            #update node sets
            node_sets.remove(mini_1)
            node_sets.remove(mini_2)
            node_sets.append(node)

        return node_sets[0]


    def _pre_order_traverse(self, root, path, mappings):
        '''先序遍历'''

        #左右子树均为空，叶子节点
        if not root.lchild and not root.rchild:
            mappings[root.id] = copy(path)
            return
        #左子树不为空
        if root.lchild:
            self._pre_order_traverse(root.lchild, path + [1], mappings)

        if root.rchild:
            self._pre_order_traverse(root.rchild,path+[0],mappings)


    #给定word的id，返回其
    def paths(self,word_id):
        return self._mappings[word_id]


    @property
    def root(self):
        return self._root

