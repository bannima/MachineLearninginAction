#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: feature.py
Description: implementation of feature template and feature functions
Author: Barry Chow
Date: 2020/8/6 9:27 PM
Version: 0.1
"""


class FeatureTemplate(object):
    '''
    implementation of feature template class
    '''

    def __init__(self, type, x_indexs):
        '''

        Parameters
        ----------
        type: U for Unigram and B for Bigram
        x_indexs: the indexs of related X input
        '''
        self.type = type
        self.x_indexs = x_indexs


class FeatureBuilder(object):
    '''
    implementation of generate feature functions
    '''

    def __init__(self, dataset, tags, template_builder):
        '''
        build specific feature function set according to given dataset and tags

        Parameters
        ----------
        dataset: input dataset
        tags: corresponding tags for input dataset
        template_builder: template of feature functions
        '''
        self.dataset = dataset
        self.tags = tags
        self.feature_templates = template_builder.feature_templates
        self._build()

    def _build(self):
        '''
        generate feature functions according to data, tags, and feature templates

        Returns
        -------

        '''
        self.tag_set = set()
        self.features_set = {}
        feature_vectors = []
        for sentence, tag in zip(self.dataset, self.tags):
            feature_vector = []

            for ind in range(len(tag)):
                # tag set
                self.tag_set.add(tag[ind])

                for template in self.feature_templates:
                    prior_state = tag[ind - 1] if ind - 1 >= 0 else '#'
                    current_state = tag[ind]
                    # feature_function_num,feature_id = self._is_match(sentence, tag, ind, template)
                    feature_function_num, feature_id = self._is_match(sentence, prior_state, current_state, ind,
                                                                      template)

                    # save if not exist in feature function set
                    if feature_function_num == -1:
                        self.features_set[feature_id] = len(self.features_set)

                        feature_vector.append(self.features_set[feature_id])

            feature_vectors.append(feature_vector)

        self.feature_nums = len(self.features_set)
        print("######num of features: " + str(self.feature_nums))

        return feature_vectors

    def _is_match(self, sentence, prior_state, current_state, ind, template):
        '''

        Parameters
        ----------
        sentence: input word sequence
        prior_state:corresponding tags of sentence in position index -1
        current_state: corresponding tags of sentence in position index
        ind: index in the sentence
        template: given feature template

        Returns: feature function num and feature id
        in detai:
                 feature function num : -1 means matched but not exist in feature set
                                       -2 means not matched
                                       other positive interger: the specific feature function num in feature set
        -------

        '''
        # out of array index
        if template.x_indexs[0] + ind < 0 or template.x_indexs[-1] + ind >= len(sentence):
            return (-2, None)

        # Unigram features
        if template.type.startswith('U'):
            # state = tag[ind]
            state = current_state

        # Bigram features
        elif template.type.startswith('B'):
            # state = "".join(tag[i] for i in [ind - 1, ind])
            if prior_state == '#':
                return (-2, None)
            state = prior_state + current_state

        # unknown type
        else:
            return (-2, None)

        # save feature ids
        x_values = "".join([sentence[i + ind] for i in template.x_indexs])

        feature_id = template.type + state + x_values

        return (self.features_set[feature_id], feature_id) if feature_id in self.features_set else (-1, feature_id)

    def match_with_index(self, sentence, prior_state, current_state, ind):
        '''
        given word sentence and prior state tag ,current state tag ,current index,
        return the hit feature function id list

        Parameters
        ----------
        sentence
        prior_state: prior state tag
        current_state: current state tag
        ind: current index in sentence

        Returns
        -------

        '''
        feature_vector = []

        for template in self.feature_templates:
            # feature funciton numbers
            feature_function_num, feature_id = self._is_match(sentence, prior_state, current_state, ind, template)
            # ignore -1 and -2 case, only non-negative integer means the matched feature function number.
            if feature_function_num >= 0:
                feature_vector.append(feature_function_num)

        return feature_vector

    def match(self, sentence, tag):
        '''
        given word sentence and corresponding tags, return the hit feature function id list

        Parameters
        ----------
        sequence
        tags

        Returns
        -------

        '''
        feature_vector = []

        for ind in range(len(tag)):
            for template in self.feature_templates:
                # means not exist, just skip this B type feature function
                prior_state = tag[ind - 1] if ind - 1 >= 0 else '#'
                current_state = tag[ind]
                # feature funciton numbers
                feature_function_num, feature_id = self._is_match(sentence, prior_state, current_state, ind, template)
                # feature_function_num,feature_id = self._is_match(sentence, tag, ind, template)

                # ignore -1 and -2 case, only non-negative integer means the matched feature function number.
                if feature_function_num >= 0:
                    feature_vector.append(feature_function_num)

        return feature_vector
