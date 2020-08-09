#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: template.py
Description: implementation of parsing feature template
Author: Barry Chow
Date: 2020/8/6 8:20 PM
Version: 0.1
"""
import os

from .feature import FeatureTemplate


class TemplateBuilder(object):
    def __init__(self, filename):
        assert filename is not None
        self.filename = filename
        self._parse()

    def _parse(self):
        '''

        Returns feature templates using Feature Class
        -------

        '''
        feature_templates = []
        base_dir = os.path.dirname(__file__)
        pathfile = os.path.join(base_dir, self.filename)
        for line in open(pathfile, 'r'):
            line = line.strip()
            if line.startswith('#') or len(line) == 0:
                continue
            line = line.strip()
            type = line.split(':')[0]
            indexs = [val for val in line.split(':')[1].split('%X[') if len(val) > 0]
            indexs = [int(val.split(',')[0]) for val in indexs]
            indexs.sort()  # ascending order, ex: [1,2,3]
            feature_templates.append(FeatureTemplate(type, indexs))
        self.feature_templates = feature_templates
