#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: test_template.py
Description: 
Author: Barry Chow
Date: 2020/8/6 9:23 PM
Version: 0.1
"""
from crf import TemplateBuilder


class TestTemplate(object):
    def test_read_template(self):
        temp = TemplateBuilder("feature.template")
