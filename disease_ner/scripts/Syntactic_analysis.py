#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/5 22:40
# @Author  : XuHaolei
# @File    : Syntactic_analysis.py

import spacy
from spacy import displacy
"""
python -m spacy download en_core_web_sm
"""
# 还没写完 未来GCN用
nlp = spacy.load('en')
doc = nlp("spaCy uses the terms head and child to describe the words" )
displacy.serve(doc, style='dep')