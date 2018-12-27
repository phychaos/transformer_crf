#!/usr/bin python3
# -*- coding: utf-8 -*-
# @Time    : 18-12-27 上午9:54
# @Author  : 林利芳
# @File    : config.py

import os

PATH = os.getcwd()
TRAIN_DATA = os.path.join(PATH, 'data/train.char.bmes')
DEV_DATA = os.path.join(PATH, 'data/dev.char.bmes')
TEST_DATA = os.path.join(PATH, 'data/test.char.bmes')

TOKEN_DATA = os.path.join(PATH, 'data/vocab/token.json')
TAG_DATA = os.path.join(PATH, 'data/vocab/tag.json')
TOKEN_FRE_DATA = os.path.join(PATH, 'data/token_fre.data')
TAG_FRE_DATA = os.path.join(PATH, 'data/tag_fre.data')

logdir = os.path.join(PATH, 'logdir')
