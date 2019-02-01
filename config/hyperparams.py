#!/usr/bin python3
# -*- coding: utf-8 -*-
# @Time    : 18-12-25 上午10:41
# @Author  : 林利芳
# @File    : hyperparams.py


class HyperParams:
	# training
	seg = 'LSTM'  # [GRU,LSTM,IndRNN,F-LSTM]
	batch_size = 128  # alias = N
	lr = 0.0001  # learning rate. In paper, learning rate is adjusted to the global step.
	num_layer = 2
	# model
	max_len = 50  # Maximum number of words in a sentence. alias = T.
	# Feel free to increase this if you are ambitious.
	min_cnt = 20  # words whose occurred less than min_cnt are encoded as <UNK>.
	num_units = 512  # alias = C
	num_blocks = 6  # number of encoder/decoder blocks
	num_epochs = 100
	num_heads = 8
	filters = [2, 3, 4, 5]
	clip = 5
	dropout_rate = 0.1
	eps = 1e-9
	sinusoid = False  # If True, use sinusoid. If false, positional embedding.
