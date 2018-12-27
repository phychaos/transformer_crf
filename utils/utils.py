#!/usr/bin python3
# -*- coding: utf-8 -*-
# @Time    : 18-12-27 上午9:47
# @Author  : 林利芳
# @File    : utils.py
from config.config import *
from collections import Counter
import json
import numpy as np

UNK = '<unk>'


def create_vocab():
	sentences, tags, _ = read_data(TRAIN_DATA)
	token = merge_list(sentences)
	tags = merge_list(tags)
	counter = [(UNK, 100000)] + Counter(token).most_common()
	counter_tag = Counter(tags).most_common()
	word2id = {word: _id for _id, (word, fre) in enumerate(counter)}
	id2word = {str(_id): word for word, _id in word2id.items()}
	
	tag2id = {tag: _id for _id, (tag, fre) in enumerate(counter_tag)}
	id2tag = {str(_id): tag for tag, _id in tag2id.items()}
	
	save_json(TOKEN_DATA, [word2id, id2word])
	save_json(TAG_DATA, [tag2id, id2tag])
	
	save_data(TOKEN_FRE_DATA, counter)
	save_data(TAG_FRE_DATA, counter_tag)


def read_data(filename):
	sentences = []
	tags = []
	seq_lens = []
	with open(filename, 'r') as fp:
		sentence = []
		tag = []
		for line in fp.readlines():
			token = line.strip().split()
			if len(token) == 2:
				sentence.append(token[0])
				tag.append(token[1])
			elif len(token) == 0 and len(sentence) > 0:
				assert len(sentence) == len(tag)
				seq_lens.append(len(sentence))
				sentences.append(sentence)
				tags.append(tag)
				sentence = []
				tag = []
		if sentence:
			assert len(sentence) == len(tag)
			seq_lens.append(len(sentence))
			sentences.append(sentence)
			tags.append(tag)
	return sentences, tags, seq_lens


def merge_list(data_list):
	data = []
	for sub_data in data_list:
		if isinstance(sub_data, list):
			data.extend(merge_list(sub_data))
		elif isinstance(sub_data, str):
			data.append(sub_data)
	return data


def save_json(filename, data):
	with open(filename, 'w') as fp:
		json.dump(data, fp)


def save_data(filename, data):
	with open(filename, 'w', encoding='utf8') as fp:
		for word, fre in data:
			fp.writelines(word + '\t' + str(fre) + '\n')


def load_data(filename):
	with open(filename, 'rb') as fp:
		data = json.load(fp)
	return data


def generate_data(filename, word2id, tag2id, max_len=50):
	_sentences, _tags, _seq_lens = read_data(filename)
	
	sentences = []
	tags = []
	seq_lens = []
	for _sentence, seq_len in zip(_sentences, _seq_lens):
		sentence = [word2id.get(word, 0) for word in _sentence]
		if seq_len <= max_len:
			sentence += [0] * (max_len - seq_len)
			seq_lens.append(seq_len)
			sentences.append(sentence)
	default = tag2id['O']
	for _tag, seq_len in zip(_tags, _seq_lens):
		tag = [tag2id.get(label, default) for label in _tag]
		if seq_len <= max_len:
			tag += [default] * (max_len - seq_len)
			tags.append(tag)
	
	return sentences, tags, seq_lens, _sentences, _tags


def batch_data(x, y, seq_lens, batch_size):
	total_batch = len(seq_lens) // batch_size
	
	for ii in range(total_batch):
		start, end = ii * batch_size, (ii + 1) * batch_size
		
		x_batch = np.array(x[start:end], dtype=np.int32)
		y_batch = np.array(y[start:end], dtype=np.int32)
		len_batch = np.array(seq_lens[start:end], dtype=np.int32)
		yield x_batch, y_batch, len_batch
