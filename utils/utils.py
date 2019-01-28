#!/usr/bin python3
# -*- coding: utf-8 -*-
# @Time    : 18-12-27 上午9:47
# @Author  : 林利芳
# @File    : utils.py
import pickle
import re

from config.config import *
from collections import Counter
import json
import numpy as np

UNK = '<unk>'


def create_vocab(filename):
	"""
	创建词库
	:return:
	"""
	sentences, tags, _ = read_data(filename)
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
	"""
	读取标注数据
	:param filename:
	:return:
	"""
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
	"""
	合并列表
	:param data_list:
	:return:
	"""
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
	with open(filename, 'r') as fp:
		data = json.load(fp)
	return data


def save_model(model_file, model):
	with open(model_file, 'wb') as fp:
		pickle.dump(model, fp)


def load_model(model_file):
	"""
	加载模型
	:param model_file:
	:return:
	"""
	if not os.path.isfile(model_file):
		print("Error: 模型文件不存在!")
		return -1
	with open(model_file, 'rb') as f:
		model = pickle.load(f)
	return model


def output_file(x_test, y_test, max_ys, res_file):
	"""
	输出文件
	:param x_test:
	:param y_test:
	:param max_ys:
	:param res_file:
	:return:
	"""
	if res_file is None:
		return 0
	result = []
	for seq_id, text in enumerate(x_test):
		sentence = []
		for loc_id in range(len(text)):
			line = ""
			for x in text[loc_id]:
				line += x + '\t'
			if y_test:
				line += y_test[seq_id][loc_id] + "\t"
			line += max_ys[seq_id][loc_id] + '\n'
			sentence.append(line)
		result.append(''.join(sentence))
	with open(res_file, 'w', encoding='utf-8') as fp:
		fp.write('\n'.join(result))
	return 0


def generate_data(filename, word2id, tag2id, max_len=50):
	"""
	pad 补全<max_len 数据
	:param filename:
	:param word2id:
	:param tag2id:
	:param max_len:
	:return:
	"""
	_sentences, _tags, _seq_lens = read_data(filename)
	
	sentences = []
	tags = []
	seq_lens = []
	source_sentences = []
	source_tags = []
	for _sentence, seq_len in zip(_sentences, _seq_lens):
		sentence = [word2id.get(word, 0) for word in _sentence]
		if seq_len <= max_len:
			sentence += [0] * (max_len - seq_len)
			seq_lens.append(seq_len)
			sentences.append(sentence)
			source_sentences.append(_sentence)
	default = 0
	for _tag, seq_len in zip(_tags, _seq_lens):
		tag = [tag2id.get(label, default) for label in _tag]
		if seq_len <= max_len:
			tag += [default] * (max_len - seq_len)
			tags.append(tag)
			source_tags.append(_tag)
	return sentences, tags, seq_lens, source_sentences, source_tags


def batch_data(x, y, seq_lens, batch_size):
	"""
	生成小批量数据
	:param x:
	:param y:
	:param seq_lens:
	:param batch_size:
	:return:
	"""
	total_batch = len(seq_lens) // batch_size + 1
	
	for ii in range(total_batch):
		start, end = ii * batch_size, (ii + 1) * batch_size
		
		x_batch = np.array(x[start:end], dtype=np.int32)
		y_batch = np.array(y[start:end], dtype=np.int32)
		len_batch = np.array(seq_lens[start:end], dtype=np.int32)
		yield x_batch, y_batch, len_batch


def read_template(filename):
	"""
	读取模板特征
	:param filename: 模板文件  U08:%x[-1,0]/%x[0,0]
	:return: tp_list [[U00,[0,0],[1,0]]]
	"""
	if not os.path.isfile(filename):
		print("模板文件[{}]不存在!".format(filename))
		exit()
	tp_list = []
	pattern = re.compile(r'\[-?\d+,-?\d+\]')  # -?[0-9]*
	with open(filename, 'r', encoding='utf-8') as fp:
		for line in fp.readlines():
			line = line.strip()
			if len(line) == 0:
				continue
			if line[0] == "#":
				continue
			fl = line.find("#")
			if fl != -1:  # 移除注释
				line = line[0:fl]
			if valid_template_line(line) is False:
				continue
			fl = line.find(":")
			if fl != -1:  # just a symbol 模板符号 -> U00:%x[0,0]
				each_list = [line[0:fl]]
			else:
				each_list = [line[0]]
			
			for a in list(pattern.finditer(line)):
				loc_str = line[a.start() + 1:a.end() - 1]
				loc = loc_str.split(",")
				each_list.append(loc)
			tp_list.append(each_list)
	print("有效模板数量:", len(tp_list))
	return tp_list


def valid_template_line(line):
	if_valid = True
	if line.count("[") != line.count("]"):
		if_valid = False
	if "UuBb".find(line[0]) == -1:
		if_valid = False
	if if_valid is False:
		print("模板错误:", line)
	return if_valid


def random_param(f_size):
	theta = np.ones(f_size)
	return theta
