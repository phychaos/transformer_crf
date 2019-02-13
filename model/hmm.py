#!/usr/bin python3
# -*- coding: utf-8 -*-
# @Time    : 19-2-13 上午10:30
# @Author  : 林利芳
# @File    : hmm.py
import numpy as np
import collections
import pickle

from utils.utils import save_model, load_model


class HMM(object):
	def __init__(self, alpha=1e-8):
		self.alpha = alpha
		self.word2id = {"UNK": 0}
		self.label2id = {'O': 0}
		self.initial_proba = None
		self.observation_proba = None
		self.transition_proba = None
	
	def fit(self, x, y, model_file=None):
		self.create_vocab(x)
		self.create_label(y)
		self.count_likelihood(x, y)
		if model_file:
			self.save_model(model_file)
	
	def count_likelihood(self, x, y):
		word_num = len(self.word2id)
		label_num = len(self.label2id)
		self.initial_proba = np.zeros(label_num)
		self.observation_proba = np.zeros((label_num, word_num))
		self.transition_proba = np.zeros((label_num, label_num))
		for sentence, labels in zip(x, y):
			pre_label_id = 0
			for ii, (word, label) in enumerate(zip(sentence, labels)):
				word_id = self.word2id.get(word, 0)
				label_id = self.label2id.get(label, 0)
				self.observation_proba[label_id, word_id] += 1
				if ii == 0:
					self.initial_proba[label_id] += 1
				else:
					self.transition_proba[pre_label_id, label_id] += 1
				pre_label_id = label_id
		self.initial_proba /= np.sum(self.initial_proba)
		self.observation_proba /= np.sum(self.observation_proba, axis=1)[:, np.newaxis]
		self.transition_proba /= np.sum(self.transition_proba, axis=1)[:, np.newaxis]
		
		self.initial_proba += self.alpha
		self.observation_proba += self.alpha
		self.transition_proba += self.alpha
	
	def predict(self, x, model_file=None):
		if model_file:
			self.load_model(model_file)
		id2label = {value: key for key, value in self.label2id.items()}
		y = []
		for sentence in x:
			labels = self.tagging(sentence)
			labels = [id2label.get(idx, 'O') for idx in labels]
			y.append(labels)
		return y
	
	def tagging(self, sentence):
		matrix_alpha = np.zeros((len(sentence), len(self.label2id)))
		matrix_index = []
		# 正向维特比算法
		for ii, word in enumerate(sentence):
			word_id = self.word2id.get(word, 0)
			observation_proba = self.observation_proba[:, word_id]
			if ii == 0:
				matrix_alpha[0] = self.initial_proba * observation_proba
			else:
				mt = matrix_alpha[ii - 1][:, np.newaxis] * self.transition_proba * observation_proba
				matrix_alpha[ii] = mt.max(axis=0)
				matrix_index.append(mt.argmax(axis=0))
		# 反向追踪路径
		ty = matrix_alpha[-1].argmax()
		max_state = [ty]
		for loc_index in reversed(matrix_index):
			ty = loc_index[ty]
			max_state.append(ty)
		return max_state[::-1]
	
	def create_vocab(self, x):
		words = [word for sentence in x for word in sentence]
		counter = collections.Counter(words).most_common()
		start_id = 1
		for word, _ in counter:
			if word in self.word2id:
				continue
			self.word2id[word] = start_id
			start_id += 1
	
	def create_label(self, y):
		labels = [label for sentence in y for label in sentence]
		counter = collections.Counter(labels).most_common()
		start_id = 1
		for label, _ in counter:
			if label in self.label2id:
				continue
			self.label2id[label] = start_id
			start_id += 1
	
	def save_model(self, model_file):
		data = {
			"word2id": self.word2id,
			"label2id": self.label2id,
			"initial_proba": self.initial_proba,
			"observation_proba": self.observation_proba,
			"transition_proba": self.transition_proba,
		}
		save_model(model_file, data)
	
	def load_model(self, model_file):
		data = load_model(model_file)
		self.word2id = data['word2id']
		self.label2id = data['label2id']
		self.initial_proba = data['initial_proba']
		self.observation_proba = data['observation_proba']
		self.transition_proba = data['transition_proba']
