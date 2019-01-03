#!/usr/bin python3
# -*- coding: utf-8 -*-
# @时间   : 18-12-11 下午5:44
# @作者   : Lin lifang
# @文件   : feature.py
from utils.utils import read_template
import numpy as np


class Feature(object):
	def __init__(self, fd=5):
		self.fd = fd
		self.fss = None
		self.bf_size = 0
		self.uf_size = 0
		self.f_size = 0
		self.num_k = 0
		self.node_obs = dict()
		self.edge_obs = dict()
		self.oby_dict = dict()
		self.node_fs = []
		self.edge_fs = []
		self.tp_list = [
			['U00', ['-2', '0']],
			['U01', ['-1', '0']],
			['U02', ['0', '0']],
			['U03', ['1', '0']],
			['U04', ['2', '0']],
			['U05', ['-2', '0'], ['-1', '0'], ['0', '0']],
			['U06', ['-1', '0'], ['0', '0'], ['1', '0']],
			['U07', ['0', '0'], ['1', '0'], ['2', '0']],
			['U08', ['-1', '0'], ['0', '0']],
			['U09', ['0', '0'], ['1', '0']],
			['B'], ]

	def process_features(self, texts):
		"""
		特征提取
		:param texts: 序列文本 [[['你',],['好',]],[['你',],['好',]]]
		:return:
		"""
		print("特征提取...")
		uf_obs = dict()
		bf_obs = dict()

		for text in texts:
			seq_uf, seq_bf = self.feature_vector(text)
			for loc_id, (loc_uf, loc_bf) in enumerate(zip(seq_uf, seq_bf)):
				for fs in loc_bf:
					fs_id = bf_obs.get(fs)
					bf_obs[fs] = fs_id + 1 if fs_id is not None else 1
				for fs in loc_uf:
					fs_id = uf_obs.get(fs)
					uf_obs[fs] = fs_id + 1 if fs_id is not None else 1

		node_fs = [key for key, v in sorted(uf_obs.items(), key=lambda x: x[1], reverse=True) if v >= self.fd]
		edge_fs = [key for key, v in sorted(bf_obs.items(), key=lambda x: x[1], reverse=True) if v >= self.fd]
		self.node_obs = {key: kk * self.num_k for kk, key in enumerate(node_fs)}
		self.edge_obs = {key: kk * self.num_k * self.num_k for kk, key in enumerate(edge_fs)}

		self.uf_size = len(node_fs) * self.num_k
		self.bf_size = len(edge_fs) * self.num_k * self.num_k
		self.f_size = self.uf_size + self.bf_size
		print("B 特征:\t{}\nU 特征:\t{}\n总特征:\t{}\n".format(self.bf_size, self.uf_size, self.f_size))

	def feature_vector(self, text, init=True):
		"""
		特征序列化
		:param text:
		:param init:
		:return:
		"""
		seq_bf = []
		seq_uf = []
		for loc_id in range(len(text)):
			loc_uf, loc_bf = self.expand_observation(text, loc_id, init)
			seq_bf.append(loc_bf)
			seq_uf.append(loc_uf)
		return seq_uf, seq_bf

	def expand_observation(self, sentence, loc_id, init=True):
		"""
		expend the observation at loc_id for sequence
		:param sentence: 字符序列
		:param loc_id: 字符在sentence的位置序号
		:param init: 是否初始化
		:return:
		"""
		loc_uf = []
		loc_bf = []
		for tp in self.tp_list:
			fs = tp[0]
			for li in tp[1::]:
				row = loc_id + int(li[0])
				col = int(li[1])
				if len(sentence) > row >= 0:
					if len(sentence[row][col]) > col >= 0:
						fs += ":" + sentence[row][col]
				else:
					fs += ':B' + li[0]
			if fs[0] == "U":
				if init:
					loc_uf.append(fs)
				else:
					fs_id = self.node_obs.get(fs)
					if fs_id is not None:
						loc_uf.append(fs_id)
			if fs[0] == "B":
				if init:
					loc_bf.append(fs)
				else:
					fs_id = self.edge_obs.get(fs)
					if fs_id is not None:
						loc_bf.append(fs_id)
		return loc_uf, loc_bf

	def cal_observe_on(self, texts, init=False):
		"""
		获取文本特征 [[['U:你','U:你:好'],['U:你','U:你:好'],[]],[],[]] =[[[145,456,566],[3455,]],[]]
		:param texts:
		:param init:
		:return:
		"""
		self.node_fs = []
		self.edge_fs = []
		for text in texts:
			seq_uf, seq_bf = self.feature_vector(text, init)
			self.node_fs.append(seq_uf)
			self.edge_fs.append(seq_bf)
		return self.node_fs, self.edge_fs

	def cal_fss(self, labels, y0):
		"""
		统计特征数量 每个特征对应 num_k 个特征
		:param labels: 标签
		:param y0: 起始值0
		:return:
		"""
		self.fss = np.zeros((self.f_size,))
		fss_b = self.fss[0:self.bf_size]
		fss_u = self.fss[self.bf_size:]
		for seq_id, label in enumerate(labels):
			y_p = y0
			for loc_id, y in enumerate(label):
				for fs_id in self.node_fs[seq_id][loc_id]:
					fss_u[fs_id + y] += 1.0
				for fs_id in self.edge_fs[seq_id][loc_id]:
					fss_b[fs_id + y_p * self.num_k + y] += 1.0
				y_p = y

	def save_feature(self):
		result = ['#CRF Feature Templates.\n\n']
		for tp in self.tp_list:
			feature = tp[0] + ':'
			for start, end in tp[1:]:
				feature += '%x[' + start + ',' + end + ']'
			result.append(feature)
		result.append('\n\n#U')
		u_feature = list(sorted(self.node_obs.keys(), key=lambda x: x))
		result.extend(u_feature)
		with open('feature.txt', 'w', encoding='utf-8') as fp:
			fp.write('\n'.join(result))

	def process_state(self, labels):
		"""
		状态预处理
		:param labels:
		:return:
		"""
		new_label = []
		oby_id = 0
		for sentence in labels:
			s_label = []
			for label in sentence:
				label_id = self.oby_dict.get(label)
				if label_id is None:
					label_id = oby_id
					self.oby_dict[label] = oby_id
					oby_id += 1
				s_label.append(label_id)
			new_label.append(s_label)
		self.num_k = len(self.oby_dict)
		return new_label

	def __call__(self, texts, labels, template_file, y0=0, *args, **kwargs):
		if template_file:
			self.tp_list = read_template(template_file)
		self.seq_lens = [len(x) for x in labels]
		labels = self.process_state(labels)
		self.process_features(texts)
		self.cal_observe_on(texts)
		self.cal_fss(labels, y0)
		self.save_feature()
