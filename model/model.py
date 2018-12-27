#!/usr/bin python3
# -*- coding: utf-8 -*-
# @Time    : 18-12-26 下午5:49
# @Author  : 林利芳
# @File    : model.py
import tensorflow as tf
from config.hyperparams import HyperParams as hp
from model.module.modules import *


class Model(object):
	def __init__(self, vocab_size, num_tags, is_training=True):
		self.vocab_size = vocab_size
		self.num_tags = num_tags
		self.is_training = is_training
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.x = tf.placeholder(tf.int32, shape=(None, hp.max_len))
			self.y = tf.placeholder(tf.int32, shape=(None, hp.max_len))
			self.seq_lens = tf.placeholder(dtype=tf.int32, shape=[None])
			self.global_step = tf.train.create_global_step()
			
			# layers
			outputs = self.multi_head_attention()
			# outputs = self.rnn_layer(outputs)
			self.logits = self.logits_layer(outputs)
			self.loss, self.transition = self.crf_layer()
			self.train_op = self.optimize()
	
	# tf.summary.scalar('crf_loss', self.loss)
	# merged = tf.summary.merge_all()
	
	def rnn_layer(self, embed):
		with tf.variable_scope("lstm"):
			# embed = embedding(self.x, vocab_size=self.vocab_size, num_units=hp.num_units, scale=True, scope="enc_embed")
			cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hp.num_units)
			b_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hp.num_units)
			(fw_output, bw_output), _ = tf.nn.bidirectional_dynamic_rnn(cell, b_cell, embed,
																		sequence_length=self.seq_lens,
																		dtype=tf.float32)
			output = tf.add(fw_output, bw_output)
			return output
	
	def multi_head_attention(self):
		with tf.variable_scope("encoder"):
			# Embedding
			embed = embedding(self.x, vocab_size=self.vocab_size, num_units=hp.num_units, scale=True, scope="enc_embed")
			
			# Positional Encoding
			embed += positional_encoding(self.x, num_units=hp.num_units, zero_pad=False, scale=False, scope="enc_pe")
			
			# Dropout
			embed = tf.layers.dropout(embed, rate=hp.dropout_rate, training=tf.convert_to_tensor(self.is_training))
			output = self.multi_head_block(embed, embed)
			return output
	
	def multi_head_block(self, query, key, causality=False):
		"""
		多头注意力机制
		:param query:
		:param key:
		:param causality:
		:return:
		"""
		for i in range(hp.num_blocks):
			with tf.variable_scope("num_blocks_{}".format(i)):
				# multi head Attention ( self-attention)
				query = multihead_attention(
					queries=query, keys=key, num_units=hp.num_units, num_heads=hp.num_heads,
					dropout_rate=hp.dropout_rate, is_training=self.is_training, causality=causality,
					scope="self_attention")
				# Feed Forward
				# query = feedforward(query, num_units=[4 * hp.num_units, hp.num_units])
		return query
	
	def logits_layer(self, outputs):
		"""
		logits
		:param outputs:
		:return:
		"""
		w = tf.get_variable(name='w', dtype=tf.float32, shape=[hp.num_units, self.num_tags])
		b = tf.get_variable(name='b', dtype=tf.float32, shape=[self.num_tags])
		
		outputs = tf.reshape(outputs, [-1, hp.num_units])
		logits = tf.matmul(outputs, w) + b
		logits = tf.reshape(logits, [-1, hp.max_len, self.num_tags])
		return logits
	
	def crf_layer(self):
		log_likelihood, transition = tf.contrib.crf.crf_log_likelihood(self.logits, self.y, self.seq_lens)
		loss = tf.reduce_mean(-log_likelihood)
		return loss, transition
	
	def optimize(self):
		"""
		优化器
		:return:
		"""
		# optimize = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
		# param = tf.trainable_variables()
		# gradients = tf.gradients(self.loss, param)
		# clip_grad, clip_norm = tf.clip_by_global_norm(gradients, hp.clip)
		# train_op = optimize.apply_gradients(zip(clip_grad, param), global_step=self.global_step)
		optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
		train_op = optimizer.minimize(self.loss, global_step=self.global_step)
		return train_op
	
	def predict(self, logits, transition, seq_lens):
		pre_seqs = []
		for score, seq_len in zip(logits, seq_lens):
			pre_seq, pre_score = tf.contrib.crf.viterbi_decode(score[:seq_len], transition)
			pre_seqs.append(pre_seq)
		return pre_seqs
