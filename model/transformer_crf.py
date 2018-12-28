#!/usr/bin python3
# -*- coding: utf-8 -*-
# @Time    : 18-12-26 下午5:49
# @Author  : 林利芳
# @File    : transformer_crf.py
import tensorflow as tf
from config.hyperparams import HyperParams as hp
from model.module.modules import *
from model.module.rnn import ForgetLSTMCell, IndRNNCell


class TransformerCRFModel(object):
	def __init__(self, vocab_size, num_tags, is_training=True, seg='rnn'):
		self.vocab_size = vocab_size
		self.num_tags = num_tags
		self.is_training = is_training
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.x = tf.placeholder(tf.int32, shape=(None, hp.max_len))
			self.y = tf.placeholder(tf.int32, shape=(None, hp.max_len))
			self.seq_lens = tf.placeholder(dtype=tf.int32, shape=[None])
			self.global_step = tf.train.create_global_step()
			
			# layers embedding multi_head_attention rnn
			outputs = embedding(self.x, vocab_size=self.vocab_size, num_units=hp.num_units, scale=True,
								scope="enc_embed")
			outputs = self.rnn_layer(outputs, seg)
			outputs = self.encoder(outputs)
			
			self.logits = self.logits_layer(outputs)
			self.loss, self.transition = self.crf_layer()
			self.train_op = self.optimize()
	
	def rnn_layer(self, embed, seg):
		"""
		创建双向RNN层
		:param embed:
		:param seg: LSTM GRU F-LSTM, IndRNN
		:return:
		"""
		if seg == 'LSTM':
			fw_lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=hp.num_units)
			bw_lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=hp.num_units)
		
		elif seg == 'GRU':
			fw_lstm = tf.nn.rnn_cell.GRUCell(num_units=hp.num_units)
			bw_lstm = tf.nn.rnn_cell.GRUCell(num_units=hp.num_units)
		elif seg == 'F-LSTM':
			fw_lstm = ForgetLSTMCell(num_units=hp.num_units)
			bw_lstm = ForgetLSTMCell(num_units=hp.num_units)
		elif seg == 'IndRNN':
			fw_lstm = IndRNNCell(num_units=hp.num_units)
			bw_lstm = IndRNNCell(num_units=hp.num_units)
		else:
			fw_lstm = tf.nn.rnn_cell.BasicRNNCell(num_units=hp.num_units)
			bw_lstm = tf.nn.rnn_cell.BasicRNNCell(num_units=hp.num_units)
		# 双向rnn
		(fw_output, bw_output), _ = tf.nn.bidirectional_dynamic_rnn(fw_lstm, bw_lstm, embed,
																	sequence_length=self.seq_lens,
																	dtype=tf.float32)
		# 合并双向rnn的output batch_size * max_seq * (hidden_dim*2)
		outputs = tf.add(fw_output, bw_output)
		return outputs
	
	def encoder(self, embed):
		with tf.variable_scope("Transformer_Encoder"):
			# Positional Encoding
			embed += positional_encoding(self.x, num_units=hp.num_units, zero_pad=False, scale=False, scope="enc_pe")
			# Dropout
			embed = tf.layers.dropout(embed, rate=hp.dropout_rate, training=tf.convert_to_tensor(self.is_training))
			output = self.multi_head_block(embed, embed)
			return output
	
	def decoder(self, enc):
		"""
		解码层
		:param enc:
		:return:
		"""
		with tf.variable_scope("Transformer_Decoder"):
			# Embedding
			dec = embedding(self.y, vocab_size=self.num_tags, num_units=hp.num_units, scale=True, scope="dec_embed")
			
			# Positional Encoding
			dec += positional_encoding(self.y, num_units=hp.num_units, zero_pad=False, scale=False, scope="dec_pe")
			# Dropout
			dec = tf.layers.dropout(dec, rate=hp.dropout_rate, training=tf.convert_to_tensor(self.is_training))
			
			output = self.multi_head_block(dec, enc, decoding=True, causality=True)
			return output
	
	def multi_head_block(self, query, key, decoding=False, causality=False):
		"""
		多头注意力机制
		:param query:
		:param key:
		:param decoding:
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
				if decoding:
					# multi head Attention ( vanilla attention)
					query = multihead_attention(
						queries=query, keys=key, num_units=hp.num_units, num_heads=hp.num_heads,
						dropout_rate=hp.dropout_rate, is_training=self.is_training, causality=False,
						scope="vanilla_attention")
				# Feed Forward
				query = feedforward(query, num_units=[4 * hp.num_units, hp.num_units])
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
		optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
		train_op = optimizer.minimize(self.loss, global_step=self.global_step)
		return train_op
	
	def predict(self, logits, transition, seq_lens):
		pre_seqs = []
		for score, seq_len in zip(logits, seq_lens):
			pre_seq, pre_score = tf.contrib.crf.viterbi_decode(score[:seq_len], transition)
			pre_seqs.append(pre_seq)
		return pre_seqs
