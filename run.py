#!/usr/bin python3
# -*- coding: utf-8 -*-
# @Time    : 18-12-27 上午9:46
# @Author  : 林利芳
# @File    : run.py
from config.config import *
from model.transformer_crf import TransformerCRFModel
from model.rnn_crf import BiRnnCRF
from model.crf import CRF
from utils.metric import get_ner_fmeasure, recover_label
from utils.utils import create_vocab, load_data, generate_data, batch_data, output_file
from config.hyperparams import HyperParams as hp
from utils.utils import read_data
import tensorflow as tf
import numpy as np


def train(is_vocab=False):
	seg = 'LSTM'  # seg = [GRU,LSTM,IndRNN,F-LSTM]
	if is_vocab:
		create_vocab(TRAIN_DATA)
	word2id, id2word = load_data(TOKEN_DATA)
	tag2id, id2tag = load_data(TAG_DATA)
	x_train, y_train, seq_lens, _, _ = generate_data(TRAIN_DATA, word2id, tag2id, max_len=hp.max_len)
	x_dev, y_dev, dev_seq_lens, _, source_tag = generate_data(DEV_DATA, word2id, tag2id, max_len=hp.max_len)
	# exit()
	vocab_size = len(word2id)
	num_tags = len(tag2id)
	
	model = TransformerCRFModel(vocab_size, num_tags, is_training=True, seg=seg)
	# model = BiRnnCRF(vocab_size, num_tags, seg=seg)
	sv = tf.train.Supervisor(graph=model.graph, logdir=logdir, save_model_secs=0)
	with sv.managed_session() as sess:
		for epoch in range(1, hp.num_epochs + 1):
			if sv.should_stop():
				break
			train_loss = []
			for x_batch, y_batch, len_batch in batch_data(x_train, y_train, seq_lens, hp.batch_size):
				feed_dict = {model.x: x_batch, model.y: y_batch, model.seq_lens: len_batch}
				loss, _ = sess.run([model.loss, model.train_op], feed_dict=feed_dict)
				train_loss.append(loss)
			
			dev_loss = []
			predict_lists = []
			for x_batch, y_batch, len_batch in batch_data(x_dev, y_dev, dev_seq_lens, hp.batch_size):
				feed_dict = {model.x: x_batch, model.y: y_batch, model.seq_lens: len_batch}
				loss, logits = sess.run([model.loss, model.logits], feed_dict)
				dev_loss.append(loss)
				
				transition = model.transition.eval(session=sess)
				pre_seq = model.predict(logits, transition, len_batch)
				pre_label = recover_label(pre_seq, len_batch, id2tag)
				predict_lists.extend(pre_label)
			train_loss_v = np.round(float(np.mean(train_loss)), 4)
			dev_loss_v = np.round(float(np.mean(dev_loss)), 4)
			print('****************************************************')
			acc, p, r, f = get_ner_fmeasure(source_tag, predict_lists)
			print('epoch:\t{}\ttrain loss:\t{}\tdev loss:\t{}'.format(epoch, train_loss_v, dev_loss_v))
			print('acc:\t{}\tp:\t{}\tr:\t{}\tf:\t{}'.format(acc, p, r, f))
			print('****************************************************\n\n')
		# gs = sess.run(model.global_step)
		# sv.saver.save(sess, logdir + '/model_epoch_{}_{}'.format(epoch, gs))


def train_crf(is_vocab=False):
	if is_vocab:
		create_vocab(TRAIN_DATA)
	word2id, id2word = load_data(TOKEN_DATA)
	tag2id, id2tag = load_data(TAG_DATA)
	_, _, train_, x_train, y_train = generate_data(TRAIN_DATA, word2id, tag2id, max_len=hp.max_len)
	_, _, dev_seq_lens, x_dev, y_dev = generate_data(DEV_DATA, word2id, tag2id, max_len=hp.max_len)
	model_file = "logdir/model_crf"
	model = CRF()
	model.fit(x_train, y_train, template_file='model/module/templates.txt', model_file=model_file, max_iter=20)
	pre_seq = model.predict(x_dev, model_file=model_file)
	acc, p, r, f = get_ner_fmeasure(y_dev, pre_seq)
	print('acc:\t{}\tp:\t{}\tr:\t{}\tf:\t{}\n'.format(acc, p, r, f))


if __name__ == "__main__":
	train(is_vocab=True)
# train_crf(is_vocab=False)
