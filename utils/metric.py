#!/usr/bin python3
# -*- coding: utf-8 -*-
# @Time    : 18-12-27 上午11:34
# @Author  : 林利芳
# @File    : metric.py


def recover_label(tags, seq_lens, id2tag):
	"""
	恢复标签 ID->tag
	:param tags:
	:param seq_lens:
	:param id2tag:
	:return:
	"""
	labels = []
	for tag, seq_len in zip(tags, seq_lens):
		pre = [id2tag.get(str(_id), 'O') for _id in tag[:seq_len]]
		labels.append(pre)
	return labels


def get_ner_fmeasure(golden_lists, predict_lists, label_type="BMES"):
	golden_full = []
	predict_full = []
	right_full = []
	right_tag = 0
	all_tag = 0
	for idx, (golden_list, predict_list) in enumerate(zip(golden_lists, predict_lists)):
		for golden_tag, predict_tag in zip(golden_list, predict_list):
			if golden_tag == predict_tag:
				right_tag += 1
		all_tag += len(golden_list)
		if label_type == "BMES":
			gold_matrix = get_ner_BMES(golden_list)
			pred_matrix = get_ner_BMES(predict_list)
		else:
			gold_matrix = get_ner_BIO(golden_list)
			pred_matrix = get_ner_BIO(predict_list)
		# 交集
		right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
		golden_full += gold_matrix
		predict_full += pred_matrix
		right_full += right_ner
	right_num = len(right_full)
	golden_num = len(golden_full)
	predict_num = len(predict_full)
	if predict_num == 0:
		precision = -1
	else:
		precision = (right_num + 0.0) / predict_num
	if golden_num == 0:
		recall = -1
	else:
		recall = (right_num + 0.0) / golden_num
	if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
		f_measure = -1
	else:
		f_measure = 2 * precision * recall / (precision + recall)
	accuracy = (right_tag + 0.0) / all_tag
	print("gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num)
	return round(accuracy, 4), round(precision, 4), round(recall, 4), round(f_measure, 4)


def reverse_style(input_string):
	target_position = input_string.index('[')
	input_len = len(input_string)
	output_string = input_string[target_position:input_len] + input_string[0:target_position]
	return output_string


def get_ner_BMES(label_list):
	list_len = len(label_list)
	begin_label = 'B'
	end_label = 'E'
	single_label = 'S'
	whole_tag = ''
	index_tag = ''
	tag_list = []
	stand_matrix = []
	for i in range(list_len):
		# wordlabel = word_list[i]
		current_label = label_list[i].upper()
		tags = current_label.split('-')
		if begin_label in current_label:
			if index_tag != '':
				tag_list.append(whole_tag + ',' + str(i - 1))
			whole_tag = tags[-1] + '[' + str(i)
			index_tag = tags[-1]
		
		elif single_label in current_label:
			if index_tag != '':
				tag_list.append(whole_tag + ',' + str(i - 1))
			whole_tag = tags[-1] + '[' + str(i)
			tag_list.append(whole_tag)
			whole_tag = ""
			index_tag = ""
		elif end_label in current_label:
			if index_tag != '':
				tag_list.append(whole_tag + ',' + str(i))
			whole_tag = ''
			index_tag = ''
		else:
			continue
	if (whole_tag != '') & (index_tag != ''):
		tag_list.append(whole_tag)
	tag_list_len = len(tag_list)
	
	for i in range(0, tag_list_len):
		if len(tag_list[i]) > 0:
			tag_list[i] = tag_list[i] + ']'
			insert_list = reverse_style(tag_list[i])
			stand_matrix.append(insert_list)
	# print stand_matrix
	return stand_matrix


def get_ner_BIO(label_list):
	# list_len = len(word_list)
	# assert(list_len == len(label_list)), "word list size unmatch with label list"
	list_len = len(label_list)
	begin_label = 'B-'
	inside_label = 'I-'
	whole_tag = ''
	index_tag = ''
	tag_list = []
	stand_matrix = []
	for i in range(0, list_len):
		# wordlabel = word_list[i]
		current_label = label_list[i].upper()
		if begin_label in current_label:
			if index_tag == '':
				whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
				index_tag = current_label.replace(begin_label, "", 1)
			else:
				tag_list.append(whole_tag + ',' + str(i - 1))
				whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
				index_tag = current_label.replace(begin_label, "", 1)
		
		elif inside_label in current_label:
			if current_label.replace(inside_label, "", 1) == index_tag:
				whole_tag = whole_tag
			else:
				if (whole_tag != '') & (index_tag != ''):
					tag_list.append(whole_tag + ',' + str(i - 1))
				whole_tag = ''
				index_tag = ''
		else:
			if (whole_tag != '') & (index_tag != ''):
				tag_list.append(whole_tag + ',' + str(i - 1))
			whole_tag = ''
			index_tag = ''
	
	if (whole_tag != '') & (index_tag != ''):
		tag_list.append(whole_tag)
	tag_list_len = len(tag_list)
	
	for i in range(0, tag_list_len):
		if len(tag_list[i]) > 0:
			tag_list[i] = tag_list[i] + ']'
			insert_list = reverse_style(tag_list[i])
			stand_matrix.append(insert_list)
	return stand_matrix
