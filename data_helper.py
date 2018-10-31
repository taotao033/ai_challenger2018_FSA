import os
import re
import sys
import json
import pickle
import jieba
import logging
import itertools
import numpy as np
import pandas as pd
import gensim as gs
from pprint import pprint
from collections import Counter
from tensorflow.contrib import learn

logging.getLogger().setLevel(logging.INFO)


def clean_str(s):
	s = re.sub("", "", s)
	s = re.sub("b", "", s)
	s = re.sub("【", " ", s)
	s = re.sub("】", " ", s)
	s = re.sub(r",", " ", s)
	s = re.sub(r"!", " ", s)
	s = re.sub("『", " ", s)
	s = re.sub("』", " ", s)
	s = re.sub("๑", " ", s)
	s = re.sub("º ั", " ", s)
	s = re.sub("╰", " ", s)
	s = re.sub("╯", " ", s)
	s = re.sub("≧", " ", s)
	s = re.sub("=", " ", s)
	s = re.sub("≦", " ", s)
	s = re.sub("-", " ", s)
	s = re.sub("─", " ", s)
	s = re.sub("_", " ", s)
	s = re.sub("➕", " ", s)
	s = re.sub("▽", " ", s)
	s = re.sub("＾", " ", s)
	s = re.sub("♪", " ", s)
	s = re.sub("~", " ", s)
	s = re.sub("～", " ", s)
	s = re.sub("\.", " ", s)
	s = re.sub("…", " ", s)
	s = re.sub("/", " ", s)
	s = re.sub("，", " ", s)
	s = re.sub("、", " ", s)
	s = re.sub("！", " ", s)
	s = re.sub("：", " ", s)
	s = re.sub("？", " ", s)
	s = re.sub("。", " ", s)
	s = re.sub("\+", " ", s)
	s = re.sub("⊙", " ", s)
	s = re.sub("✧", " ", s)
	s = re.sub("٩", " ", s)
	s = re.sub("ˊ", " ", s)
	s = re.sub("ω", " ", s)
	s = re.sub("ˋ", " ", s)
	s = re.sub("ゝ", " ", s)
	s = re.sub(" و", " ", s)
	s = re.sub("ㅂ", " ", s)
	s = re.sub("◡", " ", s)
	s = re.sub("̳", " ", s)
	s = re.sub("ฅ", " ", s)
	s = re.sub("\+", " ", s)
	s = re.sub("Ｏ", " ", s)
	s = re.sub("o", " ", s)
	s = re.sub("@", " ", s)
	s = re.sub("#", " ", s)
	s = re.sub("¥", " ", s)
	s = re.sub("￥", " ", s)
	s = re.sub("%", " ", s)
	s = re.sub("&", " ", s)
	s = re.sub("🙈", " ", s)
	s = re.sub("“", " ", s)
	s = re.sub("”", " ", s)
	s = re.sub("\"", " ", s)
	s = re.sub("’", " ", s)
	s = re.sub("‘", " ", s)
	s = re.sub("；", " ", s)
	s = re.sub("《", " ", s)
	s = re.sub("》", " ", s)
	s = re.sub("【", " ", s)
	s = re.sub("】", " ", s)
	s = re.sub("'", " ", s)
	s = re.sub("（", " ", s)
	s = re.sub("）", " ", s)
	s = re.sub("•", " ", s)
	s = re.sub("★ ★ ★", " ", s)
	s = re.sub("☆", " ", s)
	s = re.sub("★ ★ ★", " ", s)
	s = re.sub("❀", " ", s)
	s = re.sub("[a-zA-Z0-9]", " ", s)
	s = re.sub("😁", " ", s)
	s = re.sub("😯", " ", s)
	s = re.sub("😁", " ", s)
	s = re.sub("😊", " ", s)
	s = re.sub("😝", " ", s)
	s = re.sub("😰", " ", s)
	s = re.sub("🙏", " ", s)
	s = re.sub("〈", " ", s)
	s = re.sub("∠", " ", s)
	s = re.sub("∩", " ", s)
	s = re.sub(r"\(", " ", s)
	s = re.sub(r"\)", " ", s)
	s = re.sub(r"\?", " ", s)
	s = re.sub(r"\t", " ", s)
	s = re.sub(r"\n", " ", s)
	s = re.sub(r"\r", " ", s)
	s = re.sub(r"\s{2,}", " ", s)

	return s.strip()


def get_balance_train_data(path="./dataset/data_reform/train_reform_content_after_cut",):  #balance of labels{-2 , -1, 0, 1},
	df = pd.read_csv(path, encoding="utf-8")

	location_traffic_convenience_without_minus2_and_1 = df[(df["location_traffic_convenience"] != -2)
													 & (df["location_traffic_convenience"] != 1)]
	location_traffic_convenience_label1 = location_traffic_convenience_without_minus2_and_1["location_traffic_convenience"]
	location_traffic_convenience_content1 = location_traffic_convenience_without_minus2_and_1["content"]

	location_traffic_convenience_only_minus2 = df[df["location_traffic_convenience"] == -2]
	location_traffic_convenience_label2 = location_traffic_convenience_only_minus2["location_traffic_convenience"]
	location_traffic_convenience_content2 = location_traffic_convenience_only_minus2["content"]

	location_traffic_convenience_only_1 = df[df["location_traffic_convenience"] == 1]
	location_traffic_convenience_label3 = location_traffic_convenience_only_1["location_traffic_convenience"]
	location_traffic_convenience_content3 = location_traffic_convenience_only_1["content"]

	frames_label_location_traffic_convenience = [location_traffic_convenience_label1,
												 location_traffic_convenience_label2[0:1000],
	                                             location_traffic_convenience_label3[0:1000]]
	location_traffic_convenience_label_concat = pd.concat(frames_label_location_traffic_convenience)

	frames_content_location_traffic_convenience = [location_traffic_convenience_content1,
												   location_traffic_convenience_content2[0:1000],
												   location_traffic_convenience_content3[0:1000]]
	location_traffic_convenience_content_concat = pd.concat(frames_content_location_traffic_convenience)

	# location_distance_from_business_district_label = location["location_distance_from_business_district"]
	# location_easy_to_find_label = location["location_easy_to_find"]
	# service_wait_time_label = service["service_wait_time"]
	# service_waiters_attitude_label = service_reform["service_waiters_attitude"]
	# service_parking_convenience_label = service["service_parking_convenience"]
	# service_serving_speed_label = service["service_serving_speed"]
	# price_level_label = price["price_level"]
	# price_cost_effective_label = price["price_cost_effective"]
	# price_discount_label = price["price_discount"]
	# environment_decoration_label = environment["environment_decoration"]
	# environment_noise_label = environment["environment_noise"]
	# environment_space_label = environment["environment_space"]
	# environment_cleaness_label = environment["environment_cleaness"]
	# dish_portion_label = dish["dish_portion"]
	# dish_taste_label = dish["dish_taste"]
	# dish_look_label = dish["dish_look"]
	# dish_recommendation_label = dish["dish_recommendation"]
	# others_overall_experience_label = others["others_overall_experience"]
	# others_willing_to_consume_again_label = others["others_willing_to_consume_again"]
	#

	# location_distance_from_business_district_content = location["content"]
	# location_easy_to_find_content = location["content"]
	# service_wait_time_content = service["content"]
	# service_waiters_attitude_content = service_reform["content"]
	# service_parking_convenience_content = service["content"]
	# service_serving_speed_content = service["content"]
	# price_level_content = price["content"]
	# price_cost_effective_content = price["content"]
	# price_discount_content = price["content"]
	# environment_decoration_content = environment["content"]
	# environment_noise_content = environment["content"]
	# environment_space_content = environment["content"]
	# environment_cleaness_content = environment["content"]
	# dish_portion_content = dish["content"]
	# dish_taste_content = dish["content"]
	# dish_look_content = dish["content"]
	# dish_recommendation_content = dish["content"]
	# others_overall_experience_content = others["content"]
	# others_willing_to_consume_again_content = others["content"]
	#
	# location_reform2 = location[location["location_traffic_convenience"] == 1]
	# location_traffic_convenience_label_add_only_label1 = location_reform2["location_traffic_convenience"]
	# location_traffic_convenience_content_add_add_only_label1 = location_reform2["content"]
	#
	# location_reform2_lo = location[location["location_traffic_convenience"] == -2]
	# location_traffic_convenience_label_add_only_label_minus2 = location_reform2_lo["location_traffic_convenience"]
	# location_traffic_convenience_content_add_add_only_label_minus2 = location_reform2_lo["content"]
	#
	# frames_label1 = [location_traffic_convenience_label, location_traffic_convenience_label_add_only_label1[0:20000],
	# 				 location_traffic_convenience_label_add_only_label_minus2]
	# location_traffic_convenience_label_concat = pd.concat(frames_label1)
	#
	# frames_content1 = [location_traffic_convenience_content,
	# 				   location_traffic_convenience_content_add_add_only_label1[0:20000],
	# 				   location_traffic_convenience_content_add_add_only_label_minus2]
	# location_traffic_convenience_content_concat = pd.concat(frames_content1)
	#
	# service_reform2 = service[service["service_waiters_attitude"] == 1]
	# service_waiters_attitude_label_add_only_label1 = service_reform2["service_waiters_attitude"]
	# service_waiters_attitude_content_add_only_label1 = service_reform2["content"]
	# frames_label2 = [service_waiters_attitude_label, service_waiters_attitude_label_add_only_label1[0:21372]]
	# service_waiters_attitude_label_concat = pd.concat(frames_label2)
	# frames_content2 = [service_waiters_attitude_content, service_waiters_attitude_content_add_only_label1[0:21372]]
	# service_waiters_attitude_content_concat = pd.concat(frames_content2)
	#
	segment_data_dic = {
		"location_traffic_convenience": [location_traffic_convenience_content_concat,
										 location_traffic_convenience_label_concat]
	#
	# 	"location_distance_from_business_district": [location_distance_from_business_district_content,
	# 												 location_distance_from_business_district_label],
	# 	"location_easy_to_find": [location_easy_to_find_content, location_easy_to_find_label],
	# 	"service_wait_time": [service_wait_time_content, service_wait_time_label],
	# 	"service_waiters_attitude": [service_waiters_attitude_content_concat, service_waiters_attitude_label_concat],
	# 	"service_parking_convenience": [service_parking_convenience_content, service_parking_convenience_label],
	# 	"service_serving_speed": [service_serving_speed_content, service_serving_speed_label],
	# 	"price_level": [price_level_content, price_level_label],
	# 	"price_cost_effective": [price_cost_effective_content, price_cost_effective_label],
	# 	"price_discount": [price_discount_content, price_discount_label],
	# 	"environment_decoration": [environment_decoration_content, environment_decoration_label],
	# 	"environment_noise": [environment_noise_content, environment_noise_label],
	# 	"environment_space": [environment_space_content, environment_space_label],
	# 	"environment_cleaness": [environment_cleaness_content, environment_cleaness_label],
	# 	"dish_portion": [dish_portion_content, dish_portion_label],
	# 	"dish_taste": [dish_taste_content, dish_taste_label],
	# 	"dish_look": [dish_look_content, dish_look_label],
	# 	"dish_recommendation": [dish_recommendation_content, dish_recommendation_label],
	# 	"others_overall_experience": [others_overall_experience_content, others_overall_experience_label],
	# 	"others_willing_to_consume_again": [others_willing_to_consume_again_content,
	# 										others_willing_to_consume_again_label]
	#
	}
	# for key in segment_data_dic.keys():
	#
	# 	label_list = segment_data_dic[key][1]
	#
	# 	label_list_onehot = []
	# 	for data in label_list:
	#
	# 		label = [0, 0, 0, 0]
	#
	# 		if data == 1:
	# 			label[0] = 1
	# 		if data == 0:
	# 			label[1] = 1
	# 		if data == -1:
	# 			label[2] = 1
	# 		if data == -2:
	# 			label[3] = 1
	# 		label_list_onehot.append(label)
	# 	segment_data_dic[key][1] = label_list_onehot

	# return segment_data_dic, len(vocab), 4
	return segment_data_dic


def seg_word(content):

	content_list = []
	for text in content:
		token = jieba.cut(text)

		arr_temp = []
		for item in token:
			arr_temp.append(item)
		content_list.append(" ".join(arr_temp))
	return content_list


def load_embeddings(vocabulary):
	word_embeddings = {}
	for word in vocabulary:
		word_embeddings[word] = np.random.uniform(-0.25, 0.25, 300)
	return word_embeddings


def pad_sentences(sentences, padding_word="<PAD/>", forced_sequence_length=None):
	"""Pad setences during training or prediction"""
	if forced_sequence_length is None: # Train
		sequence_length = max(len(x) for x in sentences)
	else: # Prediction
		logging.critical('This is prediction, reading the trained sequence length')
		sequence_length = forced_sequence_length
	logging.critical('The maximum length is {}'.format(sequence_length))

	padded_sentences = []
	for i in range(len(sentences)):
		sentence = sentences[i]
		num_padding = sequence_length - len(sentence)

		if num_padding < 0: # Prediction: cut off the sentence if it is longer than the sequence length
			logging.info('This sentence has to be cut off because it is longer than trained sequence length')
			padded_sentence = sentence[0:sequence_length]
		else:
			padded_sentence = sentence + [padding_word] * num_padding
		padded_sentences.append(padded_sentence)
	return padded_sentences


def build_vocab(sentences):
	word_counts = Counter(itertools.chain(*sentences))
	vocabulary_inv = [word[0] for word in word_counts.most_common()]
	vocabulary = {word: index for index, word in enumerate(vocabulary_inv)}
	return vocabulary, vocabulary_inv


def batch_iter(data, batch_size, num_epochs, shuffle=True):
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int(data_size / batch_size) + 1

	for epoch in range(num_epochs):
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data

		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]


def load_data(filename, max_length, column):
	df = pd.read_csv(filename, encoding="utf-8")
	selected = ['content', column]
	non_selected = list(set(df.columns) - set(selected))

	df = df.drop(non_selected, axis=1)
	df = df.dropna(axis=0, how='any', subset=selected)
	df = df.reindex(np.random.permutation(df.index))

	labels = sorted(list(set(df[selected[1]].tolist())))   #sort ascending, after do this ,labels = [-2, -1, 0, 1]
	num_labels = len(labels)
	one_hot = np.zeros((num_labels, num_labels), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))#{-2: array([1, 0, 0, 0]), -1: array([0, 1, 0, 0]), 0: array([0, 0, 1, 0]), 1: array([0, 0, 0, 1])}

	x_raw = df[selected[0]].apply(lambda x: clean_str(x).split(' ')).tolist()
	y_raw = df[selected[1]].apply(lambda y: label_dict[y]).tolist()
	#print(x_raw)
	x_raw = pad_sentences(x_raw, forced_sequence_length=max_length)

	vocabulary, vocabulary_inv = build_vocab(x_raw)

	x = np.array([[vocabulary[word] for word in sentence] for sentence in x_raw])
	y = np.array(y_raw)
	return x, y, vocabulary, vocabulary_inv, df, labels


if __name__ == "__main__":
	#train_file = './dataset/data_reform/train_reform_content_after_cut_mini.csv'
	#x, y, vocab, vocab_inv, df, labels = load_data(train_file, max_length=1000)
	# df = pd.read_csv("./dataset/valid.csv", encoding="utf-8")
	# content = df["content"]
	# content_list = seg_word(content)
	# df["content"] = content_list
	# df.to_csv("./dataset/valid_content_after_cut.csv", index=False, sep=",", encoding="utf-8")
	#
	balance_data_dict = get_balance_train_data(path="./dataset/data_reform/train_reform_content_after_cut.csv")
	data_frame = pd.DataFrame({"content": balance_data_dict["location_traffic_convenience"][0],
							"location_traffic_convenience": balance_data_dict["location_traffic_convenience"][1]})

	#data_frame.to_csv("./dataset/data_reform/balance_location_traffic_convenience.csv", index=False, sep=",", encoding="utf-8")
	#
	# df = pd.read_csv("./dataset/data_reform/balance_location_traffic_convenience.csv", encoding="utf-8", header=None)
	ds = data_frame.sample(frac=1)
	ds.to_csv("new_files_balance.csv", index=False, sep=",", encoding="utf-8")

	print("the number of -2: " + str(len(ds[ds["location_traffic_convenience"] == -2]["location_traffic_convenience"])))
	print("the number of -1: " + str(len(ds[ds["location_traffic_convenience"] == -1]["location_traffic_convenience"])))
	print("the number of 0: " + str(len(ds[ds["location_traffic_convenience"] == 0]["location_traffic_convenience"])))
	print("the number of 1: " + str(len(ds[ds["location_traffic_convenience"] == 1]["location_traffic_convenience"])))
	#print(str(len(ds[ds["location_traffic_convenience"] == "location_traffic_convenience"])))
	# df = pd.read_csv("./new_files_update.csv", encoding="utf-8")
	# #print(len(df[df["location_traffic_convenience"] == -2]))
	# print(len(df[df["location_traffic_convenience"] == 0]))
	# print(len(df[df["location_traffic_convenience"] == -1]))
	#df_new.to_csv("./new_files_update.csv", index=False, sep=",", encoding="utf-8")
