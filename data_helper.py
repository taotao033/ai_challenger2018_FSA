import os
import re
import sys
import json
import pickle
import logging
import itertools
import numpy as np
import pandas as pd
import gensim as gs
from pprint import pprint
from collections import Counter
from tensorflow.contrib import learn

logging.getLogger().setLevel(logging.INFO)
filters = '!?"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n\r！@#￥%…&*（）：“”’‘；《》？，。'

def clean_str(str):
	str = re.sub("", "", str)
	str = re.sub("b", "", str)
	str = re.sub("【", " ", str)
	str = re.sub("】", " ", str)
	str = re.sub(r",", " ", str)
	str = re.sub(r"!", " ", str)
	str = re.sub("『", " ", str)
	str = re.sub("』", " ", str)
	str = re.sub("๑", " ", str)
	str = re.sub("º ั", " ", str)
	str = re.sub("╰", " ", str)
	str = re.sub("╯", " ", str)
	str = re.sub("≧", " ", str)
	str = re.sub("=", " ", str)
	str = re.sub("≦", " ", str)
	str = re.sub("-", " ", str)
	str = re.sub("_", " ", str)
	str = re.sub("➕", " ", str)
	str = re.sub("▽", " ", str)
	str = re.sub("＾", " ", str)
	str = re.sub("~", " ", str)
	str = re.sub("～", " ", str)
	str = re.sub("\.", " ", str)
	str = re.sub("…", " ", str)
	str = re.sub("/", " ", str)
	str = re.sub("，", " ", str)
	str = re.sub("、", " ", str)
	str = re.sub("！", " ", str)
	str = re.sub("：", " ", str)
	str = re.sub("？", " ", str)
	str = re.sub("。", " ", str)
	str = re.sub("\+", " ", str)
	str = re.sub("@", " ", str)
	str = re.sub("#", " ", str)
	str = re.sub("¥", " ", str)
	str = re.sub("￥", " ", str)
	str = re.sub("%", " ", str)
	str = re.sub("&", " ", str)
	str = re.sub("\*", " ", str)
	str = re.sub("“", " ", str)
	str = re.sub("”", " ", str)
	str = re.sub("\"", " ", str)
	str = re.sub("’", " ", str)
	str = re.sub("‘", " ", str)
	str = re.sub("；", " ", str)
	str = re.sub("《", " ", str)
	str = re.sub("》", " ", str)
	str = re.sub("【", " ", str)
	str = re.sub("】", " ", str)
	str = re.sub("'", " ", str)
	str = re.sub("（", " ", str)
	str = re.sub("）", " ", str)
	str = re.sub("★ ★ ★", " ", str)
	str = re.sub("☆", " ", str)
	str = re.sub("★ ★ ★", " ", str)
	str = re.sub("[a-zA-Z0-9]", " ", str)
	str = re.sub("😁", " ", str)
	str = re.sub("😯", " ", str)
	str = re.sub("😁", " ", str)
	str = re.sub("😊", " ", str)
	str = re.sub("😝", " ", str)
	str = re.sub("🙏", " ", str)
	str = re.sub("〈", " ", str)
	str = re.sub("∠", " ", str)
	str = re.sub(r"\(", " ", str)
	str = re.sub(r"\)", " ", str)
	str = re.sub(r"\?", " ", str)
	str = re.sub(r"\t", " ", str)
	str = re.sub(r"\n", " ", str)
	str = re.sub(r"\r", " ", str)
	str = re.sub(r"\s{2,}", " ", str)

	return str.strip()

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

def load_data(filename , max_length=1000):
	df = pd.read_csv(filename, encoding="utf-8")
	selected = ['content', 'location_traffic_convenience']
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
	train_file = './dataset/data_reform/train_reform_content_after_cut_mini.csv'
	x, y, vocab, vocab_inv, df, labels = load_data(train_file, max_length=1000)


