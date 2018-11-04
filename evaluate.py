import os
import json
import shutil
import pickle
import data_helper
import numpy as np
import pandas as pd
import tensorflow as tf
from text_cnn_rnn import TextCNNRNN
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)


def load_trained_params(trained_dir):
	params = json.loads(open(trained_dir + 'trained_parameters.json').read())
	words_index = json.loads(open('vocab.json').read())
	labels = json.loads(open(trained_dir + 'labels.json').read())

	with open(trained_dir + 'embeddings.pickle', 'rb') as input_file:
		fetched_embedding = pickle.load(input_file)
	embedding_mat = np.array(fetched_embedding, dtype = np.float32)
	return params, words_index, labels, embedding_mat

def load_test_data(test_file, labels, column):
	df = pd.read_csv(test_file, sep=',')
	select = ['content']

	df = df.dropna(axis=0, how='any', subset=select)
	test_examples = df[select[0]].apply(lambda x: data_helper.clean_str(x).split(' ')).tolist()

	num_labels = len(labels)
	one_hot = np.zeros((num_labels, num_labels), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))

	y_ = None
	if column in df.columns:
		select.append(column)
		y_ = df[select[1]].apply(lambda x: label_dict[x]).tolist()

	not_select = list(set(df.columns) - set(select))
	df = df.drop(not_select, axis=1)
	return test_examples, y_, df

def map_word_to_index(examples, words_index):
	x_ = []
	for example in examples:
		temp = []
		for word in example:
			if word in words_index:
				temp.append(words_index[word])
			else:
				temp.append(0)
		x_.append(temp)
	return x_

def predict_unseen_data(column, model_path, pre_path):
	trained_dir = model_path
	if not trained_dir.endswith('/'):
		trained_dir += '/'
	test_file = pre_path

	params, words_index, labels, embedding_mat = load_trained_params(trained_dir)

	x_, y_, df = load_test_data(test_file, labels, column)
	x_ = data_helper.pad_sentences(x_, forced_sequence_length=params['sequence_length'])
	x_ = map_word_to_index(x_, words_index)

	x_test, y_test = np.asarray(x_), None
	if y_ is not None:
		y_test = np.asarray(y_)

	predicted_dir = './val_predicted_results/val_predicted_results_' + column + '/'
	if os.path.exists(predicted_dir):
		shutil.rmtree(predicted_dir)
	os.makedirs(predicted_dir)

	with tf.Graph().as_default():
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			cnn_rnn = TextCNNRNN(
				embedding_mat=embedding_mat,
				non_static=params['non_static'],
				hidden_unit=params['hidden_unit'],
				sequence_length=len(x_test[0]),
				max_pool_size=params['max_pool_size'],
				filter_sizes=map(int, params['filter_sizes'].split(",")),
				num_filters=params['num_filters'],
				num_classes=len(labels),
				embedding_size=params['embedding_dim'],
				l2_reg_lambda=params['l2_reg_lambda'])

			def real_len(batches):
				return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size']) for batch in batches]

			def predict_step(x_batch):
				feed_dict = {
					cnn_rnn.input_x: x_batch,
					cnn_rnn.dropout_keep_prob: 1.0,
					cnn_rnn.batch_size: len(x_batch),
					cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
					cnn_rnn.real_len: real_len(x_batch),
				}
				predictions = sess.run([cnn_rnn.predictions], feed_dict)
				return predictions

			checkpoint_file = trained_dir + 'best_model.ckpt'
			saver = tf.train.Saver(tf.all_variables())
			saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
			saver.restore(sess, checkpoint_file)
			logger.critical('{} has been loaded'.format(checkpoint_file))

			batches = data_helper.batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)

			predictions, predict_labels = [], []
			for x_batch in batches:
				batch_predictions = predict_step(x_batch)[0]
				for batch_prediction in batch_predictions:
					predictions.append(batch_prediction)
					predict_labels.append(labels[batch_prediction])

			# Save the predictions back to file
			df['NEW_PREDICTED'] = predict_labels
			columns = sorted(df.columns, reverse=False)
			df.to_csv(predicted_dir + 'predictions_all.csv', index=False, columns=columns, sep=',')

			if y_test is not None:
				y_test = np.array(np.argmax(y_test, axis=1))
				accuracy = sum(np.array(predictions) == y_test) / float(len(y_test))
				logger.critical('The prediction accuracy is: {}'.format(accuracy))

			logger.critical("%s Prediction is complete" % column)


column_list = [
	 "location_traffic_convenience",
	 "location_distance_from_business_district",
	 "location_easy_to_find",
	 "service_wait_time",
	 "service_waiters_attitude",
	 "service_parking_convenience",
	 "service_serving_speed",
	 "price_level",
	 "price_cost_effective",
	 "price_discount",
	 "environment_decoration",
	 "environment_noise",
	 "environment_space",
	 "environment_cleaness",
	 "dish_portion",
	 "dish_taste",
	 "dish_look",
	 "dish_recommendation",
	 "others_overall_experience",
	 "others_willing_to_consume_again"
]

if __name__ == '__main__':

	for column in column_list:
		model_path = "./trained_results/trained_results_" + column
		pre_path = "./dataset/valid_content_after_cut.csv"
		predict_unseen_data(column, model_path, pre_path)
	logger.info("Prediction is complete, start merge data")

	#next step:merger predicted labels
	df = pd.read_csv("./dataset/valid.csv", encoding="utf-8")
	for column in column_list:
		df_predicted = pd.read_csv('./val_predicted_results/val_predicted_results_' + column + '/' + 'predictions_all.csv', encoding="utf-8")
		df[column] = df_predicted['NEW_PREDICTED']
	predict_saved_path = "./output/val_predicted_F1_0.61.csv"
	df.to_csv(predict_saved_path, index=False, sep=",", encoding="utf-8")
	logger.info("Compplete all data merge")
	logger.info("Complete all prediction,predict results have been saved:{}".format(predict_saved_path))

