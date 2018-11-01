import os
import sys
import json
import time
import shutil
import pickle
import logging
import data_helper
import numpy as np
import pandas as pd
import tensorflow as tf
from text_cnn_rnn import TextCNNRNN
from sklearn.model_selection import train_test_split

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

#logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)

def train_cnn_rnn(train_path, val_path, max_length, column):
	#input_file = sys.argv[1]
	x_, y_, x_val, y_val, vocabulary_dict, vocabulary, df, labels = data_helper.load_data(train_filename=train_path,
																		                 val_filename=val_path,
																		   				 max_length=max_length, column=column)
	#training_config = sys.argv[2]
	training_config = "./training_config.json"
	params = json.loads(open(training_config).read())

	# Assign a 300 dimension vector to each word
	word_embeddings = data_helper.load_embeddings(vocabulary_dict)
	embedding_mat = [word_embeddings[word] for index, word in enumerate(vocabulary)]
	embedding_mat = np.array(embedding_mat, dtype=np.float32)

	# Split the original dataset into train set and test set
    #x, x_test, y, y_test = train_test_split(x_, y_, test_size=0.1)
	x_test = x_val
	y_test = y_val
	# Split the train set into train set and dev set
	x_train, x_dev, y_train, y_dev = train_test_split(x_, y_, test_size=0.1)

	logger.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
	logger.info('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))

	# Create a directory, everything related to the training will be saved in this directory
	timestamp = str(int(time.time()))
	print(column + "_" + "timestamp: " + timestamp)
	trained_dir = './trained_results_' + column + '/'

	if os.path.exists(trained_dir):
		shutil.rmtree(trained_dir)
	os.makedirs(trained_dir)

	graph = tf.Graph()
	with graph.as_default():
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			cnn_rnn = TextCNNRNN(
				embedding_mat=embedding_mat,
				sequence_length=x_train.shape[1],
				num_classes=y_train.shape[1],
				non_static=params['non_static'],
				hidden_unit=params['hidden_unit'],
				max_pool_size=params['max_pool_size'],
				filter_sizes=map(int, params['filter_sizes'].split(",")),
				num_filters=params['num_filters'],
				embedding_size=params['embedding_dim'],
				l2_reg_lambda=params['l2_reg_lambda'])

			global_step = tf.Variable(0, name='global_step', trainable=False)
			optimizer = tf.train.RMSPropOptimizer(1e-3, decay=0.9)
			grads_and_vars = optimizer.compute_gradients(cnn_rnn.loss)
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

			# Checkpoint files will be saved in this directory during training
			checkpoint_dir = './checkpoints_' + column + '/'
			if os.path.exists(checkpoint_dir):
				shutil.rmtree(checkpoint_dir)
			os.makedirs(checkpoint_dir)
			checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

			def real_len(batches):
				return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size']) for batch in batches]

			def train_step(x_batch, y_batch):
				feed_dict = {
					cnn_rnn.input_x: x_batch,
					cnn_rnn.input_y: y_batch,
					cnn_rnn.dropout_keep_prob: params['dropout_keep_prob'],
					cnn_rnn.batch_size: len(x_batch),
					cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
					cnn_rnn.real_len: real_len(x_batch),
				}
				_, step, loss, accuracy = sess.run([train_op, global_step, cnn_rnn.loss, cnn_rnn.accuracy], feed_dict)

			def dev_step(x_batch, y_batch):
				feed_dict = {
					cnn_rnn.input_x: x_batch,
					cnn_rnn.input_y: y_batch,
					cnn_rnn.dropout_keep_prob: 1.0,
					cnn_rnn.batch_size: len(x_batch),
					cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
					cnn_rnn.real_len: real_len(x_batch),
				}
				step, loss, accuracy, num_correct, predictions = sess.run(
					[global_step, cnn_rnn.loss, cnn_rnn.accuracy, cnn_rnn.num_correct, cnn_rnn.predictions], feed_dict)
				return accuracy, loss, num_correct, predictions

			saver = tf.train.Saver()
			sess.run(tf.global_variables_initializer())

			# Training starts here
			train_batches = data_helper.batch_iter(list(zip(x_train, y_train)), params['batch_size'], params['num_epochs'])
			best_accuracy, best_at_step = 0, 0

			# Train the model with x_train and y_train
			for train_batch in train_batches:
				x_train_batch, y_train_batch = zip(*train_batch)
				train_step(x_train_batch, y_train_batch)
				current_step = tf.train.global_step(sess, global_step)

				# Evaluate the model with x_dev and y_dev
				if current_step % params['evaluate_every'] == 0:
					dev_batches = data_helper.batch_iter(list(zip(x_dev, y_dev)), params['batch_size'], 1)

					total_dev_correct = 0
					for dev_batch in dev_batches:
						x_dev_batch, y_dev_batch = zip(*dev_batch)
						acc, loss, num_dev_correct, predictions = dev_step(x_dev_batch, y_dev_batch)
						total_dev_correct += num_dev_correct
					accuracy = float(total_dev_correct) / len(y_dev)
					logging.info('Accuracy on dev set: {}'.format(accuracy))

					if accuracy >= best_accuracy:
						best_accuracy, best_at_step = accuracy, current_step
						path = saver.save(sess, checkpoint_prefix, global_step=current_step)
						logging.critical('Saved model {} at step {}'.format(path, best_at_step))
						logging.critical('Best accuracy {} at step {}'.format(best_accuracy, best_at_step))
			logger.critical('Training is complete, testing the best model on x_test and y_test')

			# Save the model files to trained_dir. predict.py needs trained model files. 
			saver.save(sess, trained_dir + "best_model.ckpt")

			# Evaluate x_test and y_test
			saver.restore(sess, checkpoint_prefix + '-' + str(best_at_step))
			test_batches = data_helper.batch_iter(list(zip(x_test, y_test)), params['batch_size'], 1, shuffle=False)
			total_test_correct = 0
			for test_batch in test_batches:
				x_test_batch, y_test_batch = zip(*test_batch)
				acc, loss, num_test_correct, predictions = dev_step(x_test_batch, y_test_batch)
				total_test_correct += int(num_test_correct)
			logger.critical('Accuracy on test set: {}'.format(float(total_test_correct) / len(y_test)))

	# Save trained parameters and files since predict.py needs them
	with open(trained_dir + 'words_index.json', 'w') as outfile:
		json.dump(vocabulary, outfile, indent=4, ensure_ascii=False)
	with open(trained_dir + 'embeddings.pickle', 'wb') as outfile:
		pickle.dump(embedding_mat, outfile, pickle.HIGHEST_PROTOCOL)
	with open(trained_dir + 'labels.json', 'w') as outfile:
		json.dump(labels, outfile, indent=4, ensure_ascii=False)

	params['sequence_length'] = x_train.shape[1]
	with open(trained_dir + 'trained_parameters.json', 'w') as outfile:
		json.dump(params, outfile, indent=4, sort_keys=True, ensure_ascii=False)

column_list = [
     # "location_traffic_convenience",
     # "location_distance_from_business_district",
     # "location_easy_to_find",
     # "service_wait_time",
     # "service_waiters_attitude",
     # "service_parking_convenience",
     # "service_serving_speed",
     # "price_level",
     # "price_cost_effective",
     #"price_discount",
     # "environment_decoration",
     # "environment_noise",
     # "environment_space",
     # "environment_cleaness",
     "dish_portion",
     "dish_taste",
     "dish_look",
     "dish_recommendation",
     "others_overall_experience",
     "others_willing_to_consume_again"
]


if __name__ == '__main__':

	maxlength = 1000
	for column in column_list:
		train_path = "./dataset/data_reform/train_reform_content_after_cut.csv"
		val_path = "./dataset/valid_content_after_cut.csv"
		train_cnn_rnn(train_path=train_path, val_path=val_path, max_length=maxlength, column=column)
