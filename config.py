# -*- coding: utf-8 -*-

class Config:

	model_dir = "./ckpt"
	log_dir = "./log"

	m = 39 # num of field
	n = 999997 # num of feature
	k = 4 # num of embedding

	# batch_size = 2**13 # 8192
	batch_size = 256
	shuffle = True # shuffled when get batch generation
	num_epoch = 20 # num of epochs for training
	l2_norm = 0.00001 # param of L2 regularizer
	learning_rate = 0.1

	num_cpu = 4 # num of CPU cores used for training

config = Config()