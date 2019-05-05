# -*- coding: utf-8 -*-

class Config:

	model_dir = "./ckpt/ffm.ckpt"
	log_dir = "./log"

	m = 39 # num of field
	n = 999997 # num of feature
	k = 4 # num of embedding

	batch_size = 1024
	shuffle = True # shuffled when get batch generation
	num_epoch = 2 # num of epochs for training
	l2_norm = 0.1 # param of L2 regularizer
	learning_rate = 0.001

	num_cpu = 2 # num of CPU cores used for training

config = Config()