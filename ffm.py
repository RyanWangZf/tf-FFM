# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
import pdb

class FFM:

	def __init__(self,config):
		self.config = config

	def build(self):
		# build components
		self._inference()
		self._loss_func()
		self._optimizer()

		# build session
		sess_config = tf.ConfigProto(device_count = {"CPU":self.config.num_cpu},
			inter_op_parallelism_threads=0,
			intra_op_parallelism_threads=0)
		self.sess = tf.Session(config=sess_config)

	def train(self,dataset,va_dataset=None):
		if va_dataset is not None:
			va_data = va_dataset.get_all_data()

		config = self.config
		sample_total_num = len(dataset.raw_data)
		num_iter_one_epoch = sample_total_num // config.batch_size

		# restore model
		self.sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
		var_list = tf.contrib.framework.get_variables_to_restore(include=["ffm"])
		try:
			saver_to_restore = tf.train.Saver(var_list=var_list)
			saver_to_restore.restore(self.sess,config.model_dir)
		except:
			print("[INFO] Cannot find saved checkpoint in {}.".format(config.model_dir))

		saver = tf.train.Saver(max_to_keep=2)

		# training
		for step in range(config.num_epoch):
			for epoch in range(num_iter_one_epoch):
				batch_data = dataset.get_batch()
				batch_feature = batch_data[1]
				batch_label = batch_data[3]

				batch_loss,_,batch_logloss = self.sess.run([self.loss,self.train_op,self.log_loss],
						feed_dict = {self.features:batch_feature,self.label:batch_label})

				sys.stdout.write("\r")
				sys.stdout.write("=> [INFO] Process {:.0%} in Step {:d}: Train log-loss: {:.2f} <= \r".format(epoch/num_iter_one_epoch, step+1,batch_logloss))
				sys.stdout.flush()
				# debug
				if epoch == 1000:
					break

			# validation
			if va_dataset is not None:
				va_loss = self.sess.run(self.log_loss,
					feed_dict={self.features:va_data["feature"],self.label:va_data["label"]})
				print("=> STEP {}, val_loss: {:.4f} <=".format(step+1,va_loss))

			# save model
			saver.save(self.sess,config.model_dir,global_step=step+1)

		return

	def _inference(self):
		config = self.config
		self.label = tf.placeholder(shape=(None),dtype=tf.float32)
		self.features = tf.placeholder(shape=(None,config.m),dtype=tf.int32)
		with tf.variable_scope("ffm"):
			with tf.variable_scope("linear"):
				weights = tf.get_variable("weights",
					shape=[config.n,1],
					dtype=tf.float32,
					initializer=tf.truncated_normal_initializer(stddev=0.01,mean=0)) # [n,1]

				bias = tf.get_variable("bias",
					shape=[1],
					dtype=tf.float32,
					initializer=tf.zeros_initializer())

				linear_term = tf.gather(weights,self.features) # [None,m,1]
				linear_term = tf.add(bias, tf.reduce_sum(linear_term,[-1,-2])) # [None,]

				tf.add_to_collection(tf.GraphKeys.WEIGHTS,weights)
				tf.add_to_collection(tf.GraphKeys.WEIGHTS,bias)

			with tf.variable_scope("quadratic"):
				embedding = tf.get_variable("embedding",
					shape=[config.n,config.m,config.k],
					dtype=tf.float32,
					initializer=tf.truncated_normal_initializer(stddev=0.01,mean=0)) # [n,m,k]

				quad_term = tf.gather(embedding,self.features)
				quad_term = tf.reduce_sum(quad_term * tf.transpose(quad_term,[0,2,1,3]),-1) # [None,m,m]
				temp = []
				for i in range(config.m):
					if i != 0:
						temp.append(quad_term[:,i,:i])
				quad_term = tf.reduce_sum(tf.concat(temp,-1),-1) # [None,]

				tf.add_to_collection(tf.GraphKeys.WEIGHTS,embedding)

			logit = linear_term + quad_term
			self.prob = tf.sigmoid(logit)

	def _loss_func(self):
		config = self.config

		with tf.name_scope("l2_loss"):
			# l2 normalization
			regularizer = tf.contrib.layers.l2_regularizer(config.l2_norm)
			reg_term = tf.contrib.layers.apply_regularization(regularizer,weights_list=None)

		with tf.name_scope("logistic_loss"):
			# logistic loss
			logit_1 = tf.log(self.prob + 1e-20)
			logit_0 = tf.log(1 - self.prob + 1e-20)
			self.log_loss = -1 * tf.reduce_mean(self.label * logit_1 + (1- self.label) * logit_0)

		self.loss = self.log_loss + reg_term

	def _optimizer(self):
		config = self.config
		# build optimizer
		opt = tf.train.AdamOptimizer(config.learning_rate)

		# build train op
		params = tf.trainable_variables()
		gradients = tf.gradients(self.loss,params,colocate_gradients_with_ops=True)
		clipped_grads, gradient_norm = tf.clip_by_global_norm(gradients, 5.0)
		self.grad_norm = gradient_norm
		self.train_op = opt.apply_gradients(zip(clipped_grads, params))


def main():
	from config import config

	ffm = FFM(config)
	ffm.build()

	from dataset import DataSet
	tr_filename = "data/criteo.tr.r100.gbdt0.ffm"
	va_filename = "data/criteo.va.r100.gbdt0.ffm"
	dataset = DataSet(tr_filename,config.batch_size,config.shuffle)

	va_dataset = DataSet(va_filename)

	ffm.train(dataset,va_dataset)

	pdb.set_trace()


	return

if __name__ == '__main__':
	main()