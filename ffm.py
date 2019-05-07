# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np 
from sklearn.metrics import roc_auc_score

import time

import sys
import pdb

# TODO
# early stop
# load and predict

class FFM:
    def __init__(self,config):
        self.config = config
        # build session
        sess_config = tf.ConfigProto(device_count = {"CPU":self.config.num_cpu},
            inter_op_parallelism_threads=0,
            intra_op_parallelism_threads=0)
        self.sess = tf.Session(config=sess_config)

    def train(self,dataset):
        config = self.config
        # start training
        sample_total_num = dataset.all_data[0]["feature"].shape[0]
        num_iter_one_epoch = sample_total_num // config.batch_size
        feat_tensor,label_tensor = dataset.get_batch()

        # build graph
        pred = self._inference(feat_tensor)
        # pred = self._inference_complex(feat_tensor)

        loss, logloss = self._loss_func(pred,label_tensor)
        train_op = self._optimizer(loss)

        self.sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

        # var_list = tf.contrib.framework.get_variables_to_restore(include=["ffm"])
        # try:
        #   saver_to_restore = tf.train.Saver(var_list=var_list)
        #   saver_to_restore.restore(self.sess,config.model_dir)
        # except:
        #   print("[WARNING] Cannot load checkpoint from {}.".format(config.model_dir))
        
        saver = tf.train.Saver(max_to_keep=2)

        for step in range(config.num_epoch):
            # init training dataset
            dataset.init_iterator(self.sess,is_training=True)

            epoch_start_time = time.time()
            for iteration in range(num_iter_one_epoch):
                
                iter_start_time = time.time()
                batch_logloss,_ = \
                     self.sess.run([logloss,train_op])
                one_iter_time = time.time() - iter_start_time

                sys.stdout.write("\r")
                sys.stdout.write("=> [INFO] Process {:.0%} in Epoch {:d}: [Train] log-loss: {:.5f} one iter {:.1f} sec <= \r".format(
                        iteration/num_iter_one_epoch, step+1,batch_logloss,one_iter_time))
                sys.stdout.flush()
                # debug
                if iteration == 10:
                    break

            if dataset.va_filename is not None:
                print("\n")
                # init va dataset
                dataset.init_iterator(self.sess,is_training=False)
                va_epoch_loss = 0
                val_count = 0
                va_pred = []
                try:
                    while True:
                        va_b_epoch_loss,va_b_pred = self.sess.run([logloss,pred])
                        va_epoch_loss += va_b_epoch_loss
                        val_count += 1
                        va_pred.extend(va_b_pred.tolist())

                except tf.errors.OutOfRangeError:
                    va_pred = np.array(va_pred)
                    val_auc = roc_auc_score(dataset.all_data[1]["label"],va_pred)
                    epoch_time = time.time() - epoch_start_time
                    print("=> [INFO] STEP {}, [Val] val_loss: {:.5f}, val_auc: {:.3f} one epoch in {:.1f} sec <=".format(
                        step+1,va_epoch_loss/val_count,val_auc, epoch_time))
            
            # save model
            saver.save(self.sess,config.model_dir+"/ffm.ckpt",global_step=step+1)

    def _optimizer(self,loss):
        config = self.config
        # build optimizer
        opt = tf.train.AdamOptimizer(config.learning_rate)

        # build train op
        params = tf.trainable_variables()
        gradients = tf.gradients(loss,params,colocate_gradients_with_ops=True)
        clipped_grads, gradient_norm = tf.clip_by_global_norm(gradients, 5.0)
        train_op = opt.apply_gradients(zip(clipped_grads, params))
        return train_op

    def _loss_func(self,pred,label):
        config = self.config

        with tf.name_scope("l2_loss"):
            # l2 normalization
            regularizer = tf.contrib.layers.l2_regularizer(config.l2_norm)
            reg_term = tf.contrib.layers.apply_regularization(regularizer,weights_list=None)

        with tf.name_scope("logistic_loss"):
            # logistic loss
            logit_1 = tf.log(pred + 1e-10)
            logit_0 = tf.log(1 - pred + 1e-10)
            log_loss = -1 * tf.reduce_mean(label * logit_1 + (1- label) * logit_0)  
            # log_loss = tf.losses.log_loss(label,pred,epsilon=1e-10)

        total_loss = log_loss + reg_term

        return total_loss,log_loss

    def _inference(self,feat_tensor):
        config = self.config
        with tf.variable_scope("ffm"):
            with tf.variable_scope("linear"):
                weights = tf.get_variable("weights",
                    shape=[config.n,1],
                    dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=0.1,mean=0)) # [n,1]

                bias = tf.get_variable("bias",
                    shape=[1],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer())

                linear_term = tf.gather(weights,feat_tensor) # [None,m,1]
                linear_term = tf.add(bias, tf.reduce_sum(linear_term,[-1,-2])) # [None,]
                # linear_term = tf.reduce_sum(linear_term,[-1,-2]) # [None,]

                tf.add_to_collection(tf.GraphKeys.WEIGHTS,weights)
                tf.add_to_collection(tf.GraphKeys.WEIGHTS,bias)

            with tf.variable_scope("quadratic"):
                embedding = tf.get_variable("embedding",
                    shape=[config.n,config.m,config.k],
                    dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=0.1,mean=0)) # [n,m,k]

                quad_term = tf.gather(embedding,feat_tensor)
                quad_term = tf.reduce_sum(quad_term * tf.transpose(quad_term,[0,2,1,3]),-1) # [None,m,m]
                temp = []
                for i in range(config.m):
                    if i != 0:
                        temp.append(quad_term[:,i,:i])
                quad_term = tf.reduce_sum(tf.concat(temp,-1),-1) # [None,]

                tf.add_to_collection(tf.GraphKeys.WEIGHTS,embedding)

            logit = linear_term + quad_term
            prob = tf.sigmoid(logit)        

        return prob

    def _inference_complex(self,feat_tensor):
        """Deprecated, too time-consuming.
        """
        config = self.config
        with tf.variable_scope("ffm"):
            with tf.variable_scope("linear"):
                weights = tf.get_variable("weights",
                    shape=[config.n,1],
                    dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=0.1,mean=0))

                bias = tf.get_variable("bias",
                    shape=[1],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer())

                linear_term = tf.nn.embedding_lookup(weights,feat_tensor) # None,n,1
                linear_term = tf.squeeze(linear_term,axis=2)
                linear_term = tf.add(tf.reduce_sum(linear_term,1),bias)
                tf.add_to_collection(tf.GraphKeys.WEIGHTS,weights)
                tf.add_to_collection(tf.GraphKeys.WEIGHTS,bias)

            with tf.variable_scope("quadratic"):
                embedding = tf.get_variable("embedding",
                    shape=[config.m,config.n,config.k],
                    dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=0.1,mean=0)) # [n,m,k]
                quad_term = None
                for i in range(config.m):
                    for j in range(i+1,config.m):
                        vi_fj = tf.nn.embedding_lookup(embedding[j], feat_tensor[:,i]) # None,k
                        vj_fi = tf.nn.embedding_lookup(embedding[i], feat_tensor[:,j]) # None,k
                        vij = tf.multiply(vi_fj, vj_fi)
                        if quad_term is None:
                            quad_term = tf.reduce_sum(vij,1)
                        else:
                            quad_term += tf.reduce_sum(vij,1)

                tf.add_to_collection(tf.GraphKeys.WEIGHTS,embedding)

            logit = linear_term + quad_term
            prob = tf.sigmoid(logit)        

        return prob


def train_ffm():
    from config import config
    from dataset import Dataset

    ffm = FFM(config)
    tr_filename = "data/criteo.tr.r100.gbdt0.ffm"
    va_filename = "data/criteo.va.r100.gbdt0.ffm"
    dataset = Dataset(tr_filename,va_filename,config.batch_size,config.shuffle)
    ffm.train(dataset)

    pdb.set_trace()


    return

if __name__ == '__main__':
    train_ffm()