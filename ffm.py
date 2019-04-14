"""Tensorflow based Field-aware Factorization Machine.

TODO:
    1. early stopping
    2. model save and test
    3. write log and visualize training process

"""
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pdb
import os

from config import config
from dataset import Dataset

model_dir = config.model_dir
log_path = config.log_dir
data_dir = config.data_dir
train_dir = config.train_filename
val_dir = config.val_filename

sess_config = tf.ConfigProto(device_count = {"CPU":config.num_cpu}, # num of cpu to be used
    inter_op_parallelism_threads=0, # auto select
    intra_op_parallelism_threads=0, # auto select
    )

class FFM(object):
    def __init__(self):
        self.feature_id_map = utils.pickle_load(config.feat_map_filename)
        
    def inference(self,inputs_feature,inputs_value):
            """With inputs placeholder, do inference and get predictions.
        predict = b0 + sum(Vi * feature_i) + sum(Vij * Vji * feature_i * feature_j)

        Args:
            inputs_feature: a tensor of input feature idx in each field, shape [None,num_field]
            inputs_value: a tensor of input value of each feature, shape [None,num_field]
        Returns:
            logit: output logits tensor with shape [batch_size,1]
            pred: predictions with shape [batch_size,1]
        """
        # map input feature
        inputs_feature = self.parse_feature_id(inputs_feature)

        # build linear term
        with tf.variable_scope("linear"):
            weights = tf.get_variable("weights",
                shape=[feature_num,1],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.01,mean=0)) # [n,1]

            bias = tf.get_variable("bias",
                shape=[1,1],
                dtype=tf.float32,
                initializer=tf.zeros_initializer())

            batch_weights = tf.nn.embedding_lookup(weights,inputs_feature) # [None,f,1]
            batch_weights = tf.squeeze(batch_weights,axis=2) # [None,f]

            linear_term = tf.multiply(inputs_value,batch_weights) # [None,f]
            linear_term = tf.reduce_sum(linear_term,axis=1,keepdims=True) # [None,1]
            linear_term = tf.add(linear_term,bias) # [None,1]

        # build quadratic term
        with tf.variable_scope("quad"):
            embeddings = tf.get_variable("embeddings",
                shape=[field_num,feature_num,embedding_num],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.01,mean=0)) # [f,n,k]

            quad_term = None
            for i in range(field_num):
                for j in range(i+1,field_num):
                    vi_fj = tf.nn.embedding_lookup(embeddings[j],inputs_feature[:,i]) # [None,k]
                    vj_fi = tf.nn.embedding_lookup(embeddings[i],inputs_feature[:,j]) # [None,k]
                    wij = tf.multiply(vi_fj,vj_fi) # [None,k]

                    x_i = tf.expand_dims(inputs_value[:,i],1) # [None,1]
                    x_j = tf.expand_dims(inputs_value[:,j],1) # [None,1]
                    xij = tf.multiply(x_i,x_j) # [None,1]

                    if quad_term is None:
                        quad_term = tf.reduce_sum(tf.multiply(wij,xij),axis=1,keepdims=True)
                    else:
                        quad_term = tf.add(quad_term,tf.reduce_sum(tf.multiply(wij,xij),axis=1,keepdims=True))
        
        # build predictions
        with tf.variable_scope("output"):
            logit = tf.add(linear_term,quad_term,name="logit")
            pred = tf.nn.sigmoid(logit,name="pred")

        return logit,pred

    def parse_feature_id(self,inputs_feature):
        def func(inputs):
            pdb.set_trace()
            return inputs
        return tf.py_func(func,[inputs_feature],tf.int64)

def main():

    """
    Train phase
    """
    # define dataset  and inputs
    train_iterator = Dataset(config).get_dataset(config.train_filename,mode="train")
    train_inputs = train_iterator.get_next()
    # do inference
    logit,pred = inference(train_inputs["feature"],train_inputs["value"])
    # pdb.set_trace()
    # loss function
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(train_inputs["label"],tf.float32),logits=logit)
    loss = tf.reduce_mean(loss)
    # train op
    optimizer = tf.train.AdagradOptimizer(learning_rate=config.learning_rate)
    grad = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grad)

    """
    Validation phase
    """

    # auc = tf.metrics.auc(labels=label,predictions=pred,name="tr_auc")
    # acc = tf.metrics.accuracy(labels=label,
    #         predictions=tf.cast(pred>threshold,tf.float32),
    #         name="tr_accuracy")

    with tf.Session(config=sess_config) as sess:
        # summary writer
        # tf.summary.scalar("tr_losses",loss)
        # tf.summary.scalar("tr_auc",auc[1])
        # tf.summary.scalar("tr_accuracy",acc[1])
        # merged = tf.summary.merge_all()
        # writer = tf.summary.FileWriter(log_path,sess.graph)

        sess.run(tf.global_variables_initializer())
        while True:
            batch_loss,_ = sess.run([loss,train_op])
            print(batch_loss)
        # get batch data
        # ...

        # train
        # ...

        # validate every %d epoch
        # ...

        # do early stop and save best model every %d epoch
        # ...


    pdb.set_trace()
    pass

def inference(inputs_feature,inputs_value):
    """With inputs placeholder, do inference and get predictions.
    predict = b0 + sum(Vi * feature_i) + sum(Vij * Vji * feature_i * feature_j)

    Args:
        inputs_feature: a tensor of input feature idx in each field, shape [None,num_field]
        inputs_value: a tensor of input value of each feature, shape [None,num_field]
    Returns:
        logit: output logits tensor with shape [batch_size,1]
        pred: predictions with shape [batch_size,1]
    """
    # map input feature
    inputs_feature = tf_map_feature(inputs_feature)

    # build linear term
    with tf.variable_scope("linear"):
        weights = tf.get_variable("weights",
            shape=[feature_num,1],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.01,mean=0)) # [n,1]

        bias = tf.get_variable("bias",
            shape=[1,1],
            dtype=tf.float32,
            initializer=tf.zeros_initializer())

        batch_weights = tf.nn.embedding_lookup(weights,inputs_feature) # [None,f,1]
        batch_weights = tf.squeeze(batch_weights,axis=2) # [None,f]

        linear_term = tf.multiply(inputs_value,batch_weights) # [None,f]
        linear_term = tf.reduce_sum(linear_term,axis=1,keepdims=True) # [None,1]
        linear_term = tf.add(linear_term,bias) # [None,1]

    # build quadratic term
    with tf.variable_scope("quad"):
        embeddings = tf.get_variable("embeddings",
            shape=[field_num,feature_num,embedding_num],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.01,mean=0)) # [f,n,k]

        quad_term = None
        for i in range(field_num):
            for j in range(i+1,field_num):
                vi_fj = tf.nn.embedding_lookup(embeddings[j],inputs_feature[:,i]) # [None,k]
                vj_fi = tf.nn.embedding_lookup(embeddings[i],inputs_feature[:,j]) # [None,k]
                wij = tf.multiply(vi_fj,vj_fi) # [None,k]

                x_i = tf.expand_dims(inputs_value[:,i],1) # [None,1]
                x_j = tf.expand_dims(inputs_value[:,j],1) # [None,1]
                xij = tf.multiply(x_i,x_j) # [None,1]

                if quad_term is None:
                    quad_term = tf.reduce_sum(tf.multiply(wij,xij),axis=1,keepdims=True)
                else:
                    quad_term = tf.add(quad_term,tf.reduce_sum(tf.multiply(wij,xij),axis=1,keepdims=True))
    
    # build predictions
    with tf.variable_scope("output"):
        logit = tf.add(linear_term,quad_term,name="logit")
        pred = tf.nn.sigmoid(logit,name="pred")

    return logit,pred

def tf_map_feature(inputs_feature):
    pass


if __name__ == '__main__':
    main()




