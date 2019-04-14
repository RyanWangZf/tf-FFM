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

import utils

class FFM(object):
    def __init__(self,config):
        self.config = config
        self.feature_id_map = utils.pickle_load(config.feat_map_filename)

    def build_model(self):
        config = self.config

        self.inputs_feature = tf.placeholder(shape=[None,config.field_num],
                dtype=tf.int64,name="inputs_feature")
        self.inputs_value = tf.placeholder(shape=[None,config.field_num],
                dtype=tf.float32,name="inputs_value")
        self.label = tf.placeholder(shape=[None,1],
                dtype=tf.int64,name="inputs_label")

        self.logit,self.pred = self.inference(self.inputs_feature,self.inputs_value)
        self.losses = self.loss_function(self.logit,self.label)

        optimizer = tf.train.AdagradOptimizer(learning_rate=config.learning_rate)
        grad = optimizer.compute_gradients(self.losses)
        # grad = tf.clip_by_value(grad, clip_value_min=,clip_value_max=10.0, name="cliped_grad")
        # TODO apply gradient clip

        self.train_op = optimizer.apply_gradients(grad)

    def train(self,data_element):
        
        return

    def loss_function(self,logit,label):
        losses = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(label,tf.float32),
            logits=logit)
        losses = tf.reduce_mean(losses)

        self.log_loss = losses # save the logistic loss

        # add l2 loss for training
        # TODO
        # losses = add_l2_loss(losses)

        return losses

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
        config = self.config
        feature_num = config.feature_num
        field_num = config.field_num
        embedding_num = config.embedding_num

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

def main():
    from config import config
    ffm = FFM(config)
    pdb.set_trace()

if __name__ == '__main__':
    main()