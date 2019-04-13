"""A tensorflow based Field-aware Factorization Machine.
"""

# -*- coding: utf-8 -*-
import numpy as np 
import tensorflow as tf 

import pdb
import os

# configurations
batch_size = 8
feature_num = 10
field_num = 2

log_path = "./log"
data_path = "./data/libffm_toy"

def main():
    # define inputs placeholders
    label = tf.placeholder(tf.float32,shape=(batch_size))
    feature_value = []
    for i in range(feature_num):
        value = tf.placeholder(tf.float32,shape=(batch_size),name="feature_{}".format(i))
        feature_value.append(value)

    # shape: [batch_size,feature_num]
    feature_value = tf.transpose(tf.convert_to_tensor(feature_value),perm=[1,0])

    # inference
    pred = inference(feature_value)

    pdb.set_trace()

def inference(feature_value):
    """Takes inference by inputs feature values.
    predict = b0 + sum(Vi * feature_i) + sum(Vij * Vji * feature_i * feature_j)

    Args:
        feature_value: A tensor with shape [batch_size,feature_num]
    Returns:
        pred: prediction by FFM
    """
    # define bias
    b0 = tf.get_variable(name="bias_0",shape=[1],dtype=tf.float32)
    tf.summary.histogram("b0",b0)

    # define linear weight
    linear_weight = tf.get_variable(name="linear_weight",
        shape=[feature_num],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.01)
        )
    tf.summary.histogram("linear_weight",linear_weight)

    # define 2nd degree embeddings, shape is [feature_num, field_num]
    field_embedding = []
    for idx in range(0,feature_num):
        vij = tf.get_variable(name="field_embedding_{}".format(idx),
                shape=[field_num],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.01)
                )
        field_embedding.append(vij)
        tf.summary.histogram("filed_vec_{}".format(idx),vij)

    # compute linear term
    linear_term = tf.multiply(feature_value,linear_weight)
    linear_term = tf.reduce_sum(linear_term,1) # [batch_size,]

    # compute quad term
    quad_term = tf.get_variable(name="quad_term",dtype=tf.float32,shape=[batch_size,])
    for j1 in range(0,feature_num-1):
        for j2 in range(j1+1,feature_num):
            # V12 = tf.nn.embedding_lookup(field_embedding[j1],feature2field[j2])
            # V21 = tf.nn.embedding_lookup(field_embedding[j2],feature2field[j1])
            # V12V21 = tf.multiply(V12,V21)
            value_conjunc = tf.multiply(feature_value[:,j1],feature_value[:,j2])
            quad_term = quad_term + value_conjunc

    pred = b0 + linear_term + quad_term # [batch_size,]
    pdb.set_trace()
    return pred

if __name__ == '__main__':
    main()


