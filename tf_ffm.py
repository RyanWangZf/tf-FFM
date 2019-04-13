# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

feature_num = 10
field_num = 2

def main():

    # define placeholders
    label = tf.placeholder(tf.float32,shape=(None,))



    pass

def inference(inputs):
    """With inputs placeholder, do inference and get predictions.
    predict = b0 + sum(Vi * feature_i) + sum(Vij * Vji * feature_i * feature_j)

    Args:
        inputs: A inputs tensor with shape [batch_size,feature_num]
    Returns:
        pred: predictions with shape [batch_size,1]
    """
    with tf.variable_scope("linear"):
        bias = tf.get_variable("bias",
            shape=[1,1],
            dtype=tf.float32,
            initializer=tf.initializers.zeros()) # [1,1]
        weights = tf.get_variable("weights",
            shape=[feature_num,1],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.01)) # [feature_num,1]

        batch_weights = tf.nn.embedding_lookup(weights,)






    return



if __name__ == '__main__':
    main()




