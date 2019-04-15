"""Achieve ffm with sparse tensor.
"""
# -*- coding: utf-8 -*-

import tensorflow as tf 
import numpy as np 
import pdb

from config import config

class SparseFFM(object):
    def __init__(self,config):
        self.config = config

    def build_model(self,data_element,mode="train"):
        assert mode in ["train","test","val"]
        inputs_feature,inputs_value = data_element["feature"],data_element["value"]
        if mode in ["train","val"]:
            inputs_label = data_element["label"]
            logit,pred = self.inference(inputs_feature,inputs_value)
            losses = self.loss_function(logit,inputs_label)

            self.losses = losses
            self.pred = pred

            optimizer = tf.train.AdagradOptimizer(learning_rate=config.learning_rate)
            self.train_op = optimizer.minimize(losses,global_step=tf.train.get_global_step())

        pdb.set_trace()
        return

    def train(self):
        pdb.set_trace()
        config = self.config
        # sess_config = tf.ConfigProto(
        #         device_count = {"CPU":config.num_cpu}, # num of cpu to be used
        #         inter_op_parallelism_threads=0, # auto select
        #         intra_op_parallelism_threads=0, # auto select
        #         )
        # self.sess = tf.Session(config=sess_config)
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        epoch = 0
        while True:
            try:
                batch_loss,_ = self.sess.run([self.losses,self.train_op])
            except tf.errors.OutOfRangeError:
                break
            print("batch {}, loss {}".format(epoch,batch_loss))
            epoch += 1

        return

    def loss_function(self,logit,label):
        losses = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(label,tf.float32),
            logits=logit)
        losses = tf.reduce_mean(losses)
        # TODO
        # Add l2_norm loss
        return losses

    def inference(self,inputs_feature,inputs_value):
        config = self.config
        feature_num = config.feature_num
        field_num = config.field_num
        embedding_num = config.embedding_num
        # print("sparse tensor: {0},\n {1}".format(inputs_feature, inputs_value))

        # linear term
        with tf.variable_scope("linear"):
            weights = tf.get_variable("weights",
                shape=[feature_num,1],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.01,mean=0)) # [n,1]

            bias = tf.get_variable("bias",
                shape=[1],
                dtype=tf.float32,
                initializer=tf.zeros_initializer())

            linear_term = tf.nn.embedding_lookup_sparse(weights,
                    inputs_feature,
                    inputs_value,combiner="sum")
            linear_term = tf.add(linear_term,bias) # [None,1]

        with tf.variable_scope("quadratic"):
            embeddings = tf.get_variable("embeddings",
                shape=[field_num,feature_num,embedding_num],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.01,mean=0)) # [f,n,k]

            quad_term = None
            pdb.set_trace()
            # TODO
            # something wrong here, wait to be solved !!!
            for i in range(field_num):
                for j in range(i+1,field_num):
                    vi_fj = tf.nn.embedding_lookup_sparse(embeddings[j], # [n,k]
                        inputs_feature,
                        inputs_value,
                        combiner="sum",
                        ) # [None,k]
                    vj_fi = tf.nn.embedding_lookup_sparse(embeddings[i], # [n,k]
                        inputs_feature,
                        inputs_value,
                        combiner="sum",
                        ) # [None,k]

                    if quad_term is None:
                        quad_term = tf.reduce_sum(tf.multiply(vi_fj,vj_fi),axis=1,keepdims=True) # [None,1]
                    else:
                        quad_term = tf.add(quad_term,
                                tf.reduce_sum(tf.multiply(vi_fj,vj_fi),axis=1,keepdims=True))
            
        # build output
        with tf.variable_scope("output"):
            logit = tf.add(linear_term,quad_term,name="logit")
            pred = tf.nn.sigmoid(logit,name="pred")

        # debug
        # sess = tf.Session()
        # sess.run(tf.global_variables_initializer())
        # sess.run(pred)

        return logit,pred

def main():
    from dataset import Dataset
    dataloader = Dataset(config)
    data = dataloader.get_dataset(config.train_filename,mode="train")

    ffm = SparseFFM(config)
    ffm.build_model(data,"train")
    ffm.train()

if __name__ == '__main__':
    main()