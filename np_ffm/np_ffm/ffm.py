"""An numpy based field-aware factorization machine.
"""
# -*- coding: utf-8 -*-
import numpy as np 
import pdb

from config import config 
from dataset import Dataset

class FFM(object):
    def __init__(self,config):
        self.config = config

    def build_model(self):
        n = config.feature_num
        f = config.field_num
        k = config.embedding_num

        # initialize embeddings with truncated normal distribution
        embedding = np.random.randn(f,n,k)
        embedding[np.where(embedding > 3)] = 0.0
        embedding[np.where(embedding < -3)] = 0.0
        self.embedding = embedding

    def inference(self,feature,value):
        """Compute quadratic and linear terms.

        Args:
            feature: numpy.ndarray with format [field_0: feature_id, field_1: feature_id,...]
                the shape is [batch_size,num_field]
            value: numpy.ndarry with format [feature:value, feature:value,...]
                the shape is [batch_size,num_field]

        Returns:
            the computed term of inputs, with shape [batch_size,]
        """
        n = config.feature_num
        f = config.field_num
        k = config.embedding_num
        embedding = self.embedding

        # TODO: linear term
        # quad term
        quad_term = None
        for i in range(f):
            for j in range(i+1,f):
                vi_fj = embedding[j,feature[:,i]]
                vj_fi = embedding[i,feature[:,j]]
                if quad_term is None:
                    quad_term = np.dot(vi_fj,vj_fi.T)* value[:,i] * value[:,j]
                else:
                    quad_term += np.dot(vi_fj,vj_fi.T) * value[:,i] * value[:,j]
                # print(i,j)
        quad_term = np.diag(quad_term)
        logit = quad_term
        pred = 1 / (1 + np.exp(-logit))
        return logit,pred

def main():
    # get data
    train_dataset = Dataset(config.train_filename,
                                            config.batch_size,
                                            config.num_epoch)
    # build model
    ffm = FFM(config)
    ffm.build_model()
    for i in range(1):
        batch_train = train_dataset.next()
        feature = batch_train["feature"]
        value = batch_train["value"]
        res = ffm.inference(feature,value)
        print(res)
        pdb.set_trace()



if __name__ == '__main__':
    main()



