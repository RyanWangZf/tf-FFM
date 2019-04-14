"""Convert libFFM format data to tfrecord file.
"""
# -*- coding: utf-8 -*-
import tensorflow as tf 
import os
import pdb
import numpy as np

import utils
from config import config

# configurations
data_dir = config.libffm_data_dir
tr_path = config.tr_path
va_path = config.va_path
read_names= config.read_names
save_names = config.save_names

def main():
    # first loop, encode feature to feature_id
    print("Encode features...")
    feature_id = []
    for name in read_names:
        f = open(name,"r")
        for row in f.readlines():
                feature,value,label = process_sample(row)
                feature_id.extend(feature)

        f.close()
        feature_id = np.unique(feature_id).tolist()
    print("Encode feature done.")

    # save feat_id map dict
    feat_id_dict = dict()
    for i,feat_ in enumerate(feature_id):
        feat_id_dict[feat_] = i
    utils.pickle_save(config.feat_map_filename,feat_id_dict)
    print("Convert to tfrecord...")

    # second loop, save tfrecord file, use dict to encode feature id
    data_stat = [0,0]     # field num,feature num, train set num, validate set num
    # save path    
    for i,filename in enumerate(zip(read_names,save_names)):
        read_name,save_name = filename
        print("Read from {}\nSave in {}".format(read_name,save_name))
        with tf.python_io.TFRecordWriter(save_name) as tfrecord_writer:
            f = open(read_name,"r")
            for count,row in enumerate(f.readlines()):
                feature,value,label = process_sample(row)
                feature = encode_feature(feature,feat_id_dict)
                example = convert_to_example(feature,value,label)
                tfrecord_writer.write(example.SerializeToString())
            if i == 0:
                data_stat[0] = count+1
            else:
                data_stat[1] = count+1
            f.close()

    print("Convert Done.")
    print("\n=======================> dataset summary <================================\n")
    print("[Train set]:",data_stat[0])
    print("[Validate set]:",data_stat[1])
    print("[Number of field]: ",len(feature))
    print("[Number of feature]",len(feature_id))
    print("\n=======================> dataset summary <================================\n")

def process_sample(row):
    row = row.split()
    label = int(row[0])
    feature = []
    value = []
    for pair in row[1:]:
       feature.append(int(pair.split(":")[1]))
       value.append(float(pair.split(":")[2]))

    return feature,value,label

def encode_feature(feature,feat_id_dict):
    for i in range(len(feature)):
        feature[i] = feat_id_dict[feature[i]]
    return feature

def convert_to_example(feature,value,label):
    example = tf.train.Example(
            features = tf.train.Features(feature={
                    "feature":int64_feature(feature), # [num_field,]
                    "label":int64_feature(label),      # [1,]
                    "value":float_feature(value),      # [num_field,]
                }))
    return example

def float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value,list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto
    """
    if not isinstance(value,list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto
    """
    if not isinstance(value,list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


if __name__ == '__main__':
    main()

