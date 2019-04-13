"""Convert libFFM format data to tfrecord file.
"""
# -*- coding: utf-8 -*-
import tensorflow as tf 
import os
import pdb
import numpy as np

# configurations
data_dir = "./data/libffm_toy"
tr_path = os.path.join(data_dir,"criteo.tr.r100.gbdt0.ffm")
va_path = os.path.join(data_dir,"criteo.va.r100.gbdt0.ffm")
read_names=[tr_path,va_path]

def main():
    # field num,feature num, train set num, validate set num
    data_stat = [0,0,0,0]

    # save path
    save_names = [os.path.join("./data","{}.tfrecord".format(k)) for k in ["criteo.tr","criteo.va"]]
    
    for i,filename in enumerate(zip(read_names,save_names)):
        read_name,save_name = filename
        print("Read from {}\nSave in {}".format(read_name,save_name))
        with tf.python_io.TFRecordWriter(save_name) as tfrecord_writer:
            f = open(read_name,"r")
            for count,row in enumerate(f.readlines()):
                feature,value,label = process_sample(row,data_stat)
                example = convert_to_example(feature,value,label)
                tfrecord_writer.write(example.SerializeToString())
            if i == 0:
                data_stat[2] = count+1
            else:
                data_stat[3] = count+1
            f.close()

    print("\n=======================> dataset summary <================================\n")
    print("[Train set]:",data_stat[2])
    print("[Validate set]:",data_stat[3])
    print("[Number of field]: ",data_stat[0])
    print("[Number of feature]",data_stat[1])
    print("\n=======================> dataset summary <================================\n")

def process_sample(row,data_stat=[0,0,0,0]):
    row = row.split()
    label = int(row[0])
    feature = []
    value = []
    for pair in row[1:]:
       feature.append(int(pair.split(":")[1]))
       value.append(float(pair.split(":")[2]))

    max_feature = np.max(feature)
    max_field = len(feature)
    if max_feature > data_stat[1]:
        data_stat[1] = max_feature
    if max_field > data_stat[0]:
        data_stat[0] = max_field

    return feature,value,label

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

