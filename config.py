"""Train and Test config.
"""
# -*- coding: utf-8 -*-
import os

class Config():
    """dir and path"""
    # for convert_to_tfrecord.py
    libffm_data_dir = "./data/libffm_toy"
    tr_path = os.path.join(libffm_data_dir,"criteo.tr.r100.gbdt0.ffm")
    va_path = os.path.join(libffm_data_dir,"criteo.va.r100.gbdt0.ffm")
    read_names= [tr_path,va_path]
    save_names =  [os.path.join("./data","{}.tfrecord".format(k)) for k in ["criteo.tr","criteo.va"]]

    # for ffm.py
    sparse_ffm = True
    model_dir = "./ckpt"
    log_dir = "./log"
    data_dir = "./data"
    train_filename = os.path.join(data_dir,"criteo.tr.tfrecord")
    val_filename = os.path.join(data_dir,"criteo.va.tfrecord")
    feat_map_filename = os.path.join(data_dir,"feat_id.pkl")

    """dataset statistic"""
    field_num = 39 # val_filename
    feature_num = 303943 # num_epoch
    embedding_num = 4 # k

    """train setup"""
    threshold = 0.5
    learning_rate = 0.1
    num_epoch = 2
    early_stop_patience = 5
    l2_param = 0.0002
    batch_size = 32

    """val and test setup"""
    val_batch_size = 2

    """others"""
    num_cpu = 2


config = Config()

