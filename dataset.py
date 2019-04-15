# -*- coding: utf-8 -*-
import tensorflow as tf 
import pdb
import os

# configurations
data_dir = "./data"
train_path = os.path.join(data_dir,"criteo.tr.tfrecord")
val_path = os.path.join(data_dir,"criteo.va.tfrecord")

class Dataset():
    def __init__(self,config):
        self.config = config

    def get_dataset(self,tf_filenames,mode="train"):
        """Get dataset for training or validation, us tf.data.TFRecordDataset(),
        can switch between train and validation data by `make_initializable_iterator()`.

        Args:
            tf_filenames: a list of *.tfrecord file names.
            mode: a string set as `train` or `val`.
        Returns:
            output batch result from `tf.data.Iterator.get_next()` .
        """
        assert mode in ["train","val","test"]
        config = self.config

        if not isinstance(tf_filenames,list):
            tf_filenames = [tf_filenames]

        dataset = tf.data.TFRecordDataset(tf_filenames)

        if self.config.sparse_ffm == False: # not sparse tensor
            dataset = dataset.map(self.parse_function)
        else: # sparse tensor
            dataset = dataset.map(self.sparse_parse_function)

        if mode in ["train"]:
            dataset = dataset.shuffle(buffer_size=config.batch_size * 10)
            dataset = dataset.batch(config.batch_size,drop_remainder=False)
            dataset = dataset.repeat(config.num_epoch)
        elif mode in ["test","val"]:
            dataset = dataset.repeat(1)
            dataset = dataset.batch(config.val_batch_size,drop_remainder=False)

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def parse_function(self,example_proto):
        field_num = self.config.field_num
        keys_to_features = {
            "feature": tf.FixedLenFeature(shape=[field_num,],dtype=tf.int64),
            "label": tf.FixedLenFeature(shape=[1,],dtype=tf.int64),
            "value": tf.FixedLenFeature(shape=[field_num,],dtype=tf.float32)
        }
        parsed = tf.parse_single_example(example_proto, keys_to_features)

        result = {
            "feature":parsed["feature"],
            "value":parsed["value"],
            "label":parsed["label"],
        }
        return result

    def sparse_parse_function(self,example_proto):
        keys_to_features = {
            "feature": tf.VarLenFeature(dtype=tf.int64),
            "value": tf.VarLenFeature(dtype=tf.float32),
            "label": tf.FixedLenFeature(shape=[1,],dtype=tf.int64),
        }
        parsed = tf.parse_single_example(example_proto, keys_to_features)
        result = {
            "feature":parsed["feature"],
            "value":parsed["value"],
            "label":parsed["label"],
        }
        return result

def main():
    from config import config
    dataloader = Dataset(config)
    data = dataloader.get_dataset(config.val_filename,"val")
    sess = tf.Session()
    sess.run(data)
    pdb.set_trace()
    return

if __name__ == '__main__':
    main()