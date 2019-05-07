import linecache
import numpy as np
import tensorflow as tf
import pdb

class Dataset(object):
    """A data generator based on tf.data.Dataset API.
    """
    def __init__(self,tr_filename,va_filename=None,batch_size=None,shuffle=True):
        self.all_data = []
        self.tr_filename = tr_filename
        self.va_filename = va_filename
        self.batch_size = batch_size
        self.shuffle = shuffle

        # create training dataset
        self._load_and_parse_raw_data(tr_filename)
        self.feat_plhd = tf.placeholder(dtype = tf.int32,
            shape = [None,self.all_data[0]["feature"].shape[1]],
            name = "input_feature")
        self.label_plhd = tf.placeholder(dtype = tf.float32,
            shape = [None,],
            name = "input_label")

        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.feat_plhd,self.label_plhd))

    # create validating dataset
    if va_filename is not None:
        self.val_dataset = tf.data.Dataset.from_tensor_slices((self.feat_plhd,self.label_plhd))
        self._load_and_parse_raw_data(va_filename)
        self._make_init_op()

    def get_batch(self):
        return self.next_element

    def init_iterator(self,sess,is_training=True):
        if is_training:
            sess.run(self.train_init_op,
                feed_dict={self.feat_plhd:self.all_data[0]["feature"],
                self.label_plhd:self.all_data[0]["label"]})
        else:
            sess.run(self.val_init_op,
                feed_dict={self.feat_plhd:self.all_data[1]["feature"],
                self.label_plhd:self.all_data[1]["label"]})

    def _make_init_op(self):
        self.train_dataset = self.train_dataset.shuffle(buffer_size=10000)
        self.train_dataset = self.train_dataset.batch(self.batch_size)
        # self.train_dataset = self.train_dataset.batch(self.batch_size,drop_remainder=True)
        self.train_dataset = self.train_dataset.repeat()

        # pdb.set_trace()
        self.iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types,
            self.train_dataset.output_shapes)
        self.train_init_op = self.iterator.make_initializer(self.train_dataset)

        if self.va_filename is not None:
            self.val_dataset = self.val_dataset.batch(8196)
            self.val_init_op = self.iterator.make_initializer(self.val_dataset)

        self.next_element = self.iterator.get_next()

    def _load_and_parse_raw_data(self,filename):
        print("=> [INFO] Load and parse raw data from {} ... <=".format(filename))
        raw_data = linecache.getlines(filename)
        feature_data = []
        label_data = []
        for line in raw_data:
            _,feat,_,label = self._parse_line(line)
            feature_data.append(feat)
            label_data.append(label)
        self.all_data.append({ "feature" : np.array(feature_data), "label" : np.array(label_data)})

    def _parse_line(self,line):
        line = line.strip().split()
        label = np.float32(line[0])
        line_data = np.array([l.split(":") for l in line[1:]])
        field_idx = line_data[:,0].astype(np.int32)
        feat_idx = line_data[:,1].astype(np.int32)
        vals  = line_data[:,2].astype(np.float32)
        return field_idx,feat_idx,vals,label

def main():
    """A tutorial on how to use this dataset API.
    """
    from config import config
    tr_filename = "data/criteo.tr.r100.gbdt0.ffm"
    va_filename = "data/criteo.va.r100.gbdt0.ffm"
    dataset = Dataset(tr_filename,va_filename,config.batch_size,config.shuffle)
    batch = dataset.get_batch()

    sess = tf.Session()
    for i in range(10):
        dataset.init_iterator(sess,True)
        for j in range(10):
            print("{}/{}".format(i,j))
            tr_batch = sess.run(batch)

        dataset.init_iterator(sess,False)
        va_count = 0
        try:
            while True:
                print("va",va_count)
                va_batch = sess.run(batch)
                va_count += 1
        except tf.errors.OutOfRangeError:
            print("validate done!")

    print("validation batch size:",va_batch[0].shape)
    print("training batch size:",tr_batch[0].shape)

    return

if __name__ == '__main__':
    main()
