# -*- coding: utf-8 -*-
import linecache
import numpy as np

import time

import pdb

class DataSet(object):
	def __init__(self,filename,batch_size=None,shuffle=True):
		self.filename = filename
		self.total_num = self._count_data_len(filename)
		self.raw_data = linecache.getlines(filename)
		if batch_size is not None:
			self.batch_gen = self._batch_generator(batch_size,shuffle)

	def get_batch(self):
		return self.batch_gen.__next__()

	def get_all_data(self):
		feature_data = []
		label_data = []
		for data in self.raw_data:
			_,feat,_,label = self._parse_line(data)
			feature_data.append(feat)
			label_data.append(label)

		all_data = {"feature":np.array(feature_data),
					"label":np.array(label_data)}
		self.all_data = all_data
		return self.all_data

	def stats(self):
		"""Get statistics of the dataset.
		"""
		m = n = 0
		for data in self.raw_data:
			field, feat, _, _ = self._parse_line(data)
			m = max(m, np.max(field) + 1)
			n = max(n, np.max(feat) + 1)
		self.total_field = m
		self.total_feature = n
		print("-"*10 + ">Data Statistics<" + "-"*10)
		print("Num of feature", n)
		print("Num of field", m)
		return

	def _batch_generator(self,batch_size,shuffle=True):
		indices = list(range(self.total_num))
		if shuffle:
			np.random.seed(724)
			np.random.shuffle(indices)

		batch_count = 0
		while True:
			if (batch_count + 1) * batch_size > self.total_num:
				batch_count = 0
				if shuffle:
					np.random.shuffle(indices)

			start_idx = batch_count * batch_size
			end_idx = start_idx + batch_size
			batch_count += 1

			batch_data = self.raw_data[start_idx:end_idx]
			batch_field, batch_feature, batch_val, batch_label = [],[],[],[]
			for b in batch_data:
				field,feat,val,label = self._parse_line(b)
				batch_field.append(field)
				batch_feature.append(feat)
				batch_val.append(val)
				batch_label.append(label)

			yield np.array(batch_field),np.array(batch_feature),\
				np.array(batch_val),np.array(batch_label)

	def _parse_line(self,line):
		line = line.strip().split()
		label = int(line[0])
		line_data = np.array([l.split(":") for l in line[1:]])
		field_idx = line_data[:,0].astype(int)
		feat_idx = line_data[:,1].astype(int)
		vals  = line_data[:,2].astype(np.float32)
		return field_idx,feat_idx,vals,label

	def _count_data_len(self,filename):
		with open(filename) as f:
			nr_of_lines = sum(1 for line in f)
		return nr_of_lines


def main():
	# debug
	filename = "data/criteo.va.r100.gbdt0.ffm"
	dataset = DataSet(filename,32,True)
	for i in range(10):
		batch = dataset.get_batch()
	print(batch)

	dataset.stats()


if __name__ == '__main__':
	main()
