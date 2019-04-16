# -*- coding: utf-8 -*-
import numpy as np

from config import config
import time

class Dataset(object):
    def __init__(self,filename,batch_size,num_epoch):
        self.filename = filename
        self.dataset = self.batch_generator(batch_size,num_epoch)

    def batch_generator(self,batch_size,num_epoch):
        filename = self.filename
        epoch_count = 0
        batch = 0
        feature,value,label = [],[],[]
        f = open(filename,"r")
        while True:
            line = f.readline()
            # print(batch)
            if not line:
                epoch_count+=1
                print("epoch:",epoch_count)
                if epoch_count >= num_epoch:
                    f.close()
                    break
                else:
                    f.close()
                    f = open(filename,"r")
                    continue
            else:
                res = self.parse_sample(line)
                feature.append(res[0])
                value.append(res[1])
                label.append(res[2])
                batch += 1        
            if batch >= batch_size:
                yield {"feature":np.asarray(feature,dtype="int"),
                            "value":np.asarray(value,dtype="float32"),
                            "label":np.asarray(label,dtype="int")
                            }
                feature,value,label = [],[],[]
                batch = 0

    def parse_sample(self,row):
        row = row.split()
        label = int(row[0])
        feature = []
        value = []
        for pair in row[1:]:
            feature.append(int(pair.split(":")[1]))
            value.append(float(pair.split(":")[2]))
        return feature,value,label

    def next(self):
        return self.dataset.__next__()

def main():
    start_time = time.time()
    dataset = Dataset("data/libffm_toy/criteo.va.r100.gbdt0.ffm",
        batch_size=32,num_epoch=2)
    while True:
        try:
            batch_data = dataset.next()
            print(batch_data)
            # if len(batch_data["label"]) != 32:
            #     print(len(batch_data["label"]))
        except:
            print("Iteration Stop.")
            break

    print("Run: {} secs".format(int(time.time()-start_time)))
if __name__ == '__main__':
    main()


