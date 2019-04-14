# -*- coding: utf-8 -*-
import pickle
import pdb

def pickle_load(filename):
    with open(filename,"rb") as f:
        data = pickle.load(f)
    return data

def pickle_save(filename,data):
    with open(filename,"wb") as f:
        pickle.dump(data,f)
    return