import datetime
import pickle
import os
import numpy as np

def make_name_prefix():
    time_ = datetime.datetime.now()
    day = time_.strftime("%d")
    month = time_.strftime("%m")
    hour = time_.strftime("%H")
    minutes = time_.strftime("%M")
    unique = day + month + '_' + hour + minutes
    return unique

def save(data, filename):
    filename = os.path.join(filename+'.pkl')
    with open(filename, "wb") as f:
        pickle.dump(data, f)
        del data
    print(f'file saved to: {filename}')

def load(file_name):
    data = pickle.load(file=open(file_name, "rb"))
    return data

def dict_to_np(d, type):
    return np.array(list(d.values()), dtype=type)
