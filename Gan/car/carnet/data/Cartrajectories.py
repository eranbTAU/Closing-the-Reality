from torch.utils.data import DataLoader
import logging
import os
import math
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

def Normalization(x, x_min, x_max):
    x_nor = (x - x_min) / (x_max - x_min)
    x_nor = torch.from_numpy(x_nor).type(torch.float)
    return x_nor

def CarNormalization(pred_seq, real_seq, x_min=-120, x_max=120, dx_min=-24.9079, dx_max=24.6524,
                     dy_min=-49.7925, dy_max=49.9884, dtheta_min=-1.3976, dtheta_max=1.3989):

    x_stand = Normalization(pred_seq, x_min, x_max)
    y_dx = Normalization(real_seq[:,0], dx_min, dx_max).view(-1,1)
    y_dy = Normalization(real_seq[:,1], dy_min, dy_max).view(-1,1)
    y_dtheta = Normalization(real_seq[:,2], dtheta_min, dtheta_max).view(-1,1)
    y_stand = torch.cat([y_dx, y_dy, y_dtheta], dim=1)
    return x_stand, y_stand

def seq_collate(data):
    (pred_seq, real_seq) = zip(*data)
    pred_seq = np.asarray([t.numpy() for t in pred_seq])
    real_seq = np.asarray([t.numpy() for t in real_seq])

    # Normalize the data
    pred_seq_stand, real_seq_stand = CarNormalization(pred_seq, real_seq)
    out = [
        pred_seq_stand, real_seq_stand]
    return tuple(out)

def calc_angle(vector_1, vector_2):  # normalized!!
    angle = np.arctan2(vector_2[1], vector_2[0]) - np.arctan2(vector_1[1], vector_1[0])
    if angle > np.pi:
        angle -= 2 * np.pi
    elif angle <= -np.pi:
        angle += 2 * np.pi
    return angle  # returns angle in radians (-pi,pi]

class TrajectoryDataset(Dataset):
    def __init__(self, data_dir):
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        Dtrain = []
        for path in all_files:
            data = pickle.load(file=open(path, "rb"))

            if not type(data[0]).__module__ == 'numpy':
                data[0] = data[0].numpy()
                data[1] = data[1].numpy()
            Dtrain += data
        # Convert numpy -> Torch Tensor
        # DATA = [np.c_[]]
        if not len(Dtrain) == 2:
            action1 = np.array(Dtrain[0])
            labels1 = np.array(Dtrain[1])
            action2 = np.array(Dtrain[2])
            labels2 = np.array(Dtrain[3])
            action = np.concatenate((action1, action2), axis=0)
            labels = np.concatenate((labels1, labels2), axis=0)
        else:
            action = np.array(Dtrain[0])
            labels = np.array(Dtrain[1])
        self.action_x = torch.from_numpy(
            action).type(torch.float)
        self.state_y = torch.from_numpy(
            labels).type(torch.float)
        self.num_samples = len(action)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        out = [
            self.action_x[index], self.state_y[index]
        ]
        return out

