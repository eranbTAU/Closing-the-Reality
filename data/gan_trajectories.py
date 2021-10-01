import logging
import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

def standardization(pred_seq, real_seq):
    x, y = pred_seq, real_seq
    meansx, stdsx = x.mean(axis=0), x.std(axis=0)
    meansy, stdsy = y.mean(axis=0), y.std(axis=0)
    x_stand = (x - meansx) / (1e-7 + stdsx)
    y_stand = (y - meansy) / (1e-7 + stdsy)
    x_stand = torch.from_numpy(x_stand).type(torch.float)
    y_stand = torch.from_numpy(y_stand).type(torch.float)
    return x_stand, y_stand

def Normalization(x, x_min, x_max):
    x_nor = (x - x_min) / (x_max - x_min)
    x_nor = torch.from_numpy(x_nor).type(torch.float)
    return x_nor

def CarNormalization(pred_seq, real_seq, x_min=-120, x_max=120, dx_min=-21.9717, dx_max=21.9717,
                     dy_min=-44.9300, dy_max=44.9875, dtheta_min=-1.3000, dtheta_max=1.2928
):
    x_stand = Normalization(pred_seq, x_min, x_max)
    y_dx = Normalization(real_seq[:,0], dx_min, dx_max).view(-1,1)
    y_dy = Normalization(real_seq[:,1], dy_min, dy_max).view(-1,1)
    y_dtheta = Normalization(real_seq[:,2], dtheta_min, dtheta_max).view(-1,1)
    y_stand = torch.cat([y_dx, y_dy, y_dtheta], dim=1)
    return x_stand, y_stand

def seq_collate_gan(data):
    (pred_seq, real_seq) = zip(*data)
    pred_seq = np.asarray([t.numpy() for t in pred_seq])
    real_seq = np.asarray([t.numpy() for t in real_seq])

    # Normalize the data
    pred_seq_stand, real_seq_stand = CarNormalization(pred_seq, real_seq)
    out = [
        pred_seq_stand, real_seq_stand]
    return tuple(out)

class TrajectoryDataset(Dataset):
    def __init__(self, data_dir):
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        Dtrain = []
        for path in all_files:
            data = pickle.load(file=open(path, "rb"))
            Dtrain += data

        action, state = Dtrain
        # Convert numpy -> Torch Tensor
        self.action_x = torch.from_numpy(
            action).type(torch.float)
        self.state_y = torch.from_numpy(
            state).type(torch.float)
        self.num_samples = len(action)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        out = [
            self.action_x[index], self.state_y[index]
        ]
        return out


class Datasets(Dataset):
    def __init__(self, data_dir, validation):
        super(Datasets, self).__init__()

        if not validation:
            self.data_dir = data_dir
            all_files = os.listdir(self.data_dir)
            all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
            for path in all_files:
                data = pickle.load(file=open(path, "rb"))
                data = np.array(data)
                action, state = data[:, :2], data[:, 2:]
        else:
            data = pickle.load(file=open(data_dir, "rb"))
            action, state = data[0], data[1]



        # Convert numpy -> Torch Tensor
        self.action_x = torch.from_numpy(
            action).type(torch.float)
        self.state_y = torch.from_numpy(
            state).type(torch.float)
        self.num_samples = len(action)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        out = [
            self.action_x[index], self.state_y[index]
        ]
        return out