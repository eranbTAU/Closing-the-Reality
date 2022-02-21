import logging
import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

def standardization(pred_seq, real_seq, x_mean=0.5058, x_std=0.2366, y_mean=0.4997, y_std=0.2258, a_mean=0.4922, a_std=0.2924,
                    an_mean=0.7162, an_std=37.6194, am_mean=28.0406, am_std=16.8835, d_mean=6.0158, d_std=1.5773,
                    dx_mean=-2.5119e-01, dx_std=22.7929, dy_mean=3.1860e+01, dy_std=32.6926, dtheta_mean=-1.4515e-02, dtheta_std=0.5072):
    """
    x, y = pred_seq, real_seq
    meansx, stdsx = x.mean(axis=0), x.std(axis=0)
    meansy, stdsy = y.mean(axis=0), y.std(axis=0)
    x_stand = (x - meansx) / (1e-7 + stdsx)
    y_stand = (y - meansy) / (1e-7 + stdsy)
    x_stand = torch.from_numpy(x_stand).type(torch.float)
    y_stand = torch.from_numpy(y_stand).type(torch.float)
    """

    x = torch.from_numpy((pred_seq[:,0] - x_mean) / (1e-7 + x_std)).type(torch.float).view(-1,1)
    y = torch.from_numpy((pred_seq[:,1] - y_mean) / (1e-7 + y_std)).type(torch.float).view(-1,1)
    a = torch.from_numpy((pred_seq[:,2] - a_mean) / (1e-7 + a_std)).type(torch.float).view(-1,1)
    an_stand = torch.from_numpy((pred_seq[:,3] - an_mean) / (1e-7 + an_std)).type(torch.float).view(-1,1)
    am_stand = torch.from_numpy((pred_seq[:,4] - am_mean) / (1e-7 + am_std)).type(torch.float).view(-1,1)
    d_stand = torch.from_numpy((pred_seq[:,5] - d_mean) / (1e-7 + d_std)).type(torch.float).view(-1,1)

    y_dx = torch.from_numpy((real_seq[:,0] - dx_mean) / (1e-7 + dx_std)).type(torch.float).view(-1,1)
    y_dy =  torch.from_numpy((real_seq[:,1] - dy_mean) / (1e-7 + dy_std)).type(torch.float).view(-1,1)
    y_dtheta = torch.from_numpy((real_seq[:,2] - dtheta_mean) / (1e-7 + dtheta_std)).type(torch.float).view(-1,1)

    x_stand = torch.cat([x, y, a, an_stand, am_stand, d_stand], dim=1)
    y_stand = torch.cat([y_dx, y_dy, y_dtheta], dim=1)

    return x_stand, y_stand


def Normalization(x, x_min, x_max):
    x_nor = (x - x_min) / (x_max - x_min)
    x_nor = torch.from_numpy(x_nor).type(torch.float)
    return x_nor

def CarNormalization(pred_seq, real_seq, an_min=-79, an_max=79, am_min=0, am_max=60, d_min=0, d_max=8,
                     dx_min=-84.65728, dx_max=76.76244, dy_min=-74.287895, dy_max=99.92079, dtheta_min=-1.5900531, dtheta_max=1.8722337
):
    an_stand = Normalization(pred_seq[:,3], an_min, an_max).view(-1,1)
    am_stand = Normalization(pred_seq[:, 4], am_min, am_max).view(-1,1)
    d_stand = Normalization(pred_seq[:, 5], d_min, d_max).view(-1,1)

    y_dx = Normalization(real_seq[:,0], dx_min, dx_max).view(-1,1)
    y_dy = Normalization(real_seq[:,1], dy_min, dy_max).view(-1,1)
    y_dtheta = Normalization(real_seq[:,2], dtheta_min, dtheta_max).view(-1,1)

    x = torch.from_numpy(pred_seq[:,0]).type(torch.float).view(-1,1)
    y = torch.from_numpy(pred_seq[:,1]).type(torch.float).view(-1,1)
    a = torch.from_numpy(pred_seq[:,2]).type(torch.float).view(-1,1)

    x_stand = torch.cat([x, y, a, an_stand, am_stand, d_stand], dim=1)
    y_stand = torch.cat([y_dx, y_dy, y_dtheta], dim=1)

    return x_stand, y_stand

def seq_collate(data):
    (pred_seq, real_seq) = zip(*data)
    pred_seq = np.asarray([t.numpy() for t in pred_seq])
    real_seq = np.asarray([t.numpy() for t in real_seq])

    # Normalize the data
    # pred_seq_stand, real_seq_stand = CarNormalization(pred_seq, real_seq)
    pred_seq_stand, real_seq_stand = standardization(pred_seq, real_seq)

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