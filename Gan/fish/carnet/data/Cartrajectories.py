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

# def standardization(pred_seq, real_seq, x_mean=0.4858, x_std=0.3927, y_mean=9.1262, y_std=20.8873, a_mean=-1.9554, a_std=13.7231,
#                     an_mean=-7.7897, an_std=29.3421, am_mean=24.0856, am_std=11.9791, d_mean=1.8730, d_std=17.5728,
#                     dx_mean=-1.1641, dx_std=24.2767, dy_mean=32.6478, dy_std=28.9605, dtheta_mean=0.2759, dtheta_std=0.5773):

def standardization(pred_seq, real_seq, x_mean=0.4909, x_std=0.2364, y_mean=0.5127, y_std=0.2301, a_mean=0.4880,
                    a_std=0.2779,
                    an_mean=1.9258, an_std=36.4344, am_mean=27.6549, am_std=16.4413, d_mean=6.1299, d_std=1.4265,
                    dx_mean=6.8760e-01, dx_std=27.1033, dy_mean=3.0670e+01, dy_std=33.4764, dtheta_mean=9.6918e-04,
                    dtheta_std=0.5003):
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

############ x_min / x_max model
def CarNormalization(pred_seq, real_seq, an_min=-79, an_max=79, am_min=0, am_max=60, d_min=0, d_max=8,
                     dx_min=-78.7, dx_max=70.39, dy_min=-74.28, dy_max=99.92, dtheta_min=-1.59, dtheta_max=1.87
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
    def __init__(self, data_dir, synthetic):
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        Dtrain = []
        for path in all_files:
            data = pickle.load(file=open(path, "rb"))
            if type(data) is pd.DataFrame:
                data = np.array(data)
            if type(data) is list:
                data = np.concatenate([data[0], data[1]], axis=1)
            data = data.tolist()
            Dtrain += data
        transition_columns = ['x', 'y', 'a','an', 'am', 'd', 'd_x', 'd_y', 'd_theta']
        Car_Data = pd.DataFrame(Dtrain, columns=transition_columns)
        Car_Data = np.asarray(Car_Data.dropna())
        # Convert numpy -> Torch Tensor
        self.action_x = torch.from_numpy(
            Car_Data[:, 0:6]).type(torch.float)
        self.state_y = torch.from_numpy(
            Car_Data[:, 6:]).type(torch.float)
        self.num_samples = len(Car_Data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        out = [
            self.action_x[index], self.state_y[index]
        ]
        return out

