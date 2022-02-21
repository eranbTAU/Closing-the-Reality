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
    pred_seq_stand, real_seq_stand = CarNormalization(pred_seq, real_seq)
    # pred_seq_stand, real_seq_stand = standardization(pred_seq, real_seq)

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

def eraseforMargin(transitions, marg=100):
    idx = np.where(np.abs(transitions[:,6:8])>marg)[0]
    filtered_transitions = np.delete(transitions, idx, axis=0)
    return filtered_transitions

def ExtractFeature(data, synthetic):
    np.seterr(divide='ignore', invalid='ignore')
    columns = ['angle', 'amplitude', 'delay', 'pos_x', 'pos_y', 'h_x', 'h_y']
    df = pd.DataFrame(data['data'], columns=columns)
    con_calib = data['params']['to_mm']
    df['out'] = False
    x_bounds = data['params']['boundX']
    y_bounds = data['params']['boundY']
    x_offset = -((x_bounds[1] - x_bounds[0]) / 2 + x_bounds[0])
    y_offset = (y_bounds[1] - y_bounds[0]) / 2 + y_bounds[0]
    x_box_size = x_bounds[1] - x_bounds[0]
    y_box_size = y_bounds[1] - y_bounds[0]

    # if outside the boundries set to True
    df.loc[(df['pos_x'] < x_bounds[0]) | (df['pos_x'] > x_bounds[1]), 'out'] = True
    df.loc[(df['pos_y'] < y_bounds[0]) | (df['pos_y'] > y_bounds[1]), 'out'] = True

    transitions_list = []

    for i in range(len(df) - 2):
        if df.loc[i, 'out'] or df.loc[i + 1, 'out'] or df.loc[i + 2, 'out']: # if outside the boundries for at least two following steps
            continue # skip
        else:
            # get pos
            x_t = (df.loc[i, 'pos_x'] + x_offset)/x_box_size # range [-0.5, 0.5]
            y_t = (-df.loc[i, 'pos_y'] + y_offset)/y_box_size # range [-0.5, 0.5]
            # scale pos to [0,1]:
            x_t = x_t + 0.5
            y_t = y_t + 0.5

            d_pos = [df.loc[i + 1, 'pos_x'] - df.loc[i, 'pos_x'],
                     df.loc[i + 1, 'pos_y'] - df.loc[i, 'pos_y']]
            d_pos_norm = np.linalg.norm(d_pos)
            heading_1 = [df.loc[i, 'h_x'], df.loc[i, 'h_y']] # normalized vector
            # calc absolute heading rotation
            h_t = np.arctan2(-heading_1[1], heading_1[0])/np.pi # return absolute normalized orientation
            # scale it to [0,1]:
            h_t = (h_t + 1)/2

            heading_2 = [df.loc[i + 1, 'h_x'], df.loc[i + 1, 'h_y']] # normalized vector
            theta = calc_angle(heading_1, heading_2)
            d_pos_normalzed = d_pos / d_pos_norm
            beta = calc_angle(heading_1, d_pos_normalzed)
            d_pos_relative = np.array([d_pos_norm * np.sin(beta), d_pos_norm * np.cos(beta)])

            transitions_list.append(
                [x_t, y_t, h_t, df.loc[i, 'angle'], df.loc[i, 'amplitude'], df.loc[i, 'delay'], d_pos_relative[0],
                 d_pos_relative[1], theta])

    dataset = np.array(transitions_list)
    if len(dataset) == 0:
        return 0
    dataset[:, 6:8] = dataset[:, 6:8] * con_calib

    dataset = eraseforMargin(dataset)
    return dataset.tolist()

class TrajectoryDataset(Dataset):
    def __init__(self, data_dir, synthetic):
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        Dtrain = []
        for path in all_files:
            data = pickle.load(file=open(path, "rb"))
            data = ExtractFeature(data, synthetic)
            if data == 0:
                continue
            Dtrain += data
        if not isinstance(data, list):
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

