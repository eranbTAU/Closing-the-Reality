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

############ first_model
# def CarNormalization(pred_seq, real_seq, x_min=-120, x_max=120, dx_min=-25, dx_max=25,
#                      dy_min=-50, dy_max=50, dtheta_min=-1.4, dtheta_max=1.4
# ):
#     x_stand = Normalization(pred_seq, x_min, x_max)
#     y_dx = Normalization(real_seq[:,0], dx_min, dx_max).view(-1,1)
#     y_dy = Normalization(real_seq[:,1], dy_min, dy_max).view(-1,1)
#     y_dtheta = Normalization(real_seq[:,2], dtheta_min, dtheta_max).view(-1,1)
#     y_stand = torch.cat([y_dx, y_dy, y_dtheta], dim=1)
#     return x_stand, y_stand

############ x_min / x_max model
# def CarNormalization(pred_seq, real_seq, x_min=-120, x_max=120, dx_min=-21.9717, dx_max=21.9717,
#                      dy_min=-44.9300, dy_max=44.9875, dtheta_min=-1.3000, dtheta_max=1.2928
# ):

############ small data model
# def CarNormalization(pred_seq, real_seq, x_min=-120, x_max=120, dx_min=-24.4098, dx_max=23.1483,
#                      dy_min=-24.9989, dy_max=24.9989, dtheta_min=-0.7000, dtheta_max=0.6997
# ):
def CarNormalization(pred_seq, real_seq, x_min=-133.2856, x_max=167.3345, dx_min=-27.8957, dx_max=25.4542,
                     dy_min=-56.4636, dy_max=44.9875, dtheta_min=-1.5085, dtheta_max=1.3781
):
    x_stand = Normalization(pred_seq, x_min, x_max)
    y_dx = Normalization(real_seq[:,0], dx_min, dx_max).view(-1,1)
    y_dy = Normalization(real_seq[:,1], dy_min, dy_max).view(-1,1)
    y_dtheta = Normalization(real_seq[:,2], dtheta_min, dtheta_max).view(-1,1)
    y_stand = torch.cat([y_dx, y_dy, y_dtheta], dim=1)
    return x_stand, y_stand

def seq_collate_gan(data):
    (pred_seq, real_seq) = zip(data)
    pred_seq = np.asarray([t.numpy() for t in pred_seq])
    real_seq = np.asarray([t.numpy() for t in real_seq])

    # Normalize the data
    pred_seq_stand, real_seq_stand = CarNormalization(pred_seq, real_seq)
    out = [
        pred_seq_stand, real_seq_stand]
    return tuple(out)

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

def eraseforMargin(transitions_list, marg_tatb=120, marg_dx=22, marg_dy=45, marg_dt=1.3):
# def eraseforMargin(transitions_list, marg_tatb=120, marg_dx=25, marg_dy=25, marg_dt=0.7): #marg_tatb=120, marg_dx=25, marg_dy=50, marg_dt=1.4
    #dx_margin
    s = (np.abs(transitions_list[:, 2]) > marg_dx).nonzero()[0]
    trans_l = pd.DataFrame(transitions_list)
    ta = np.array(trans_l.drop(s, axis=0))
    #dy_margin
    s2 = (np.abs(ta[:, 3]) > marg_dy).nonzero()[0]
    trans_l = pd.DataFrame(ta)
    ta = np.array(trans_l.drop(s2, axis=0))
    #dtheta_margin
    s3 = (np.abs(ta[:, 4]) > marg_dt).nonzero()[0]
    trans_l = pd.DataFrame(ta)
    ta = np.array(trans_l.drop(s3, axis=0))
    #ta_margin
    s31 = (np.abs(ta[:,:2]) > marg_tatb).nonzero()[0]
    trans_l = pd.DataFrame(ta)
    ta = np.array(trans_l.drop(s31, axis=0))
    return ta

def ExtractFeature(data):
    np.seterr(divide='ignore', invalid='ignore')
    columns = ['tA', 'tB', 'pos_x', 'pos_y', 'h_x', 'h_y', 'time']

    if type(data['data']) is dict:
        for l in data['data']:
            df = pd.DataFrame(data['data'][l], columns=columns)
    else:
        df = pd.DataFrame(data['data'], columns=columns)
    con_calib = data['params']['pix_to_mm']
    df['out'] = False
    df.loc[(df['pos_x'] < data['params']['boundX'][0]) | (df['pos_x'] > data['params']['boundX'][1]), 'out'] = True
    df.loc[(df['pos_y'] < data['params']['boundY'][0]) | (df['pos_y'] > data['params']['boundY'][1]), 'out'] = True
    transitions_list = []
    for i in range(len(df) - 1):
        dt = int((df.loc[i + 1, 'time'] - df.loc[i, 'time']) * 1000)
        if df.loc[i, 'out'] & df.loc[i + 1, 'out']:
            continue
        elif dt > (1000 * data['params']['rate_sec'] + 100):  # to make sure that state wasn't skipped
            continue
        else:
            d_pos = [df.loc[i + 1, 'pos_x'] - df.loc[i, 'pos_x'],
                     df.loc[i + 1, 'pos_y'] - df.loc[i, 'pos_y']]
            d_pos_norm = np.linalg.norm(d_pos)
            heading_1 = [df.loc[i, 'h_x'], df.loc[i, 'h_y']]
            heading_2 = [df.loc[i + 1, 'h_x'], df.loc[i + 1, 'h_y']]
            theta = calc_angle(heading_1, heading_2)
            d_pos_normalzed = d_pos / d_pos_norm
            beta = calc_angle(heading_1, d_pos_normalzed)
            d_pos_relative = np.array([d_pos_norm * np.sin(beta), d_pos_norm * np.cos(beta)])
            transitions_list.append([df.loc[i, 'tA'], df.loc[i, 'tB'], d_pos_relative[0], d_pos_relative[1], theta])
    transitions_list = np.array(transitions_list)
    transitions_list[:, 2:4] = transitions_list[:, 2:4] * con_calib
    ta = eraseforMargin(transitions_list)
    return ta.tolist()



def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    def __init__(self, data_dir):
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        Dtrain = []
        for path in all_files:
            data = pickle.load(file=open(path, "rb"))
            if len(data) < 5:
                for key in data['data'].keys():
                    if len(data['data'][key]) == 0:
                        os.remove(path)
                        break
                data = ExtractFeature(data)
                Dtrain += data
        if not isinstance(data, list):
            data = data.tolist()
            Dtrain += data
        transition_columns = ['tA', 'tB', 'd_x', 'd_y', 'd_theta']
        Car_Data = pd.DataFrame(Dtrain, columns=transition_columns)
        Car_Data = np.asarray(Car_Data.dropna())
        # Convert numpy -> Torch Tensor
        self.action_x = torch.from_numpy(
            Car_Data[:, 0:2]).type(torch.float)
        self.state_y = torch.from_numpy(
            Car_Data[:, 2:]).type(torch.float)
        self.num_samples = len(Car_Data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        out = [
            self.action_x[index], self.state_y[index]
        ]
        return out

