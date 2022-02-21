import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import os
import time
import re
import torch

def get_filepaths(path):
    for root, directories, files in os.walk(path):
        only_files = []
        for filename in files:
            only_files.append(os.path.join(root, filename))
    return only_files

def Normalization(x, x_min, x_max):
    x_nor = (x - x_min) / (x_max - x_min)
    x_nor = torch.from_numpy(x_nor).type(torch.float)
    return x_nor

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

def save_pickle(features, labels, path, name):
    x = time.asctime()
    timetime = x.replace(':', '_')
    filename = name + '_' + str(re.sub('[ ]', '_', timetime))[:19] + '.pkl'
    dir = os.path.join(path, filename)
    with open(dir, "wb") as f:
        pickle.dump([features, labels], f)

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

def ExtractFeature(data):
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

if __name__ == '__main__':
    path = r'E:\CarProject\Fish\datasets\val2'
    save_path = r'E:\CarProject\Fish\datasets'
    file_path = get_filepaths(path)
    Dtrain = []
    for file in file_path:
        data = pickle.load(file=open(file, "rb"))
        dataexe = ExtractFeature(data)
        if dataexe == 0:
            continue
        Dtrain += dataexe
    transition_columns = ['x', 'y', 'a','an', 'am', 'd', 'd_x', 'd_y', 'd_theta']
    Car_Data = pd.DataFrame(Dtrain, columns=transition_columns)
    Car_Data = np.asarray(Car_Data.dropna())

    features = Car_Data[:, 0:6].astype(np.float32)
    labels = Car_Data[:, 6:].astype(np.float32)

    save_pickle(features, labels, save_path, name='DataExtract')



"""


"""