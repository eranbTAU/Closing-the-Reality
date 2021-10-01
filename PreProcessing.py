import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import os
import time
import re

def get_filepaths(path):
    for root, directories, files in os.walk(path):
        only_files = []
        for filename in files:
            only_files.append(os.path.join(root, filename))
    return only_files


def heatmap2D(features):
    x1, x2 = features[:, 0], features[:, 1]
    heatmap, xedges, yedges = np.histogram2d(x1, x2, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()


def save_pickle(features, labels, path, name):
    features, labels = np.array(features).astype(np.float32), np.array(labels).astype(np.float32)
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

# marg_dx=25, marg_dy=50, marg_dt=1.4
def eraseforMargin(transitions_list, marg_tatb=120, marg_dx=22, marg_dy=45, marg_dt=1.3):
# def eraseforMargin(transitions_list, marg_tatb=120, marg_dx=25, marg_dy=25, marg_dt=0.7): #marg_dx=25, marg_dy=50, marg_dt=1.4
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

if __name__ == '__main__':
    path = r'E:\CarProject\NewCode_Project\datasets\newdata_car\train'
    save_path = r'E:\CarProject\NewCode_Project\data_split\split_data\real'
    file_path = get_filepaths(path)
    Dtrain = []

    for file in file_path:
        data = pickle.load(file=open(file, "rb"))
        # print(file)
        datakey =  [i for i in data['data'].keys()]
        for key in data['data'].keys():
            if len(data['data'][key]) == 0:
                os.remove(file)
                break
        dataexe = ExtractFeature(data)
        Dtrain += dataexe
    data_array = np.array(Dtrain)
    transition_columns = ['tA', 'tB', 'd_x', 'd_y', 'd_theta']
    Car_Data = pd.DataFrame(data_array, columns=transition_columns)
    Car_Data.replace(np.nan, 0.0, inplace = True)
    Car_Data = np.array(Car_Data)

    features = []
    labels = []
    for i in range(len(Car_Data)):
        features.append(Car_Data[i, :2])
        labels.append(Car_Data[i, 2:])

    save_pickle(features, labels, save_path, name='DataExtract')

