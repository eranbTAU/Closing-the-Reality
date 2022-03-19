import yaml
import numpy as np


def load_params(path):
    with open(path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            print(exc)

def get_rescaler(min, max):
    '''
    scaleup func. from [0,1] to original values
    :param min: 1D numpy array
    :param max: 1D numpy array
    :return: function
    '''
    return lambda x : x * (max - min) + min

def get_scaler(min, max):
    '''
    scaledown func. to [0,1]
    :param min: 1D numpy array
    :param max: 1D numpy array
    :return: function
    '''
    return lambda x : (x -  min) / (max - min)

def rotate_2D(xy, a):
    '''
    2D coordinate system rotation
    :param xy: ndarray. vector in xy coordinate
    :param a: float. angle between x and x'
    :return: ndarray. x'y'
    '''
    mat = np.array([[np.cos(a), np.sin(a)],
                   [-np.sin(a), np.cos(a)]])
    xy_ = np.matmul(mat, xy)
    return xy_

def clip_angle(a):
    ''' shift angle to range (-pi, pi]'''
    if a > np.pi:
        a -= 2 * np.pi
    elif a <= -np.pi:
        a += 2 * np.pi
    return a