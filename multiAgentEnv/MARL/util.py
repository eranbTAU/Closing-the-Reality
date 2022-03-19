import copy
import gym
import numpy as np
from gym.spaces import Box, Discrete, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable


def to_torch(input):
    return torch.from_numpy(input) if type(input) == np.ndarray else input


def to_numpy(x):
    return x.detach().cpu().numpy()


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)

def mse_loss(e):
    return e ** 2


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, avail_logits, temperature, device=torch.device('cpu')):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    if str(device) == 'cpu':
        y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    else:
        y = (logits.cpu() + sample_gumbel(logits.shape,
                                          tens_type=type(logits.data))).cuda()

    dim = len(logits.shape) - 1
    if avail_logits is not None:
        avail_logits = to_torch(avail_logits).to(device)
        y[avail_logits == 0] = -1e10
    return F.softmax(y / temperature, dim=dim)


def gaussian_noise(shape, std):
    return torch.empty(shape).normal_(mean=0, std=std)


def get_obs_shape(obs_space):
    if obs_space.__class__.__name__ == "Box":
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        obs_shape = obs_space
    else:
        raise NotImplementedError

    return obs_shape


def get_dim_from_space(space):
    if isinstance(space, Box):
        dim = space.shape[0]
    elif isinstance(space, list):
        dim = space[0].shape[0]
    else:
        raise Exception("Unrecognized space: ", type(space))
    return dim


def get_state_dim(observation_dict, action_dict):
    combined_obs_dim = sum([get_dim_from_space(space)
                            for space in observation_dict.values()])
    combined_act_dim = 0
    for space in action_dict.values():
        dim = get_dim_from_space(space)
        if isinstance(dim, np.ndarray):
            combined_act_dim += int(sum(dim))
        else:
            combined_act_dim += dim
    return combined_obs_dim, combined_act_dim, combined_obs_dim + combined_act_dim


def get_cent_act_dim(action_space):
    cent_act_dim = 0
    for space in action_space:
        dim = get_dim_from_space(space)
        if isinstance(dim, np.ndarray):
            cent_act_dim += int(sum(dim))
        else:
            cent_act_dim += dim
    return cent_act_dim
