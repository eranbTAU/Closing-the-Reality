import numpy as np
import os
import torch
import subprocess
import time
import pickle
import re
import matplotlib.pyplot as plt

def save_pickle(features, labels, path, name):
    features, labels = np.array(features).astype(np.float32), np.array(labels).astype(np.float32)
    x = time.asctime()
    filename = name + '_' + str(re.sub('[ ]', '_', x) + '.pkl')
    dir = os.path.join(path, filename)
    with open(dir, "wb") as f:
        pickle.dump([features, labels], f)

def heatmap2D(features):
    x1, x2 = features[:, 0], features[:, 1]
    heatmap, xedges, yedges = np.histogram2d(x1, x2, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()

def int_tuple(s):
    return tuple(int(i) for i in s.split(','))


def find_nan(variable, var_name):
    variable_n = variable.data.cpu().numpy()
    if np.isnan(variable_n).any():
        exit('%s has nan' % var_name)

def get_gpu_memory():
    torch.cuda.synchronize()
    opts = [
        'nvidia-smi', '-q', '--gpu=' + str(1), '|', 'grep', '"Used GPU Memory"'
    ]
    cmd = str.join(' ', opts)
    ps = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ps.communicate()[0].decode('utf-8')
    output = output.split("\n")[0].split(":")
    consumed_mem = int(output[1].strip().split(" ")[0])
    return consumed_mem

def get_total_norm(parameters, norm_type=2):
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            try:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm**norm_type
                total_norm = total_norm**(1. / norm_type)
            except:
                continue
    return total_norm

def get_dset_path(dset_name, dset_type):
    _dir = os.path.dirname(__file__)
    _dir = _dir.split("\\")[:-1]
    _dir = "/".join(_dir)
    return os.path.join(_dir, dset_name, dset_type)


def get_gandset_path(dset_name, dset_type):
    _dir = os.path.dirname(__file__)[:-7]
    return os.path.join(_dir, dset_name, dset_type)

def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)


def Save_Network(PATH, epoch, net, optimizer):

    state = {
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),

    }
    torch.save(state, PATH)

def Load_Network(PATH, net, optimizer):
    state = torch.load(PATH)
    net.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])


def plot_all(prediction, actual, title, model_name, idx):
    pred1, pred2, pred3 = prediction[:, 0], prediction[:, 1], prediction[:, 2]
    actual1, actual2, actual3 = actual[:, 0], actual[:, 1], actual[:, 2]
    x = np.arange(1, len(pred1)+1)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10,10))
    fig.suptitle(title)
    ax1.plot(x, pred1  , label='predicted')
    ax1.plot(x, actual1, label='actual' )
    ax1.set_ylabel('dx [mm]')
    ax2.plot(x, pred2, label='predicted')
    ax2.plot(x, actual2, label='actual' )
    ax2.set_ylabel('dy [mm]')
    ax3.plot(x, pred3, label='predicted')
    ax3.plot(x, actual3, label='actual' )
    ax3.set_ylabel('d_theta [radians]')
    ax3.set_xlabel('time step')
    ax1.legend()
    # plt.show()
    dir = r'E:\CarProject\NewCode_Project\plot'
    checkpoint_path = os.path.join(dir, model_name + '_' + str(idx) + '.png')
    plt.savefig(checkpoint_path, bbox_inches='tight')
    plt.close()

def plot_live(prediction, actual, title): # input: array[[dx,dy.d_theta],...] shape:(num_of_samples,3)
    plt.gcf().clear()
    pred1, pred2, pred3 = prediction[:, 0], prediction[:, 1], prediction[:, 2]
    actual1, actual2, actual3 = actual[:, 0], actual[:, 1], actual[:, 2]
    x = np.arange(1, len(pred1)+1)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10,10))

    fig.suptitle(title)

    ax1.plot(x, pred1, label='predicted')
    ax1.plot(x, actual1, label='actual')
    ax1.set_ylabel('dx [mm]')

    ax2.plot(x, pred2)
    ax2.plot(x, actual2)
    ax2.set_ylabel('dy [mm]')

    ax3.plot(x, pred3)
    ax3.plot(x, actual3)
    ax3.set_ylabel('d_theta [radians]')

    ax3.set_xlabel('time step')

    ax1.legend()

    plt.draw()
    plt.pause(0.00000000001)

def save_states(checkpoint_state, path):
    s = checkpoint_state['state']
    sp = checkpoint_state['state_predict']
    with open(path, "wb") as f:
        pickle.dump([s, sp], f)

def loss_graph(train_loss, val_loss):
    plt.plot(range(len(train_loss)), train_loss, label="train_loss")
    plt.plot(range(len(val_loss)), val_loss, label="train_loss")
    plt.legend()
    plt.savefig('lossVSvalidation.png')
    plt.show()

def predict_batch(x, model, model_name):
    x_min = -120
    x_max = 120

    ############ first_model
    # dx_min = -25
    # dx_max = 25
    # dy_min = -50
    # dy_max = 50
    # dtheta_min = -1.4
    # dtheta_max = 1.4

    ############ x_min / x_max model
    dx_min = -21.9717
    dx_max = 21.9717
    dy_min = -44.9300
    dy_max = 44.9875
    dtheta_min = -1.3000
    dtheta_max = 1.2928

    ############ small data model
    # dx_min = -24.4098
    # dx_max = 23.1483
    # dy_min = -24.9989
    # dy_max = 24.9989
    # dtheta_min = -0.7000
    # dtheta_max = 0.6997

    action_nor = (x - x_min) / (x_max - x_min)
    action_nor = torch.tensor(action_nor, dtype=torch.float)
    with torch.no_grad():
        if model_name == 'FC':
            action_nor = action_nor.cuda()
        else:
            action_nor = action_nor.cuda().unsqueeze(1)
        prediction = model(action_nor)
    prediction = prediction.detach().cpu().numpy()
    prediction[:,0] = prediction[:,0] * ((dx_max) - (dx_min)) + dx_min
    prediction[:,1] = prediction[:,1] * ((dy_max) - (dy_min)) + dy_min
    prediction[:,2] = prediction[:,2] * ((dtheta_max) - (dtheta_min)) + dtheta_min
    return prediction

def plot_loss(train_loss, val_loss, loss_plot_name):
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(loss_plot_name, bbox_inches='tight')
    plt.close()



def calc_rmse(prediction, actual): # input: array[[dx,dy,d_theta],...] shape:(num_of_samples,3)
    error = actual - prediction
    squared = np.power(error, 2)
    mean = np.mean(squared, axis=0)
    rmse = np.sqrt(mean)
    results = {'dx': round(rmse[0], 3), 'dy': round(rmse[1], 3), 'dtheta': round(rmse[2], 3)}
    print('error RMSE: ', results)
    return results


def plot_trajectory(actual, prediction, title):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title(title, fontsize=20)
    ax.set_ylim((-200, 150))
    ax.set_xlim((-250, 60))
    ax.set_ylabel('Y [mm]', size=15)
    ax.set_xlabel('X [mm]', size=15)
    actual = np.array(actual)
    actual = actual[:, :2] # slice x,y
    prediction = np.array(prediction)
    prediction = prediction[:, :2] # slice x,y
    ax.plot(actual[0, 0], actual[0, 1], marker='x', color='blue', markersize=12)
    ax.plot(prediction[0, 0], prediction[0, 1], marker='x', color='orange', markersize=12)
    ax.plot(actual[:,0],actual[:,1], label='True')
    ax.plot(prediction[:, 0], prediction[:, 1], label='Predicted')
    ax.plot(actual[-1, 0], actual[-1, 1], marker='o', color='blue', markersize=12)
    ax.plot(prediction[-1, 0], prediction[-1, 1], marker='o', color='orange', markersize=12)
    # ax.legend(prop={'size': 15})
    ax.legend(loc='best', bbox_to_anchor=(0.107, 0.115, 0.9, 0.9), prop={'size': 15})
    dir = r'E:\CarProject\NewCode_Project\Simulator\plot'
    checkpoint_path = os.path.join(dir, title + '.png')
    plt.savefig(checkpoint_path, bbox_inches='tight')
    plt.close()