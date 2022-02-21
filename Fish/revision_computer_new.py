import argparse
import gc
import logging
import os
import sys
import time
from collections import defaultdict
import re
import pickle

import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
import torch.optim as optim

from Fish.carnet.data.loader import data_loader
from Fish.carnet.losses import logcosh, weighted_mse_loss
from Fish.carnet.models import init_weights, FNet, FNetSkip, CNN, LSTM, GRU, ConFNet, ConFNet4
from Fish.carnet.utils import get_dset_path, get_total_norm, plot_all, save_states, predict_batch, plot_loss, calc_rmse

torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed(1)
parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--dataset_name', default='datasets', type=str)
parser.add_argument('--loader_num_workers', default=0, type=int)

parser.add_argument('--x_mean', default=0.5058, type=int)
parser.add_argument('--x_std', default=0.2366, type=int)
parser.add_argument('--y_mean', default=0.4997, type=int)
parser.add_argument('--y_std', default=0.2258, type=int)
parser.add_argument('--a_mean', default=0.4922, type=int)
parser.add_argument('--a_std', default=0.2924, type=int)
parser.add_argument('--an_mean', default=0.7162, type=int)
parser.add_argument('--an_std', default=37.6194, type=int)
parser.add_argument('--am_mean', default=28.0406, type=int)
parser.add_argument('--am_std', default=16.8835, type=int)
parser.add_argument('--d_mean', default=6.0158, type=int)
parser.add_argument('--d_std', default=1.5773, type=int)
parser.add_argument('--dx_mean', default=-2.5119e-01, type=int)
parser.add_argument('--dx_std', default=22.7929, type=int)
parser.add_argument('--dy_mean', default=3.1860e+01, type=int)
parser.add_argument('--dy_std', default=32.6926, type=int)
parser.add_argument('--dtheta_mean', default=-1.4515e-02, type=int)
parser.add_argument('--dtheta_std', default=0.5072, type=int)

# Normalization parameters
parser.add_argument('--an_min', default=-79, type=int)
parser.add_argument('--an_max', default=79, type=int)
parser.add_argument('--am_min', default=0, type=int)
parser.add_argument('--am_max', default=60, type=int)
parser.add_argument('--d_min', default=0, type=int)
parser.add_argument('--d_max', default=8, type=int)
parser.add_argument('--dx_min', default=-84.65728, type=int)
parser.add_argument('--dx_max', default=76.76244, type=int)
parser.add_argument('--dy_min', default=-74.287895, type=int)
parser.add_argument('--dy_max', default=99.92079, type=int)
parser.add_argument('--dtheta_min', default=-1.5900531, type=int)
parser.add_argument('--dtheta_max', default=1.8722337, type=int)


# Optimization
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iterations', default=10000, type=int)
parser.add_argument('--num_epochs', default=10, type=int)
parser.add_argument('--learning_rate', default=8.062253137577558e-05, type=float)
parser.add_argument('--beta1', default=0.7970569807817551, type=float)
parser.add_argument('--beta2', default=0.7592079885749976, type=float)
parser.add_argument('--weight_decay', default=0.007733941124966811, type=float)
parser.add_argument('--steps', default=1, type=int)

# Model Options
parser.add_argument('--input_dim', default=6, type=int)
parser.add_argument('--output_dim', default=3, type=int)
parser.add_argument('--clipping_threshold', default=0, type=float)
parser.add_argument('--best_loss', default=1.0, type=float)
parser.add_argument('--best_loss_val', default=100, type=float)

# Loss Options
parser.add_argument('--l1_loss_weight', default=0, type=float)
parser.add_argument('--best_k', default=1, type=int)

# Output
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--print_every', default=5, type=int)
parser.add_argument('--checkpoint_every', default=100, type=int)
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=1, type=int)
parser.add_argument('--num_samples_check', default=5000, type=int)

# Misc
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default="0", type=str)


def calc_mean_std(prediction, actual): # input: array[[dx,dy.d_theta],...] shape:(num_of_samples,3)
    RAD_TO_DEG = 57.2957795
    # calc mean error
    error = actual - prediction # [x, y, theta]
    pos_error = np.linalg.norm(error[:,:2], axis=1) # calc l2 norm on [x, y] error
    pos_mean_error = np.mean(pos_error)

    theta_error = np.abs(error[:,2])*RAD_TO_DEG # only [x, y] # only [theta degrees]
    theta_mean_error = np.mean(theta_error)

    # calc STD
    pos_error_std = np.std(pos_error)
    theta_error_std = np.std(theta_error)

    results = {'pos_mean_error': np.round(pos_mean_error, 3), 'pos_error_std': np.round(pos_error_std, 3), \
               'orientation_mean_error': np.round(theta_mean_error, 3), 'orientation_error_std': np.round(theta_error_std, 3)}

    # print(results)
    pos_mean_error = np.round(pos_mean_error, 3)
    theta_mean_error = np.round(theta_mean_error, 3)
    return results, pos_mean_error, theta_mean_error

def Normalz(x, x_min, x_max):
    x_nor = (x - x_min) / (x_max - x_min)
    x_nor = torch.from_numpy(x_nor).type(torch.float)
    return x_nor

def pred_batch(action, net):
    an_min = -79
    an_max = 79
    am_min = 0
    am_max = 60
    d_min = 0
    d_max = 8
    dx_min = -84.65728
    dx_max = 76.76244
    dy_min = -74.287895
    dy_max = 99.92079
    dtheta_min = -1.5900531
    dtheta_max = 1.8722337

    an_stand = Normalz(action[:,3], an_min, an_max).view(-1,1)
    am_stand = Normalz(action[:, 4], am_min, am_max).view(-1,1)
    d_stand = Normalz(action[:, 5], d_min, d_max).view(-1,1)
    x = torch.from_numpy(action[:,0]).type(torch.float).view(-1,1)
    y = torch.from_numpy(action[:,1]).type(torch.float).view(-1,1)
    a = torch.from_numpy(action[:,2]).type(torch.float).view(-1,1)
    action_nor = torch.cat([x, y, a, an_stand, am_stand, d_stand], dim=1)
    with torch.no_grad():
        # action_nor = action_nor.unsqueeze(1)
        prediction = net(action_nor.cuda())
    prediction = prediction.detach().cpu().numpy()
    prediction[:,0] = prediction[:,0] * ((dx_max) - (dx_min)) + dx_min
    prediction[:,1] = prediction[:,1] * ((dy_max) - (dy_min)) + dy_min
    prediction[:,2] = prediction[:,2] * ((dtheta_max) - (dtheta_min)) + dtheta_min
    return prediction

def pred_batch_stand(action, net):
    x_mean, x_std, y_mean, y_std, a_mean, a_std = args.x_mean, args.x_std, args.y_mean, args.y_std, args.a_mean, args.a_std
    an_mean, an_std, am_mean, am_std, d_mean, d_std = args.an_mean, args.an_std, args.am_mean, args.am_std, args.d_mean, args.d_std
    dx_mean, dx_std, dy_mean, dy_std, dtheta_mean, dtheta_std = args.dx_mean, args.dx_std, args.dy_mean, args.dy_std, args.dtheta_mean, args.dtheta_std

    x = torch.from_numpy((action[:,0] - x_mean) / (1e-7 + x_std)).type(torch.float).view(-1,1)
    y = torch.from_numpy((action[:,1] - y_mean) / (1e-7 + y_std)).type(torch.float).view(-1,1)
    a = torch.from_numpy((action[:,2] - a_mean) / (1e-7 + a_std)).type(torch.float).view(-1,1)
    an_stand = torch.from_numpy((action[:,3] - an_mean) / (1e-7 + an_std)).type(torch.float).view(-1,1)
    am_stand = torch.from_numpy((action[:,4] - am_mean) / (1e-7 + am_std)).type(torch.float).view(-1,1)
    d_stand = torch.from_numpy((action[:,5] - d_mean) / (1e-7 + d_std)).type(torch.float).view(-1,1)
    action_nor = torch.cat([x, y, a, an_stand, am_stand, d_stand], dim=1)

    with torch.no_grad():
        # action_nor = action_nor.unsqueeze(1)
        prediction = net(action_nor.cuda())
    prediction = prediction.detach().cpu().numpy()
    prediction[:,0] = prediction[:, 0] * ((1e-7) + (dx_std)) + dx_mean
    prediction[:, 1] = prediction[:, 1] * ((1e-7) + (dy_std)) + dy_mean
    prediction[:, 2] = prediction[:, 2] * ((1e-7) + (dtheta_std)) + dtheta_mean

    return prediction

def eval(fully_model):
    from car_evaluate_models import load_samples
    dir_white =
    x, y = load_samples(dir_white)

    predictions = pred_batch(x, fully_model)
    # predictions = pred_batch_stand(x, fully_model)
    results, pos_mean_error, theta_mean_error = calc_mean_std(predictions, y)
    return pos_mean_error, theta_mean_error

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)


def get_dtypes(args):
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        float_dtype = torch.cuda.FloatTensor
    return float_dtype

def denormalize_prediction(prediction, dx_min=-84.65728, dx_max=76.76244, dy_min=-74.287895,
                           dy_max=99.92079, dtheta_min=-1.5900531, dtheta_max=1.8722337):
    prediction[:,0] = prediction[:, 0] * ((dx_max) - (dx_min)) + dx_min
    prediction[:, 1] = prediction[:, 1] * ((dy_max) - (dy_min)) + dy_min
    prediction[:, 2] = prediction[:, 2] * ((dtheta_max) - (dtheta_min)) + dtheta_min
    return prediction

def destandardization_prediction(prediction, dx_mean=-2.2421e-01, dx_std=20.9514, dy_mean=4.1122e+01,
                           dy_std=21.5227, dtheta_mean=-1.9687e-02, dtheta_std=0.5069):
    prediction[:,0] = prediction[:, 0] * ((1e-7) + (dx_std)) + dx_mean
    prediction[:, 1] = prediction[:, 1] * ((1e-7) + (dy_std)) + dy_mean
    prediction[:, 2] = prediction[:, 2] * ((1e-7) + (dtheta_std)) + dtheta_mean
    return prediction


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    train_path = get_dset_path(args.dataset_name, 'train') #new/aug2
    val_path = get_dset_path(args.dataset_name, 'val')
    float_dtype = get_dtypes(args)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    logger.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path, synthetic=True)
    logger.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path, synthetic=True)

    iterations_per_epoch = len(train_dset) / args.batch_size
    if args.num_epochs:
        args.num_iterations = int(iterations_per_epoch * args.num_epochs)

    logger.info(
        'There are {} iterations per epoch'.format(iterations_per_epoch)
    )

    fully_model = ConFNet4(
        input_dim=args.input_dim,
        output_dim=args.output_dim
)

    fully_model.to(device)
    fully_model.apply(init_weights)
    fully_model.train()
    logger.info('Here is the Fully Connected:')
    logger.info(fully_model)

    l_loss = nn.L1Loss().type(float_dtype)

    optimizer = optim.Adam(fully_model.parameters(), lr=args.learning_rate,
                           betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  patience=2, factor=0.5,
                                  min_lr=1e-6, verbose=True)
    l = []
    # Maybe restore from checkpoint
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(args.output_dir,
                                    '%s_with_model.pt' % args.checkpoint_name)

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        fully_model.load_state_dict(checkpoint['state'])
        optimizer.load_state_dict(checkpoint['optim_state'])
        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']
        checkpoint['restore_ts'].append(t)
    else:
        # Starting from scratch, so initialize checkpoint data structure
        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'losses': defaultdict(list),
            'losses_epoch': defaultdict(list),
            'best_losses': defaultdict(list),
            'metrics_train': defaultdict(list),
            'norm': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'state': None,
            'allepoch_state': None,
            'optim_state': None,
            'best_state': None,
            'best_t': None,
        }
        checkpoint_val = {
            'metrics_val': defaultdict(list),
            'metrics_val2': defaultdict(list),
            'optim_state_val': None,
            'best_t': None,
            'batch_size': None,
            'learning_rate': None,
            'beta1': None,
            'beta2': None,
            'weight_decay': None,
            'num_epochs': None,
            'rmse': None,
            'counters': {
                'epoch_val': None,
            },
            'best_loss_val': defaultdict(list),
            'best_state_val': None,
        }
    t0 = None
    losses = []
    train_loss = []
    losses_chackP = {}
    checkpoint_state = {
        'state': defaultdict(list),
        'state_predict': defaultdict(list)
    }
    gt_list = []
    pred_list = []
    for idx in range(args.num_epochs):
        gc.collect()
        loss_total = 0.0
        epoch += 1
        logger.info('Starting epoch {}'.format(epoch))
        for batch in train_loader:
            if args.timing == 1:
                torch.cuda.synchronize()
                t1 = time.time()

            step_type = 'FC'
            losses, losses_chackP, loss_total, checkpoint_state = FC_step(args, idx, batch, fully_model,
                                                                          optimizer, l_loss, losses, losses_chackP,
                                                                          loss_total, checkpoint_state, pred_list,
                                                                          gt_list, device)
            checkpoint['norm'].append(
                get_total_norm(fully_model.parameters())
            )

            if args.timing == 1:
                torch.cuda.synchronize()
                t2 = time.time()
                logger.info('{} step took {}'.format(step_type, t2 - t1))

            if args.timing == 1:
                if t0 is not None:
                    logger.info('Interation {} took {}'.format(
                        t - 1, time.time() - t0
                    ))
                t0 = time.time()

            t += 1

            logger.info('t = {} / {}'.format(t + 1, args.num_iterations))
            logger.info('loss: {:.3f}'.format(losses_chackP['loss']))

        logger.info('t_epoch = {} / {}'.format(t + 1, args.num_iterations))
        logger.info('epoch_loss: {:.3f}'.format(losses_chackP['loss']))
        k = sum(losses) / len(losses)
        if k < args.best_loss:
            logger.info('{:.3f}'.format(k))
            checkpoint['best_losses'][idx].append(k)
            checkpoint['best_state'] = fully_model.state_dict()
            checkpoint['counters']['epoch'] = epoch
            checkpoint['optim_state'] = optimizer.state_dict()
            args.best_loss = k
            checkpoint['losses'][idx].append(k)


        logger.info('Checking stats on val ...')
        metrics_val = check_accuracy(val_loader, fully_model, l_loss, device)
        pos_mean_error, theta_mean_error = eval(fully_model)
        scheduler.step(pos_mean_error)
        train_loss.append(sum(losses) / len(losses))
        l.append(losses)
        losses = []

        for i, j in sorted(metrics_val.items()):
            logger.info('  [val] {}: {:.3f}'.format(i, pos_mean_error))
            checkpoint_val['metrics_val'][i].append(pos_mean_error)
            checkpoint_val['metrics_val2'][i].append(j)

        if pos_mean_error < args.best_loss_val:
            logger.info('New low for val_err')
            checkpoint_val['best_t'] = t
            checkpoint_val['best_loss_val'][idx].append(pos_mean_error)
            checkpoint_val['best_state_val'] = fully_model.state_dict()
            checkpoint_val['counters']['epoch_val'] = epoch
            checkpoint_val['optim_state_val'] = optimizer.state_dict()
            args.best_loss_val = pos_mean_error

    checkpoint_val['batch_size'] = args.batch_size
    checkpoint_val['learning_rate'] = args.learning_rate
    checkpoint_val['beta1'] = args.beta1
    checkpoint_val['beta2'] = args.beta2
    checkpoint_val['weight_decay'] = args.weight_decay
    checkpoint_val['num_epochs'] = args.num_epochs

    checkdir =
    checkpoint_val_dir =
    checkdirstate =

    timetime = time.asctime()
    timetime = timetime.replace(':', '_')
    model_name = 'model' + '_' + str(re.sub('[ ]', '_', timetime))[:19]

    state_name = 'state' + '_' + str(re.sub('[ ]', '_', timetime))[:19]
    checkpoint_path = os.path.join(checkdir, model_name + '.pt')
    checkpoint_path_val = os.path.join(checkpoint_val_dir, model_name + '.pt')
    checkpoint_statepath = os.path.join(checkdirstate, state_name + '.pkl')

    checkpoint['allepoch_state'] = fully_model.state_dict()
    logger.info('Best loss {}'.format(args.best_loss))
    logger.info('Best loss validation {}'.format(args.best_loss_val))
    logger.info('Saving checkpoint to {}'.format(checkpoint_path_val))
    logger.info('val loss {}'.format(checkpoint_val['metrics_val']['loss']))
    logger.info('val2 loss {}'.format(checkpoint_val['metrics_val2']['loss']))
    torch.save(checkpoint_val, checkpoint_path_val)
    torch.save(checkpoint, checkpoint_path)
    save_states(checkpoint_state, checkpoint_statepath)
    logger.info('Done.')


def FC_step(
        args, idx, batch, fully_model,
        optimizer, l_loss, losses, losses_chackP,
        loss_total, checkpoint_state, pred_list, gt_list, device):
    batch = [tensor.to(device) for tensor in batch]
    (sample_traj, gt_traj) = batch
    sample_traj = sample_traj.unsqueeze(1)
    pred_traj = fully_model(sample_traj)
    gt_list.append(gt_traj)
    pred_list.append(pred_traj)
    checkpoint_state['state'] = gt_list
    checkpoint_state['state_predict'] = pred_list

    error = pred_traj - gt_traj  # [x, y, theta]
    pos_error = torch.norm(error[:, :2], dim=1)  # calc l2 norm on [x, y] error
    pos_mean_error = torch.mean(pos_error)
    theta_error = torch.abs(error[:, 2])  # only [x, y] # only [theta degrees]
    theta_mean_error = torch.mean(theta_error)
    loss = pos_mean_error + theta_mean_error

    losses_chackP['loss'] = loss.item()
    loss_total += loss.item()
    losses_chackP['total_loss'] = loss_total
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    if args.clipping_threshold > 0:
        nn.utils.clip_grad_norm_(
            fully_model.parameters(), args.clipping_threshold_g
        )
    optimizer.step()

    return losses, losses_chackP, loss_total, checkpoint_state


def check_accuracy(loader, fully_model, l_loss, device):
    losses_val = []
    metrics = {}
    total_traj = 0
    fully_model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.to(device) for tensor in batch]
            (sample_traj, gt_traj) = batch
            sample_traj = sample_traj.unsqueeze(1)
            pred_traj = fully_model(sample_traj)

            error = pred_traj - gt_traj  # [x, y, theta]
            pos_error = torch.norm(error[:, :2], dim=1)  # calc l2 norm on [x, y] error
            pos_mean_error = torch.mean(pos_error)

            theta_error = torch.abs(error[:, 2])  # only [x, y] # only [theta degrees]
            theta_mean_error = torch.mean(theta_error)

            l_loss_rel = pos_mean_error + theta_mean_error
            

            losses_val.append(l_loss_rel.item())
            total_traj += pred_traj.size(1)

    metrics['loss'] = sum(losses_val) / len(losses_val)
    fully_model.train()
    return metrics

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)