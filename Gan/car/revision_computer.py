import argparse
import gc
import logging
import os
import sys
import time
from collections import defaultdict
import re
import pickle

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
import torch.optim as optim

from carnet.data.loader import data_loader
from carnet.losses import L1Loss, logcosh
from carnet.models import init_weights, ConFNet, FNet, FNetSkip, CNN, LSTM, GRU, ConFNet4
from carnet.utils import get_dset_path, get_total_norm, plot_all, save_states, predict_batch, plot_loss

torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed(1)

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--loader_num_workers', default=0, type=int)

# Tuning
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iterations', default=10000, type=int)
parser.add_argument('--num_epochs', default=6, type=int)
parser.add_argument('--learning_rate', default=0.005138363946359416, type=float)
parser.add_argument('--beta1', default=0.9182715972362874, type=float)
parser.add_argument('--beta2', default=0.9565132211253531, type=float)
parser.add_argument('--weight_decay', default=0.0006273913713486183, type=float)
parser.add_argument('--steps', default=1, type=int)

# Model Options
parser.add_argument('--input_dim', default=2, type=int)
parser.add_argument('--output_dim', default=3, type=int)
parser.add_argument('--clipping_threshold', default=0, type=float)
parser.add_argument('--best_loss', default=99.0, type=float)
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

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)

def get_dtypes(args):
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        float_dtype = torch.cuda.FloatTensor
    return float_dtype

def denormalize_prediction(prediction, dx_min=-24.9079, dx_max=24.6524, dy_min=-49.7925,
                           dy_max=49.9884, dtheta_min=-1.3976, dtheta_max=1.3989):
    prediction[:,0] = prediction[:, 0] * ((dx_max) - (dx_min)) + dx_min
    prediction[:, 1] = prediction[:, 1] * ((dy_max) - (dy_min)) + dy_min
    prediction[:, 2] = prediction[:, 2] * ((dtheta_max) - (dtheta_min)) + dtheta_min
    return prediction

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    train_path = get_dset_path('train')
    val_path = get_dset_path('test')
    float_dtype = get_dtypes(args)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    logger.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)
    logger.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)

    iterations_per_epoch = len(train_dset) / args.batch_size
    if args.num_epochs:
        args.num_iterations = int(iterations_per_epoch * args.num_epochs)

    logger.info(
        'There are {} iterations per epoch'.format(iterations_per_epoch)
    )

    fully_model = FNet(
        input_dim=args.input_dim,
        output_dim=args.output_dim
)
    fully_model.to(device)
    fully_model.apply(init_weights).train()
    logger.info('Here is the Fully Connected:')
    logger.info(fully_model)

    l_loss = L1Loss().type(float_dtype)
    optimizer = optim.Adam(fully_model.parameters(), lr=args.learning_rate,
                           betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  patience=2, factor=0.5,
                                  min_lr=1e-6, verbose=True)

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
            'optim_state_val': None,
            'best_t': None,
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
        k = losses_chackP['loss']
        if k < args.best_loss:
            logger.info('{:.3f}'.format(k))
            checkpoint['best_losses'][idx].append(k)
            checkpoint['best_state'] = fully_model.state_dict()
            checkpoint['counters']['epoch'] = epoch
            checkpoint['optim_state'] = optimizer.state_dict()
            args.best_loss = k
            checkpoint['losses'][idx].append(k)
            cstate_path = os.path.join(args.output_dir, 'checkpoint_state/checkpoint_state.pkl')
            with open(cstate_path, "wb") as f:
                pickle.dump(checkpoint_state, f)

        logger.info('Checking stats on val ...')
        metrics_val = check_accuracy(val_loader, fully_model, l_loss, device)
        scheduler.step(metrics_val['loss'])
        train_loss.append(sum(losses) / len(losses))
        losses = []
        for i, j in sorted(metrics_val.items()):
            logger.info('  [val] {}: {:.3f}'.format(i, j))
            checkpoint_val['metrics_val'][i].append(j)

        if j < args.best_loss_val:
            logger.info('New low for val_err')
            checkpoint_val['best_t'] = t
            checkpoint_val['best_loss_val'][idx].append(j)
            checkpoint_val['best_state_val'] = fully_model.state_dict()
            checkpoint_val['counters']['epoch_val'] = epoch
            checkpoint_val['optim_state_val'] = optimizer.state_dict()
            args.best_loss_val = k

            cstate_path = os.path.join(args.output_dir, 'checkpoint_state/checkpoint_state.pkl')
            with open(cstate_path, "wb") as f:
                pickle.dump(checkpoint_state, f)

        # if checkpoint_val['best_state_val'] == None:
        #     evaluate(fully_model, checkpoint['best_state'], idx)
        # else:
        #     evaluate(fully_model, checkpoint_val['best_state_val'], idx)
    checkdir = r''
    checkpoint_val_dir = r''
    checkdirstate = r''
    train_vs_loss_path = r''
    timetime = time.asctime()
    timetime = timetime.replace(':', '_')
    model_name = 'model' + '_' + str(re.sub('[ ]', '_', timetime))[:19]
    state_name = 'state' + '_' + str(re.sub('[ ]', '_', timetime))[:19]
    plot_name = 'train_Vs_val' + '_' + str(re.sub('[ ]', '_', timetime))[:19]
    plot_name = os.path.join(train_vs_loss_path, plot_name + '.png')
    plot_loss(train_loss, checkpoint_val['metrics_val']['loss'], loss_plot_name=plot_name)
    checkpoint_path = os.path.join(checkdir, model_name + '.pt')
    checkpoint_path_val = os.path.join(checkpoint_val_dir, model_name + '.pt')
    checkpoint_statepath = os.path.join(checkdirstate, state_name + '.pkl')
    checkpoint['allepoch_state'] = fully_model.state_dict()
    logger.info('Best loss {}'.format(args.best_loss))
    logger.info('Best loss val {}'.format(args.best_loss_val))
    logger.info('Saving checkpoint to {}'.format(checkpoint_path))
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
    pred_traj_denor = denormalize_prediction(pred_traj.clone())
    gt_traj_denor = denormalize_prediction(gt_traj.clone())

    error = pred_traj_denor - gt_traj_denor  # [x, y, theta]
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
            # sample_traj = sample_traj.unsqueeze(1)
            pred_traj = fully_model(sample_traj)
            pred_traj_denor = denormalize_prediction(pred_traj)
            gt_traj_denor = denormalize_prediction(gt_traj)

            error = pred_traj_denor - gt_traj_denor  # [x, y, theta]
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


def evaluate(fully_model, state, idx):
    label_dir =
    x = time.asctime()
    x = x.replace(':','_')
    timetime = '_' + str(re.sub('[ ]', '_', x))[:19]
    model_name = 'plot_model' + timetime
    state_name = 'state' + '_' + str(idx) + timetime
    data = pickle.load(file=open(label_dir, "rb"))
    x, y = data
    fully_test = fully_model
    fully_test.load_state_dict(state)
    dir_state =
    full_dir_state =  os.path.join(dir_state, state_name)
    torch.save(state, full_dir_state)
    all_predictions = predict_batch(x[:100], fully_test)
    plot_all(all_predictions, y[:100], title='GT vs Predict', model_name=model_name, idx=idx)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)