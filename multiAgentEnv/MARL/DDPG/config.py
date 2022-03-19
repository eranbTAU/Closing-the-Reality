import argparse
import json
import os
import yaml


def load_yaml(path):
    with open(path, 'r') as file:
        try:
            dic = yaml.safe_load(file)
            return dic
        except yaml.YAMLError as exc:
            print(exc)

def save_yaml(doc, path):
    with open(path, 'w') as file:
        yaml.dump(doc, file)

'''
gets the parameters as arguments from the user and updates the config. file.
'''

def update_conf_file(args, save=True):
    params = vars(args)
    config_file = os.path.join('../maenv/mpe/_mpe_utils',
                               args.robot_type + '_config.yaml')

    if params['load_params'] is not None:
        config_file = os.path.join(params['load_params'], 'config.yaml')

    # upload params
    config = load_yaml(config_file)

    # overide these params if not to load from .yaml
    if params['load_params'] is None:
        gen_params = ['robot_type', 'experiment_name', 'report_interval']
        env_params = ['env_name', 'num_agents', 'num_landmarks', 'reward_func', 'local_ratio', 'render', 'max_steps',
                      'episodes', 'seed', 'reward_divider']
        agent_params = ['algorithm', 'load_policy', 'alpha', 'beta', 'gamma', 'tau', 'fc_dims', 'batch_size',
                        'buffer_size', 'noise_sigma', 'noise_theta', 'agent_device', 'reward_divider']
        fm_params = ['fd_model', 'robot_type', 'fdm_device', 'fdm_input_dim', 'fdm_output_dim']

        config['gen_params'] = {key: params.get(key) for key in gen_params}
        config['env_params'] = {key: params.get(key) for key in env_params}
        config['agent_params'] = {key: params.get(key) for key in agent_params}
        config['forward_model'] = {key: params.get(key) for key in fm_params}

    config['gen_params']['is_train'] = params.get('is_train')

    for item, doc in config.items():
        print(f'{item} : {doc}')

    if save:
        save_yaml(config, os.path.join('/home/roblab1/PycharmProjects/MultiAgentEnv/maenv/mpe/_mpe_utils',
                               args.robot_type + '_config.yaml'))
    return config


def parse_args(args, parser):
    all_args = parser.parse_known_args(args)[0]
    return all_args

def get_config():

    parser = argparse.ArgumentParser(
        description="Multi Agent RL Training Environment", formatter_class=argparse.RawDescriptionHelpFormatter)

    # general
    parser.add_argument('--load_params', type=str,
                        default=None)
    parser.add_argument('--results_dir', type=str,
                        default='../MARL/train_results')
    parser.add_argument('--use_same_dir', action='store_true')
    parser.add_argument('--robot_type', type=str, default="car", choices=["fish", "car"])
    parser.add_argument('--experiment_name', type=str, default='exp_name')
    parser.add_argument('--is_train', action='store_true')
    parser.add_argument('--report_interval', type=int, default=5)

    # forward dynamics model
    parser.add_argument('--fd_model', type=str, default='state_87_Mon_Jul_26_15_54_36.pt')
    parser.add_argument('--fdm_device', type=str, default='cpu')
    parser.add_argument('--fdm_input_dim', type=int, default=2)
    parser.add_argument('--fdm_output_dim', type=int, default=3)

    # training simulation environment parameters
    parser.add_argument('--env_name', type=str, default='env_name')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--local_ratio', type=float, default=1.)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--num_agents', type=int, default=2)
    parser.add_argument('--num_landmarks', type=int, default=1)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--reward_func', type=str, default='rew1')
    parser.add_argument('--seed', type=int, default=1000)

    # agent parameters
    parser.add_argument('--algorithm', type=str, default='ddpg')
    parser.add_argument('--load_ploicy', action='store_true')
    parser.add_argument('--agent_device', type=str, default='cpu')
    parser.add_argument('--alpha', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--beta', type=float, default=0.001)
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--fc_dims', type=int, default=128)
    parser.add_argument('--gamma', type=int, default=0.99)
    parser.add_argument('--noise_theta', type=float, default=3.)
    parser.add_argument('--noise_sigma', type=float, default=1.)
    parser.add_argument('--reward_divider', type=int, default=5)
    parser.add_argument('--tau', type=float, default=0.001)

    return parser
