import sys
import os
import numpy as np


# p = os.path.abspath('..')
# sys.path.insert(1, p)
# sys.path.insert(1, "/home/roblab1/PycharmProjects/MultiAgentEnv/maenv")
# os.chdir("/home/roblab1/PycharmProjects/MultiAgentEnv/")
# os.chdir('../')
# print(os.getcwd())
from config import get_config, parse_args, update_conf_file, save_yaml
from utils import make_name_prefix
from maenv.mpe.cars_v0 import parallel_car_env
from agents import ddpg
from car_trainer import Trainer

'''
car train main
calls the config function to receive the passed arguments
creates agent object
passes the arguments, agent and the environment creator function to the trainer class 
run the trainer
'''

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    dict_params = update_conf_file(all_args)

    ''' store parameters in the YAML file '''
    ''' create  new directories if needed '''

    #### create env ####
    env_p = parallel_car_env(max_cycles=dict_params['env_params']['max_steps'],
                         seed=dict_params['env_params']['seed'],
                         local_ratio=dict_params['env_params']['local_ratio'])

    env_p.reset()
    dict_params['agent_params']['action_dim'] = env_p.action_spaces['agent_0'].shape[0]
    dict_params['agent_params']['obs_dim'] = env_p.observation_spaces['agent_0'].shape[0]
    dict_params['agent_params']['state_dim'] = env_p.state_space.shape[0]

    # create directories
    if all_args.use_same_dir:
        save_dir = all_args.load_params
    else:
        save_dir = os.path.join(all_args.results_dir,
                                'car',
                                dict_params['gen_params']['experiment_name']
                                + '_' + dict_params['agent_params']['algorithm']
                                + '_' + make_name_prefix())

    if not (os.path.exists(save_dir)):
        os.makedirs(save_dir)
        sub_dir = os.path.join(save_dir, 'model')
        os.makedirs(sub_dir)

    dict_params['gen_params']['save_dir'] = save_dir

    # save session configurations
    save_yaml(dict_params, os.path.join(save_dir,'config.yaml'))

    # initialize agent
    agent = ddpg.Agent(dict_params)

    # run session
    trainer = Trainer(dict_params, parallel_car_env)
    trainer.run(env_p, agent)

if __name__ == "__main__":
    main(sys.argv[1:])