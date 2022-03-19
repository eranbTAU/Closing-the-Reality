import sys
import os
import numpy as np

from config import get_config, parse_args, update_conf_file
from maenv.mpe.cars_v0 import parallel_car_env

'''
main script
calls the config function to receive the passed arguments
'''

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    dict_params = update_conf_file(all_args, save=False)

    ''' store parameters in the YAML file '''
    ''' create  new directories if needed '''

    #### API tets ####
    from maenv.test import parallel_api_test
    env_p = parallel_car_env(max_cycles=dict_params['env_params']['max_steps'],
                         seed=dict_params['env_params']['seed'],
                         local_ratio=dict_params['env_params']['local_ratio'])
    parallel_api_test(env_p, num_cycles=1000)

    #### seed test ####
    from maenv.test import parallel_seed_test
    parallel_seed_test(parallel_car_env, num_cycles=10, test_kept_state=True)

    #### max cycles test ####
    from maenv.test import max_cycles_test
    max_cycles_test(parallel_car_env)

    #### average reward ####
    from maenv.utils import average_total_reward
    env_p = parallel_car_env(max_cycles=150,
                             seed=dict_params['env_params']['seed'],
                             local_ratio=1.)
    average_total_reward(env_p, max_episodes=100, max_steps=1000)


if __name__ == "__main__":
    main(sys.argv[1:])