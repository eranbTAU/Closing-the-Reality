''' ./mpe '''

''' write down what out-of-library parameters/objects are required in each file'''

from maenv.utils.conversions import parallel_wrapper_fn
from maenv.utils import parallel_to_aec
from maenv.mpe._mpe_utils.cars_env import ParallelCarEnv, make_env
from maenv.mpe.scenarios.cars import Scenario


class raw_env(ParallelCarEnv):
    def __init__(self, max_cycles=25, seed=None, local_ratio=None):
        scenario = Scenario()
        world = scenario.make_world()
        super().__init__(scenario, world, max_cycles, seed, local_ratio)
        self.metadata['name'] = "cars_v0"


parallel_car_env = make_env(raw_env)
'''
To support the AEC API, the raw_env() function just uses the from_parallel
function to convert from a ParallelEnv to an AEC env
'''
# simple_env = parallel_to_aec(raw_env)