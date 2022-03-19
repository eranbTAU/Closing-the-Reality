''' ./mpe_utils '''

from gym.spaces import Box
import functools
import numpy as np
from gym.utils import seeding
from maenv import ParallelEnv
from maenv.utils import wrappers
from maenv.utils.agent_selector import agent_selector
# from maenv.utils import from_parallel
from .._mpe_utils.utils import load_params, rotate_2D, clip_angle



# def env():
#     '''
#     The env function wraps the environment in 3 wrappers by default. These
#     wrappers contain logic that is common to many pettingzoo environments.
#     We recommend you use at least the OrderEnforcingWrapper on your own environment
#     to provide sane error messages. You can find full documentation for these methods
#     elsewhere in the developer documentation.
#     '''
#     env = raw_env()
#     env = wrappers.CaptureStdoutWrapper(env)
#     env = wrappers.ClipOutOfBoundsWrapper(env)
#     env = wrappers.OrderEnforcingWrapper(env)
#     return env

def make_env(raw_env):
    def env(**kwargs):
        env = raw_env(**kwargs)
        # env = wrappers.CaptureStdoutWrapper(env)
        # env = wrappers.ClipOutOfBoundsWrapper(env)
        # env = wrappers.OrderEnforcingWrapper(env)
        return env
    return env # return env creator function


class ParallelCarEnv(ParallelEnv):
    '''
    contains the main components of the environment: world and scenario after they initialized
    Sets the space dims.
    defines reset, step and render functions.
    '''
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self, scenario, world, max_cycles=25, seed=None, local_ratio=None):
        super().__init__()
        '''
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_space
        - observation_space

        These attributes should not be changed after initialization.
        '''
        # calc some constants
        params = load_params('/home/roblab1/PycharmProjects/MultiAgentEnv/maenv/mpe/_mpe_utils/car_config.yaml')
        x_bounds, y_bounds = params['scalers']['x_bounds'], params['scalers']['y_bounds']
        to_mm = params['scalers']['to_mm']
        x_size, y_size = (x_bounds[1] - x_bounds[0]) * to_mm, (y_bounds[1] - y_bounds[0]) * to_mm
        self.is_out_of_box = _out_of_box(x_size, y_size)

        self.seed(seed)
        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world
        self.local_ratio = local_ratio

        self.scenario.reset_world(self.world, self.np_random)
        self.possible_agents = [agent.name for agent in self.world.agents]

        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()
        for agent in self.world.agents:
            self.action_spaces[agent.name] = Box(low=0, high=1, shape=(2,)) # 2D action input for cars
            obs_dim = len(self.scenario.observation(agent, self.world))
            self.observation_spaces[agent.name] = Box(low=-np.float32(np.inf), high=+np.float32(np.inf),
                                                             shape=(obs_dim,), dtype=np.float32)
        # global state will be defined by all entities absolute pose
        state_dim = len(self.world.entities) * 3
        self.state_space = Box(low=-np.float32(np.inf), high=+np.float32(np.inf), shape=(state_dim,),
                                      dtype=np.float32)

        # self.agent_selection = agent_selector(self.world.agents)
        # self.possible_agents = [a.name for a in self.world.agents]

        self.steps = 0

        self.viewer = None

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def reset(self):
        '''
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Returns the observations for each agent
        '''
        self.scenario.reset_world(self.world, self.np_random)

        self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.rewards = {name: 0. for name in self.agents}
        self._cumulative_rewards = {name: 0. for name in self.agents}
        self.dones = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self._reset_render()
        self.steps = 0
        self.current_actions = [None] * self.num_agents

        observations = {a.name: self.scenario.observation(a, self.world) for a in self.world.agents}

        return observations

    # def observe(self, agent):
    #     a = self.world.agents[0]
    #     return self.scenario.observation(a, self.world)

    def step(self, actions):
        '''
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        '''
        for name in self.agents:
            agent = self.world.agents[self.agent_name_mapping[name]]
            self._set_action(actions[name], agent, self.action_space(name))

        self.steps += 1

        self.world.step()

        observations = {a.name: self.scenario.observation(a, self.world) for a in self.world.agents}

        if self.steps >= self.max_cycles:
            for a in self.agents:
                self.dones[a] = True
        for i, a in enumerate(self.world.agents):  # check if any of agents is out of box bounds
            if self._colide(observations[a.name]):
                #     self.dones[agent] = True
                self.infos[a.name] = 'colide'
            if self.is_out_of_box(a.state.p_pos):
                self.infos[a.name] = 'out'
                print('out of bounds!')

        global_reward = 0.

        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(agent, self.world))
            self.rewards[agent.name] = agent_reward
            global_reward += agent_reward

        global_reward /= len(self.possible_agents)

        for agent in self.world.agents:
            if self.local_ratio is not None:
                self.rewards[agent.name] = global_reward * (1. - self.local_ratio) + self.rewards[agent.name] * self.local_ratio

        self.agents = [agent for agent in self.agents if not self.dones[agent]]

        return observations, self.rewards, self.dones, self.infos

    def state(self):
        return self.scenario.state(self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        '''
        set action for each agent. any modifications can be made here:
        add noise, time-dependent actions etc...
        '''
        agent.action.u = action

        # agent.action.c = np.zeros(self.world.dim_c)
        # if agent.movable:
        #     # physical action
        #     agent.action.u = np.zeros(self.world.dim_p)
        #     if self.continuous_actions:
        #         # Process continuous action as in OpenAI MPE
        #         agent.action.u[0] += action[0][1] - action[0][2]
        #         agent.action.u[1] += action[0][3] - action[0][4]
        #     else:
        #         # process discrete action
        #         if action[0] == 1:
        #             agent.action.u[0] = -1.0
        #         if action[0] == 2:
        #             agent.action.u[0] = +1.0
        #         if action[0] == 3:
        #             agent.action.u[1] = -1.0
        #         if action[0] == 4:
        #             agent.action.u[1] = +1.0
        #     sensitivity = 5.0
        #     if agent.accel is not None:
        #         sensitivity = agent.accel
        #     agent.action.u *= sensitivity
        #     action = action[1:]
        # if not agent.silent:
        #     # communication action
        #     if self.continuous_actions:
        #         agent.action.c = action[0]
        #     else:
        #         agent.action.c = np.zeros(self.world.dim_c)
        #         agent.action.c[action[0]] = 1.0
        #     action = action[1:]
        # # make sure we used all elements of action
        # assert len(action) == 0

    def _colide(self, obs):
        '''check whether agent is colided'''
        dx, dy = obs[::3], obs[1::3]
        dist = np.linalg.norm(np.column_stack((dx,dy)),  axis=1)
        colide = dist < 100.
        return colide.any()

    def render(self, mode="human"):
        '''
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        '''
        if self.viewer is None:
            from . import rendering
            self.viewer = rendering.Viewer(800, 800)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            # from multiagent._mpe_utils import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color[:3], alpha=0.5)
                    line = rendering.make_polyline([np.array([0., 0.]), np.array([70., 0.])])
                    line.set_color(0, 0, 0, alpha=0.8)
                    line.set_linewidth(2)
                else:
                    line = rendering.make_polyline([np.array([0., 0.]), np.array([65., 0.])])
                    geom.set_color(*entity.color[:3])
                geom.add_attr(xform)
                line.add_attr(xform)

                self.render_geoms.append(geom)
                self.render_geoms.append(line)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            self.viewer.geoms = []
            for geom in self.render_geoms:
                self.viewer.add_geom(geom)

            self.viewer.text_lines = []
            tline = rendering.TextLine(self.viewer.window, 0)
            self.viewer.text_lines.append(tline)
        self.viewer.text_lines[0].set_text(f'timestep: {self.steps}')


        # update bounds to center around agent
        all_poses = [entity.state.p_pos for entity in self.world.entities]
        cam_range = np.max(np.abs(np.array(all_poses))) + 100
        self.viewer.set_max_size(cam_range)
        # update geometry positions
        for e, entity in enumerate(self.world.entities):
            self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            self.render_geoms_xform[e].set_rotation(entity.state.orient)
        # render to display or array
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        '''
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        '''
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self._reset_render()

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None


def _out_of_box(x_size, y_size):
    def f(pos):
        '''check whether the some agent is out of bounds'''
        out_of_box = abs(pos) > max(2*x_size, 2*y_size)
        return out_of_box.any()
    return f