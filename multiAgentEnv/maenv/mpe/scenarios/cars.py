''' ./scenarios '''

import numpy as np

from .._mpe_utils.car_core import Agent, Landmark, World
from .._mpe_utils.scenario import BaseScenario
from .._mpe_utils.car_reward import get_reward_func
from .._mpe_utils.utils import load_params, rotate_2D, clip_angle

# TODO: import scaling parameters

class Scenario(BaseScenario):
    '''
    creates world object, defines scenario parameters and returns it to car_env.py
    '''
    def make_world(self):
        # load env parameters
        params = load_params('/home/roblab1/PycharmProjects/MultiAgentEnv/maenv/mpe/_mpe_utils/car_config.yaml')
        # cretae world object
        world = World()
        # add agents
        world.agents = [Agent() for i in range(params['env_params']['num_agents'])]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent_{i}'
            agent.collide = False
            agent.silent = True
            agent.size = 60
        # add landmarks
        world.landmarks = [Landmark() for i in range(params['env_params']['num_landmarks'])]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 50

        # calc distance scaler
        x_bounds, y_bounds = params['scalers']['x_bounds'], params['scalers']['y_bounds']
        to_mm = params['scalers']['to_mm']
        diag_pix = np.linalg.norm([x_bounds[0] - x_bounds[1], y_bounds[0] - y_bounds[1]])  # in pixels
        diag_mm = diag_pix * to_mm

        # set reward function
        self.calc_reward = get_reward_func(params['env_params']['reward_func'], diag_mm)

        # bounding box size in [mm]
        self.x_size, self.y_size = (x_bounds[1] - x_bounds[0]) * to_mm, (y_bounds[1] - y_bounds[0]) * to_mm

        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.7, 0.0, 0.0])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p) * np.array([self.x_size, self.y_size])
            agent.state.orient = np_random.uniform(-np.pi, np.pi)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p) * np.array([self.x_size, self.y_size]) / 2
            landmark.state.orient = np_random.uniform(-np.pi, np.pi)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):

        other_dist = []
        landmarks_dist = []
        other_rel_dir = []
        landmarks_rel_dir = []
        other_rel_orient = []
        landmarks_rel_orient = []

        for other in world.agents:
            if other is agent:
                continue
            relative_pos = rotate_2D(other.state.p_pos - agent.state.p_pos, agent.state.orient)
            dist = np.linalg.norm(relative_pos)
            relative_dir = np.arctan2(relative_pos[1], relative_pos[0])
            relative_orient = clip_angle(other.state.orient - agent.state.orient)
            other_dist.append(dist)
            other_rel_dir.append(relative_dir)
            other_rel_orient.append(relative_orient)

        for landmark in world.landmarks:
            relative_pos = rotate_2D(landmark.state.p_pos - agent.state.p_pos, agent.state.orient)
            dist = np.linalg.norm(relative_pos)
            relative_dir = np.arctan2(relative_pos[1], relative_pos[0])
            relative_orient = clip_angle(landmark.state.orient - agent.state.orient)
            landmarks_dist.append(dist)
            landmarks_rel_dir.append(relative_dir)
            landmarks_rel_orient.append(relative_orient)

        agent_pose = [agent.state.p_pos[0], agent.state.p_pos[1], agent.state.orient]
        rew = self.calc_reward(agent.action.u, agent_pose, other_dist, landmarks_dist, other_rel_dir,
                               landmarks_rel_dir, other_rel_orient, landmarks_rel_orient)
        return rew
        # dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
        # return -dist2


    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        other_pos = []
        for other in world.agents:
            if other == agent:
                continue
            # relative (x,y) target position in agent's reference frame
            relative_pos = rotate_2D(other.state.p_pos - agent.state.p_pos,  agent.state.orient)
            relative_orient = clip_angle(other.state.orient - agent.state.orient)
            other_pos.append(relative_pos)
            other_pos.append([relative_orient])

        for entity in world.landmarks:
            relative_pos = rotate_2D(entity.state.p_pos - agent.state.p_pos, agent.state.orient)
            relative_orient = clip_angle(entity.state.orient - agent.state.orient)
            entity_pos.append(relative_pos)
            entity_pos.append([relative_orient])

        return np.concatenate(other_pos + entity_pos)

    def state(self, world):
        # get positions of all entities in this agent's reference frame
        landmark_pose = []
        agent_pose = []
        for agent in world.agents:
            agent_pose.append(agent.state.p_pos)
            agent_pose.append([agent.state.orient])

        for landmark in world.landmarks:
            landmark_pose.append(landmark.state.p_pos)
            landmark_pose.append([landmark.state.orient])

        return np.concatenate(agent_pose + landmark_pose)

