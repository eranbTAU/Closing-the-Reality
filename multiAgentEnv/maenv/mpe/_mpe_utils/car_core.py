''' ./mpe_utils '''

import numpy as np
from .forward_models.forward_model import ForwardModel
from .utils import rotate_2D, clip_angle


class EntityState:  # physical/external base state of all entities
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical bearing
        self.orient = None
        # physical velocity
        self.p_vel = None


class AgentState(EntityState):  # state of agents (including communication and internal/mental state)
    def __init__(self):
        super().__init__()
        # communication utterance
        self.c = None


class Action:  # action of the agent
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None


class Entity:  # properties and state of physical world entity
    def __init__(self):
        # name
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass


class Landmark(Entity):  # properties of landmark entities
    def __init__(self):
        super().__init__()


class Agent(Entity):  # properties of agent entities
    def __init__(self):
        super().__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None


class World:  # multi-agent world
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # orientation dimensionality
        self.dim_o = 1
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

        self.forward_model = ForwardModel('car')

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        actions = np.array([a.action.u for a in self.agents])
        state_diff = self.apply_action(actions)
        self.integrate_state(state_diff)

    def apply_action(self, actions):
        # model_pred = self.forward_model.predict_single(actions[0])
        model_pred = self.forward_model.predict_batch(actions)
        return model_pred


    def integrate_state(self, pose_diff):
        for i, agent in enumerate(self.agents):
            rel_pos_diff = rotate_2D(pose_diff[i, :2], agent.state.orient)
            agent.state.p_pos += rel_pos_diff
            agent.state.orient += pose_diff[i,2]
            agent.state.orient = clip_angle(agent.state.orient)
            

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise

