import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

from MARL.DDPG.agents.replay_buffer import ReplayBuffer
from MARL.DDPG.agents.ou_noise import OUnoise

class CriticNetwork(nn.Module):
    def __init__(self, name, beta, input_dims , action_dim, fc1_dim, fc2_dim, chkpt_dir='models', device=T.device('cpu')):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.action_dim = action_dim
        self.beta = beta
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.checkpoint_dir = chkpt_dir
        # filename extension:
        self.name = 'critic_'+name
        self.checkpoint_file = os.path.join(self.checkpoint_dir,self.name+'.pt')
        # ---------structure---------
        # self.fc1 = nn.Linear(self.input_dims, self.fc1_dim)
        # self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        # self.bn1 = nn.LayerNorm(self.fc1_dim)
        # self.bn2 = nn.LayerNorm(self.fc2_dim)
        # self.action_value = nn.Linear(self.action_dim, self.fc2_dim)
        # self.q = nn.Linear(self.fc2_dim,1)

        self.fc1 = nn.Linear(self.input_dims+self.action_dim, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.bn1 = nn.LayerNorm(self.fc1_dim)
        self.bn2 = nn.LayerNorm(self.fc2_dim)
        self.q = nn.Linear(self.fc2_dim,1)

        self.init_weights()

        self.optimizer = optim.Adam(self.parameters(), lr = self.beta, weight_decay=0.01)
        #self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = device
        self.to(self.device)


    def init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)
        f3 = 0.003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)
        # nn.init.xavier_uniform_(self.action_value.weight)
        # self.action_value.bias.data.fill_(0.01)

    # def init_weights(self):
    #     f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
    #     self.fc1.weight.data.uniform_(-f1, f1)
    #     self.fc1.bias.data.uniform_(-f1, f1)
    #     f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
    #     self.fc2.weight.data.uniform_(-f2, f2)
    #     self.fc2.bias.data.uniform_(-f2, f2)
    #     f3 = 0.003
    #     self.q.weight.data.uniform_(-f3, f3)
    #     self.q.bias.data.uniform_(-f3, f3)
    #     f4 = 1. / np.sqrt(self.action_value.weight.data.size()[0])
    #     self.action_value.weight.data.uniform_(-f4, f4)
    #     self.action_value.bias.data.uniform_(-f4, f4)

    # def forward(self, state, action):
    #     state_value = self.fc1(state)
    #     state_value = self.bn1(state_value)
    #     state_value = F.relu(state_value)
    #     state_value = self.fc2(state_value)
    #     state_value = self.bn2(state_value)
    #     action_value = self.action_value(action)
    #     state_action_value = F.relu(T.add(state_value, action_value))
    #     state_action_value = self.q(state_action_value)
    #
    #     return state_action_value

    def forward(self, state, action):
        state_action = T.cat([state, action], dim=1)
        x = F.relu(self.bn1(self.fc1(state_action)))
        x = F.relu(self.bn2(self.fc2(x)))
        state_action_value = self.q(x)
        return state_action_value

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, name, input_dims, action_dim, fc1_dim=256, fc2_dim=256, chkpt_dir='models', device=T.device('cpu')):
        super(ActorNetwork, self).__init__()
        self.lr = alpha
        self.input_dims = input_dims
        self.action_dim = action_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.checkpoint_dir = chkpt_dir
        # filename extension:

        self.name = 'actor_'+name
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'.pt')

        # ----- structure ---------
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.bn1 = nn.LayerNorm(self.fc1_dim)
        self.bn2 = nn.LayerNorm(self.fc2_dim)
        self.mu = nn.Linear(self.fc2_dim, self.action_dim)

        self.init_weights()
        self.optimizer = optim.Adam(self.parameters(), lr = alpha)
        #self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = device
        self.to(self.device)

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.mu.weight)
        self.mu.bias.data.fill_(0.01)

    # def init_weights(self):
    #     f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
    #     self.fc1.weight.data.uniform_(-f1,f1)
    #     self.fc1.bias.data.uniform_(-f1,f1)
    #     f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
    #     self.fc2.weight.data.uniform_(-f2, f2)
    #     self.fc2.bias.data.uniform_(-f2, f2)
    #     self.mu.weight.data.uniform_(-3e-3,3e-3)
    #     self.mu.bias.data.uniform_(-3e-3,3e-3)


    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.mu(x)
        x = T.sigmoid(x)
        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))

class Agent():
    def __init__(self, params):
        self.params = params['agent_params']
        self.name = os.path.split(params['gen_params']['save_dir'])[1]
        alpha = self.params['alpha']
        beta = self.params['beta']
        self.gamma = self.params['gamma']
        self.tau = self.params['tau']
        self.batch_size = self.params['batch_size']
        self.is_training = params['gen_params']['is_train']
        input_dim = self.params['obs_dim']
        action_dim = self.params['action_dim']
        fc1 = self.params['fc_dims']
        fc2 = self.params['fc_dims']
        buffer_size = self.params['buffer_size']
        chkpt_dir = os.path.join(params['gen_params']['save_dir'],'model')
        self.device = T.device(self.params['agent_device'])

        self.memory = ReplayBuffer(size=buffer_size, batch_size=self.batch_size, state_dim=input_dim, action_dim=action_dim)
        self.noise = OUnoise(params, mu=np.zeros(action_dim))
        self.actor = ActorNetwork(alpha, self.name, input_dim, action_dim, fc1, fc2, chkpt_dir, self.device)
        self.critic = CriticNetwork(self.name, beta, input_dim, action_dim, fc1, fc2, chkpt_dir, self.device)
        self.actor_target = ActorNetwork(alpha, self.name+'_target', input_dim, action_dim, fc1, fc2, chkpt_dir, self.device)
        self.critic_target = CriticNetwork(self.name+'_target', beta, input_dim, action_dim, fc1, fc2, chkpt_dir, self.device)

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        if self.is_training:
            mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
            self.actor.train()
        else:
            mu_prime = mu
        return mu_prime.cpu().detach().clone().numpy()[0]

    def store_experience(self, state, action, reward, state_, done):
        self.memory.store(state, action, reward, state_, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.actor_target.save_checkpoint()
        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()
        print(f'models saved as: {self.name}')

    def load_models(self):
        self.actor.load_checkpoint()
        self.actor_target.load_checkpoint()
        self.critic.load_checkpoint()
        self.critic_target.load_checkpoint()

    def learn(self):
        if self.memory.cntr <= self.memory.batch_size*50:
            return
        states_batch, actions_batch, reward_batch, _states_batch, dones_batch = self.memory.sample()
        states = T.tensor(states_batch, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions_batch, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(reward_batch, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(_states_batch, dtype=T.float).to(self.actor.device)
        dones = T.tensor(dones_batch).to(self.actor.device)

        target_actions = self.actor_target.forward(states_) # mu
        target_value_ = self.critic_target.forward(states_, target_actions) # target_Q(s',mu(s'))
        value = self.critic.forward(states, actions) # Q(s,a)

        target_value_[dones] = 0.0
        target_value_ = target_value_.view(-1)

        target = rewards + self.gamma*target_value_
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()
        return value.detach().cpu().numpy()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.actor_target.named_parameters()
        target_critic_params = self.critic_target.named_parameters()

        actor_state_dict= dict(actor_params)
        critic_state_dict= dict(critic_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_state_dict = dict(target_critic_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*target_critic_state_dict[name].clone()
        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                      (1 - tau) * target_actor_state_dict[name].clone()

        self.critic_target.load_state_dict(critic_state_dict)
        self.actor_target.load_state_dict(actor_state_dict)


