import numpy as np

class ReplayBuffer():
    def __init__(self, size=10**6, batch_size = 64, state_dim = 14, action_dim = 2):
        self.buffer_size = size
        self.states_buffer = np.zeros((size, state_dim),  dtype=np.float)
        self.actions_buffer = np.zeros((size, action_dim),  dtype=np.float)
        self.reward_buffer = np.zeros(size,  dtype=np.float)
        self._states_buffer = np.zeros((size, state_dim),  dtype=np.float)
        self.dones_buffer = np.zeros(size,  dtype=np.bool)
        self.batch_size = batch_size
        self.cntr = 0
        self.rng = np.random.default_rng()

    def store(self, state, action, reward, state_, done):
        idx = self.cntr % self.buffer_size
        self.states_buffer[idx]= state
        self.actions_buffer[idx]= action
        self.reward_buffer[idx]= reward
        self._states_buffer[idx]= state_
        self.dones_buffer[idx] = done
        self.cntr+=1

    def sample(self):
        indices = self.rng.choice(min(self.cntr, self.buffer_size), self.batch_size, replace=False)
        states_batch = np.take(self.states_buffer, indices, axis=0)
        actions_batch = np.take(self.actions_buffer, indices, axis=0)
        reward_batch = np.take(self.reward_buffer, indices, axis=0)
        _states_batch = np.take(self._states_buffer, indices, axis=0)
        dones_batch = np.take(self.dones_buffer, indices, axis=0)

        return states_batch, actions_batch, reward_batch, _states_batch, dones_batch
