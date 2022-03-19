import numpy as np

class OUnoise():
    def __init__(self, params, mu, sigma=1, theta=1, dt=0.01, x0=None):
        try:
            self.theta = params['noise_theta']
            self.sigma = params['noise_sigma']
        except:
            self.theta = theta
            self.sigma = sigma
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.sigma* np.sqrt(self.dt) * np.random.normal(size = self.mu.shape)
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)