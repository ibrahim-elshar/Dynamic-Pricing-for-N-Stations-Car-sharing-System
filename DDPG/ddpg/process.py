import numpy as np


class OrnsteinUhlenbeck(object):

    def __init__(self, x0, theta, mu, sigma, dt=1e-2):
        self.x0 = x0
        self.x = x0
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt

    def __call__(self):
        self.x = self.x + self.theta * (self.mu - self.x) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.x.shape)
        return self.x

    def clear(self):
        self.x = self.x0

class normal_noise(object):
    
    def __init__(self, action_noise ,action_noise_decay):
        self.action_noise = action_noise
        self.action_noise_decay = action_noise_decay
        
        
    def __call__(self, action):     
        action = np.random.normal(action, self.action_noise)
        self.action_noise *= self.action_noise_decay
        return action
    
    def clear(self):
        pass  