import numpy as np
from torch.distributions import MultivariateNormal


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:-1]
        del self.states[:-1]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
import torch
from torch.distributions import MultivariateNormal

import numpy as np
max_action = 1.0
s = np.array([0.0,0.0])
print(s)
for i in range(2):
    s[i] = np.random.uniform(low = 0.3,high = .7)
robot_pos = np.array([0.0, 0.0])
print(s)
print((2/9)*(2**(1/2)))
min_a = np.array([1.0,2.0])
max_a =  np.array([2.0,10.0])
a_range = [min_a, max_a]
a_mean = (min_a + max_a) / 2.0
print(a_range)
print(a_mean)
print(2.5e5)