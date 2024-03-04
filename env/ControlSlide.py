import gym
from gym import spaces
import random
import numpy as np
from copy import deepcopy
class ControlSlideEnv(gym.Env):

    def __init__(self):
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.state = np.array([0.0])  #
        self.max_steps = 10   #
        self.bound = 1.0
        self._max_episode_steps = 10
        random.seed(1)
    def step(self, action):
        """

        """
        self.state = deepcopy(self.state)
        action = action[0]
        #
        self.state[0] += action * 2.0
        reward = abs(action) if abs(self.state[0]) < self.bound else -1 #
        self.steps += 1
        done = abs(self.state[0]) >= self.bound or self.steps >= self.max_steps #
        info = {}
        #print(self.state)
        return self.state, reward, done, info

    def reset(self):

        self.state[0] = np.random.uniform(-0.7,0.7)
        self.steps = 0
        return self.state

