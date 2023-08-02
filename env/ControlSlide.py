import gym
from gym import spaces
import random
import numpy as np
from copy import deepcopy
class ControlSlideEnv(gym.Env):
    """连续动作空间且步长小于10的环境"""
    def __init__(self):
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.state = np.array([0.0])  # 执行器位置初始化在0
        self.max_steps = 10   # 最大步数
        self.bound = 1.0
        self._max_episode_steps = 10
        random.seed(1)
    def step(self, action):
        """

        """
        self.state = deepcopy(self.state)
        action = action[0]
        # 确保步长小于10
        self.state[0] += action * 2.0
        reward = abs(action) if abs(self.state[0]) < self.bound else -1 # 奖励或惩罚
        self.steps += 1
        done = abs(self.state[0]) >= self.bound or self.steps >= self.max_steps # 是否终止
        info = {}
        #print(self.state)
        return self.state, reward, done, info

    def reset(self):
        """重置环境"""
        self.state[0] = np.random.uniform(-0.7,0.7)
        self.steps = 0
        return self.state

