
import gym
from gym import spaces
import numpy as np

class CarFindFlagMEnv(gym.Env):
    def __init__(self):
        self.o_range = [0.0,8.0]
        self.observation_space = spaces.Box(low=self.o_range[0], high=self.o_range[1], shape=(2,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.goal = np.array([4.0, 4.0])
        self.max_steps = 10
        self._max_episode_steps = 10
        self.reset()

    def reset(self):
        self.steps = 0
        self.robot_pos = self._samplePos()
        return self._get_observation()

    def step(self, action):
        self.steps += 1
        arr = np.array(action)

        #
        clipped_arr = np.clip(arr, -1.0, 1.0)
        ax, ay = clipped_arr
        self.robot_pos[0] += ax
        self.robot_pos[1] += ay
        if self.robot_pos[0] > self.o_range[1]:
            self.robot_pos[0] = self.o_range[1]
        elif self.robot_pos[0] < self.o_range[0]:
            self.robot_pos[0] = self.o_range[0]
        if self.robot_pos[1] > self.o_range[1]:
            self.robot_pos[1] = self.o_range[1]
        elif self.robot_pos[1] < self.o_range[0]:
            self.robot_pos[1] = self.o_range[0]
        reward = self._get_reward()
        done = self._is_done()
        info = {}
        return self._get_observation(), reward, done, info

    def _samplePos(self):
        iter_count = 0
        max_iters = 100
        pos = np.array([0.0,0.0])
        while iter_count < max_iters:
            pos[0] = np.random.uniform(self.o_range[0], self.o_range[1])
            pos[1] = np.random.uniform(self.o_range[0], self.o_range[1])
            if not (self.goal[0]-1.0 < pos[0] <= self.goal[0]+1.0 and self.goal[0]-1.0 < pos[1] <= self.goal[0]+1.0):
                break
        return pos



    def _get_observation(self):
        return np.concatenate([self.robot_pos])

    def _get_reward(self):
        distance = np.linalg.norm(self.robot_pos - self.goal)
        r = 0
        # if distance < 1:
        #     r += 0.5
        # if distance < 2:
        #     r += 0.3
        # if distance < 3:
        #     r += 0.2
        r = (5.0 - distance) / 5.0

        return r

    def _is_done(self):
        if self.steps >= self.max_steps:
            #print(self.robot_pos)
            return True
        else:
            return False




