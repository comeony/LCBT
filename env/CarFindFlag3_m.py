'''
    @date:2023/05/04
    @time:20:53
    @author:yxy
    The CarFindFlag is a reinforcement learning environment designed to train an agent to navigate a car in a 2D space.
The state space of the environment is defined as the Cartesian product of the position and velocity spaces,
i.e., S = X × V, where X = [-∞, ∞]² represents the position of the car in the x and y directions,
and V = [-∞, ∞]² represents the velocity of the car in the x and y directions.
The action space of the environment is defined as A = [-1, 1]², representing the acceleration of the car in the x and y directions.
The maximum number of steps per episode is limited to 10.
    The environment provides a reward based on the distance between the car and the target position,
with a high reward given if the car reaches the target position and a low reward given otherwise.
The goal of the agent is to learn a policy that maximizes the expected cumulative reward over a finite time horizon.

The dimensions and ranges of the state and action spaces are as follows:
State space: S = (x, y, vx, vy), where x, y, vx, and vy are real numbers, and their ranges are X = [-∞, ∞], Y = [-∞, ∞], VX = [-∞, ∞], and VY = [-∞, ∞], respectively.
Action space: A = (ax, ay), where ax and ay are real numbers, and their ranges are AX = [-1, 1] and AY = [-1, 1], respectively
'''
import gym
from gym import spaces
import numpy as np

DIM = 5

class CarFindFlag3MEnv(gym.Env):
    def __init__(self):
        self.o_range = [0.0, 8.0]
        self.observation_space = spaces.Box(low=self.o_range[0], high=self.o_range[1], shape=(DIM,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(DIM,))
        #self.goal = np.array([4.0, 4.0, 4.0])
        self.goal = np.full(DIM, 4.0)
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

        # 将数组中大于1.0的元素限制为1.0，小于-1.0的元素限制为-1.0
        clipped_arr = np.clip(arr, -1.0, 1.0)
        # ax, ay = clipped_arr
        # self.robot_pos[0] += ax
        # self.robot_pos[1] += ay
        # if self.robot_pos[0] > self.o_range[1]:
        #     self.robot_pos[0] = self.o_range[1]
        # elif self.robot_pos[0] < self.o_range[0]:
        #     self.robot_pos[0] = self.o_range[0]
        # if self.robot_pos[1] > self.o_range[1]:
        #     self.robot_pos[1] = self.o_range[1]
        # elif self.robot_pos[1] < self.o_range[0]:
        #     self.robot_pos[1] = self.o_range[0]
        for i in range(len(clipped_arr)):
            self.robot_pos[i] += clipped_arr[i]
            if self.robot_pos[i] > self.o_range[1]:
                self.robot_pos[i] = self.o_range[1]
            elif self.robot_pos[i] < self.o_range[0]:
                self.robot_pos[i] = self.o_range[0]
        reward = self._get_reward()
        done = self._is_done()
        info = {}
        return self._get_observation(), reward, done, info

    def _samplePos(self):
        # iter_count = 0
        # max_iters = 100
        # pos = np.array([0.0,0.0])
        # while iter_count < max_iters:
        #     pos[0] = np.random.uniform(self.o_range[0], self.o_range[1])
        #     pos[1] = np.random.uniform(self.o_range[0], self.o_range[1])
        #     if not (self.goal[0]-1.0 < pos[0] <= self.goal[0]+1.0 and self.goal[0]-1.0 < pos[1] <= self.goal[0]+1.0):
        #         break
        # return pos
        pos = np.zeros(len(self.goal))
        for i in range(len(pos)):
            while True:
                random_num = np.random.uniform(self.o_range[0], self.o_range[1])
                if abs(random_num - self.goal[i]) > 1.0:
                    pos[i] = random_num
                    break

        return pos

    def _get_observation(self):
        return np.concatenate([self.robot_pos])

    def _get_reward(self):
        distance = np.linalg.norm(self.robot_pos - self.goal)
        # if distance < 1:
        #     r += 0.5
        # if distance < 2:
        #     r += 0.3
        # if distance < 3:
        #     r += 0.2
        r = (5.0 - distance) / 5.0
        # if distance < 1.0:
        #     r = - 2.0
        return r

    def _is_done(self):
        if self.steps >= self.max_steps:
            #print(self.robot_pos)
            return True
        else:
            return False




