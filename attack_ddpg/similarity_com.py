#!/usr/bin/env python3

import numpy as np
import argparse
from copy import deepcopy
import torch
import gym

from normalized_env import NormalizedEnv
from evaluator import Evaluator
from ddpg import DDPG
from util import *
import sys

sys.path.append("..")
from attacker.Attacker import Attacker
from env.ControlSlide import ControlSlideEnv
from env.CarFindFlag import CarFindFlagEnv
from env.CarFindFlag_e import CarFindFlagEEnv
import math
def getra_(ra_piece, min_a,max_a):
	n = []
	f = []
	for i in range(len(min_a)):
		f.append((max_a[i] - min_a[i]) / ra_piece)
		n.append([0.0])
	return getDistance(n, f)

def getDistance(a, b):
	dist = math.sqrt(sum([(xi - yi) ** 2 for xi, yi in zip(a, b)]))
	return dist

def test(num_episodes, target_agent, attacked_agent,env, ra):

    target_policy = lambda x: target_agent.select_action(x, decay_epsilon=False)
    attacked_policy = lambda x: attacked_agent.select_action(x, decay_epsilon=False)
    result = []
    total_step = 0
    sim_step = 0
    for episode in range(num_episodes):

        # reset at the start of episode
        observation = env.reset()
        episode_steps = 0
        episode_reward = 0.

        assert observation is not None

        # start episode
        done = False
        while not done:
            # basic operation, action ,reward, blablabla ...
            target_action = target_policy(observation)
            attacked_action = attacked_policy(observation)
            observation, reward, done, info = env.step(target_action)
            if args.max_episode_length and episode_steps >= args.max_episode_length - 1:
                done = True
            if getDistance(target_action,attacked_action) < ra:
                sim_step += 1
            total_step += 1

            # update
            episode_reward += reward
            episode_steps += 1
        if episode % 100 == 0:
            prYellow('[Evaluate] #Episode{}: episode_reward:{}'.format(episode, episode_reward))
        result.append(episode_reward)
    print('similarity: {}'.format(sim_step / total_step))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')
    # ControlSlideEnv:
    parser.add_argument('--mode', default='test', type=str, help='support option: train/test')
    parser.add_argument('--env_name', default='ControlSlideEnv', type=str, help='open-ai gym environment')
    # Pendulum-v0 MountainCarContinuous-v0 ControlSlideEnv CarFindFlagEnv
    parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=10000, type=int,
                        help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.4, type=float, help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
    parser.add_argument('--validate_episodes', default=20, type=int,
                        help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=10, type=int, help='')
    parser.add_argument('--validate_steps', default=500, type=int,
                        help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output/attack_ddpg', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_epoch', default=5000000, type=int, help='train epoch')
    parser.add_argument('--epsilon', default=1000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=2, type=int, help='')

    # attack
    parser.add_argument('--attack_method', default="white", help='white or black')

    parser.add_argument('--ra_piece', default=32, type=float)
    parser.add_argument('--target_model', default="./TargetModel/")
    parser.add_argument('--attacked_model', default="./AttackedModel/")
    args = parser.parse_args()

    if args.env_name == "ControlSlideEnv":
        print(args.env_name)
        env = ControlSlideEnv()
    elif args.env_name == "CarFindFlagEnv":
        print(args.env_name)
        env = CarFindFlagEnv()
    elif args.env_name == "CarFindFlagEEnv":
        print(args.env_name)
        env = CarFindFlagEEnv()
    else:
        env = NormalizedEnv(gym.make(args.env_name))
    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]
    max_state = env.observation_space.high
    min_state = env.observation_space.low
    max_action = env.action_space.high
    min_action = env.action_space.low
    max_reward = env.reward_range
    print(nb_states, max_state, min_state)
    print(nb_actions, max_action, min_action)
    print(max_reward)


    target_agent = DDPG(nb_states, nb_actions, args)
    target_path = args.target_model + args.env_name + "/target7_"
    target_agent.load_weights(target_path)
    target_agent.eval()
    target_agent.is_training = False

    attacked_agent = DDPG(nb_states, nb_actions, args)
    attacked_path = args.attacked_model + args.env_name + "/" + args.attack_method + "/attacked3_"
    attacked_agent.load_weights(attacked_path)
    attacked_agent.eval()
    attacked_agent.is_training = False

    ra = getra_(args.ra_piece, min_action,max_action)
    if args.mode == 'test':
        test(10000, target_agent, attacked_agent,env, ra)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
