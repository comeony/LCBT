import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym
import sys
sys.path.append("..")
from attacker.Attacker import Attacker
from env.ControlSlide import ControlSlideEnv
from env.CarFindFlag import CarFindFlagEnv
from env.CarFindFlag_e import CarFindFlagEEnv
from env.CarFindFlag_m import CarFindFlagMEnv
from PPO import PPO


#################################### Testing ###################################
def test():
    print("============================================================================================")

    ################## hyperparameters ##################

    # env_name = "CartPole-v1"
    # has_continuous_action_space = False
    # max_ep_len = 400
    # action_std = None

    # env_name = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # env_name = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    env_name = "ControlSlideEnv"
    has_continuous_action_space = True
    max_ep_len = 10           # max timesteps in one episode
    action_std = 0.00001            # set same std for action distribution which was used while saving

    render = False              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 1000    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################

    if env_name == "ControlSlideEnv":
        env = ControlSlideEnv()
    elif env_name == "CarFindFlagEnv":
        env = CarFindFlagEnv()
    elif env_name == "CarFindFlagEEnv":
        print(env_name)
        env = CarFindFlagEEnv()
    elif env_name == "CarFindFlagMEnv":
        print(env_name)
        env = CarFindFlagMEnv()

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    # preTrained weights directory
    ppo_agent.eval()
    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    #directory = "./PPO_preTrained/ControlSlideEnv-run115/target7_model.pth"
    directory = "./TargetModel/ControlSlideEnv/target7_model.pth"
    checkpoint_path = directory
    #checkpoint_path = "PPO_preTrained/CarFindFlagEnv-run56/PPO_CarFindFlagEnv_74000_9.0.pth"

    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()

        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state,Train=False)
            state, reward, done, _ = env.step(action)
            ep_reward += reward

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                break

        # clear buffer
        #ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':

    test()
