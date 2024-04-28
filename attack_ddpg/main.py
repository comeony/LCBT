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

def train(train_epoch, agent, env,  evaluate, validate_steps, output, max_episode_length=None, debug=False,attacker = None):
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    tarj = []
    totalReward = []
    e_t = []
    while episode < train_epoch:
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        # agent pick action ...
        if step <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation)

        tarAction = action
        if args.ATTACK:
            tarAction, wh = attacker.antiAction(action, episode_steps, observation)

        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step(tarAction)
        observation2 = deepcopy(observation2)
        if max_episode_length and episode_steps >= max_episode_length -1:
            done = True

        if args.ATTACK:
            tarj.append([tarAction, reward, observation, observation2, wh])

        # agent observe and update policy
        agent.observe(reward, observation2, done)
        if step > args.warmup :
            agent.update_policy()
        
        # [optional] evaluate

        if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:
            policy = lambda x: agent.select_action(x, decay_epsilon=False)
            validate_reward = evaluate(env, policy, debug=False, visualize=False)
            prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))
            if validate_reward >= 40 and validate_reward <= 70:
                agent.save_model(output + str(step) + "_" + str(validate_reward))
        # update 
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done: # end of episode
            if episode % 100 == 0:
                prGreen('#{}: episode_reward:{} steps:{}'.format(episode,episode_reward,step))

            agent.memory.append(
                observation,
                agent.select_action(observation),
                0., False
            )
            if episode % 10000 == 0:
                agent.save_model(output + str(episode) + "_" + str(episode_reward))
                np.save(output + "reward.npy",np.array(totalReward))
                np.save(output + "steps.npy",np.array(e_t))
                if args.ATTACK:
                    np.save(output + "sim.npy", np.array(attacker.similarity))
            # [optional] save intermideate model

            if args.ATTACK:
                attacker.update(tarj)
                tarj = []

            #if episode_steps > 5:
            episode += 1
            # reset
            e_t.append(episode_steps)
            totalReward.append(episode_reward)
            observation = None
            episode_steps = 0
            episode_reward = 0.

def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False,attacker = 0):

    agent.load_weights(model_path)
    print(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)
    result = []
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
            action = policy(observation)
            tarAction = action
            print(action, observation)
            if args.ATTACK:
                tarAction, wh = attacker.antiAction(action, episode_steps, observation)
            observation, reward, done, info = env.step(tarAction)
            if args.max_episode_length and episode_steps >= args.max_episode_length - 1:
                done = True

            if visualize:
                env.render(mode='human')

            # update
            episode_reward += reward
            episode_steps += 1

        prYellow('[Evaluate] #Episode{}: episode_reward:{}'.format(episode, episode_reward))
        result.append(episode_reward)

    print(sum(result) / len(result))

    '''
    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
        prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))
    '''


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')
    #ControlSlideEnv:
    parser.add_argument('--mode', default='test', type=str, help='support option: train/test')
    parser.add_argument('--env_name', default='ControlSlideEnv', type=str, help='open-ai gym environment')
    #Pendulum-v0 MountainCarContinuous-v0 ControlSlideEnv CarFindFlagEnv
    parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=10000, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.5, type=float, help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=10, type=int, help='')
    parser.add_argument('--validate_steps', default=500, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output/attack_ddpg', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_epoch', default=100000, type=int, help='train epoch')
    parser.add_argument('--epsilon', default=1000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=0, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    # parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay') # TODO
    # parser.add_argument('--cuda', dest='cuda', action='store_true') # TODO
    #attack
    parser.add_argument('--ATTACK', default=True, type=bool, help='Attack or not')
    parser.add_argument('--attack_method', default="black",help='white or black')
    parser.add_argument('--ls', default=1.0, type=float)
    parser.add_argument('--p', default=0.5, type=float)
    parser.add_argument('--rs_piece', default=16, type=int,help="The number of shards per dimension in the state")
    parser.add_argument('--ra_piece', default= 32, type=float)
    parser.add_argument('--attack_target_model', default="./TargetModel/")
    parser.add_argument('--delta', default=0.05, type=float)
    parser.add_argument('--isWeak', default=False, type=bool)
    parser.add_argument('--multiples_of_v', default=4, type=int)
    parser.add_argument("--attack_random_seed", default=0, type=int)
    parser.add_argument('--lrs', default=1, type=int)
    parser.add_argument('--beginAttackK', default=0, type=int)
    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env_name)
    print(args.output)
    if args.resume == 'default':
        args.resume = args.attack_target_model + args.env_name + "/target7_"

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
    print(nb_states,max_state,min_state)
    print(nb_actions,max_action,min_action)
    print(max_reward)

    attacker = 0
    if args.ATTACK:
        targetAgent = DDPG(nb_states, nb_actions, args)
        attack_target_model = args.attack_target_model + args.env_name + "/target_"
        print(attack_target_model)
        targetAgent.load_weights(attack_target_model)
        targetAgent.is_training = False
        targetAgent.eval()
        # targetAgent, s_dim, a_dim, min_a, max_a, min_s, max_s,args
        attacker_policy = lambda x: targetAgent.select_action(x, decay_epsilon=False)
        attacker = Attacker(attacker_policy, nb_states, nb_actions, min_action, max_action, min_state, max_state, args)

    agent = DDPG(nb_states, nb_actions, args)
    evaluate = Evaluator(args.validate_episodes, 
        args.validate_steps, args.output, max_episode_length=args.max_episode_length)

    ################## save args ########################
    argsdict = args.__dict__
    with open(args.output + 'setting.txt', 'w') as f:
        for eachAcg in argsdict:
            f.writelines(str(eachAcg) + ':' + str(argsdict[eachAcg]) + '\n')

    if args.mode == 'train':
        train(args.train_epoch, agent, env, evaluate,
            args.validate_steps, args.output, max_episode_length=args.max_episode_length, debug=args.debug,attacker = attacker)

    elif args.mode == 'test':
        print(args.resume)
        test(100, agent, env, evaluate, args.resume,
            visualize=False, debug=args.debug,attacker = attacker)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
