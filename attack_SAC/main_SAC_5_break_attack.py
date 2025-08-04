import sys
sys.path.append('../')
import os
import argparse
from datetime import datetime
import gym

from agent import SacAgent
from env.CarFindFlag_m import CarFindFlagMEnv
from env.ControlSlide import ControlSlideEnv
from env.CarFindFlag3_m import CarFindFlag3MEnv
from attacker.Attacker import Attacker
import torch
import numpy as np
import math
def getDistance(a, b):
    dist = math.sqrt(sum([(xi - yi) ** 2 for xi, yi in zip(a, b)]))
    return dist
def getra_(ra_piece, min_a,max_a):
	n = []
	f = []
	for i in range(len(min_a)):
		f.append((max_a[i] - min_a[i]) / ra_piece)
		n.append([0.0])
	return getDistance(n, f)

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",default="test")
    parser.add_argument('--env_name', type=str, default='CarFindFlag3MEnv')
    parser.add_argument('--cuda', default=True,type=bool)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument("--max_episode_length", default=10, type=int)  # Frequency of delayed policy updates
    parser.add_argument('--ATTACK', default=True, type=bool, help='Attack or not')
    parser.add_argument('--attack_method', default="black", help='white or black')
    parser.add_argument('--ls', default=1.0, type=float)
    parser.add_argument('--p', default=pow(2, -1.0/5.0), type=float)
    parser.add_argument('--rs_piece', default=9, type=int, help="The number of shards per dimension in the state")
    parser.add_argument('--ra_piece', default=9, type=int)
    parser.add_argument('--attack_target_model', default="./TargetModel/")
    parser.add_argument('--delta', default=0.05, type=float)
    parser.add_argument('--isWeak', default=False, type=bool)
    parser.add_argument('--multiples_of_v', default=1, type=int)
    parser.add_argument('--lrs', default=2, type=int)
    parser.add_argument('--describe', default="", )
    parser.add_argument('--beginAttackK', default=50, type=int)
    parser.add_argument("--attack_random_seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds


    args = parser.parse_args()

    # You can define configs in the external json or yaml file.
    configs = {
        'num_steps': 1100000,
        'batch_size': 256,
        'lr': 0.0003,
        'hidden_units': [256, 256],
        'memory_size': 1e6,
        'gamma': 0.99,
        'tau': 0.005,
        'entropy_tuning': True,
        'ent_coef': 0.2,  # It's ignored when entropy_tuning=True.
        'multi_step': 1,
        'per': False,  # prioritized experience replay
        'alpha': 0.6,  # It's ignored when per=False.
        'beta': 0.4,  # It's ignored when per=False.
        'beta_annealing': 0.0001,  # It's ignored when per=False.
        'grad_clip': None,
        'updates_per_step': 1,
        'start_steps': 10000,
        'log_interval': 10,
        'target_update_interval': 1,
        'eval_interval': 100000,
        'cuda': args.cuda,
        'seed': args.seed
    }


    if args.env_name == "CarFindFlagMEnv":
        env = CarFindFlagMEnv()
    elif args.env_name == "ControlSlideEnv":
        env = ControlSlideEnv()
    elif args.env_name == "CarFindFlag3MEnv":
        print(args.env_name)
        env = CarFindFlag3MEnv()
    else:
        env = gym.make(args.env_id)
    log_dir = os.path.join(
        'logs', args.env_name,
        f'sac-seed{args.seed}-{datetime.now().strftime("%Y%m%d-%H%M")}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    ################## save args ########################
    argsdict = args.__dict__
    with open(log_dir + '/setting.txt', 'w') as f:
        for eachAcg in argsdict:
            f.writelines(str(eachAcg) + ':' + str(argsdict[eachAcg]) + '\n')
        for eachAcg in configs:
            f.writelines(str(eachAcg) + ':' + str(configs[eachAcg]) + '\n')

    #####################################################
    agent = SacAgent(env=env, log_dir=log_dir, **configs)
    attacker = None
    #####################################################
    if args.ATTACK:
        # Set seeds
        env.seed(args.seed)
        env.action_space.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        nb_states = env.observation_space.shape[0]
        nb_actions = env.action_space.shape[0]
        max_state = env.observation_space.high
        min_state = env.observation_space.low
        max_action = env.action_space.high
        min_action = env.action_space.low
        max_reward = env.reward_range

        targetAgent = SacAgent(env=env, log_dir=log_dir, **configs)

        attack_target_model = args.attack_target_model + args.env_name + "/"
        print(attack_target_model)
        targetAgent.load_model(attack_target_model)
        # targetAgent, s_dim, a_dim, min_a, max_a, min_s, max_s,args
        attacker_policy = lambda x: targetAgent.exploit(x)
        attacker = Attacker(attacker_policy, nb_states, nb_actions, min_action, max_action, min_state, max_state, args)


    if args.mode == "train":
        agent.run(attacker)
    elif args.mode == "test":
        agent.load_model("./TargetModel/CarFindFlag3MEnv/")
        if args.ATTACK:
            attack_agent = SacAgent(env=env, log_dir=log_dir, **configs)
            attack_agent.load_model("./logs/CarFindFlag3MEnv/sac-seed0-20250330-2338/model/")
        evl_rewards = []
        ra = getra_(args.ra_piece, min_action,max_action)
        for i in range(1):
            print("ra:", ra)
            evl_rewards.append(agent.evaluate(attack_agent, ra))
        print(np.mean(evl_rewards))


if __name__ == '__main__':
    run()
