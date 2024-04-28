import numpy as np
import torch
import gym
import argparse
import os

import utils
from utils import get_output_folder
import TD3
import OurDDPG
import DDPG
from attacker.Attacker import Attacker
from env.ControlSlide import ControlSlideEnv
from env.CarFindFlag import CarFindFlagEnv
from env.ControlSlide_AL import ControlSlideALEnv
from copy import deepcopy
from env.CarFindFlag_e import CarFindFlagEEnv
from env.CarFindFlag_m import CarFindFlagMEnv
from env.CarFindFlag3_m import CarFindFlag3MEnv
import math
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment

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

def eval_policy(t_model,a_model, env_name, seed, eval_episodes=10):
	if args.env_name == "ControlSlideEnv":
		print(args.env_name)
		eval_env = ControlSlideEnv()
	elif args.env_name == "CarFindFlagEnv":
		print(args.env_name)
		eval_env = CarFindFlagEnv()
	elif args.env_name == "ControlSlideALEnv":
		print(args.env_name)
		eval_env = ControlSlideALEnv()
	elif args.env_name == "CarFindFlagEEnv":
		print(args.env_name)
		eval_env = CarFindFlagEEnv()
	elif args.env_name == "CarFindFlagMEnv":
		print(args.env_name)
		eval_env = CarFindFlagMEnv()
	elif args.env_name == "CarFindFlag3MEnv":
		print(args.env_name)
		eval_env = CarFindFlag3MEnv()
	else:
		eval_env = gym.make(args.env_name)
	eval_env.seed(seed + 100)
	max_action = eval_env.action_space.high
	min_action = eval_env.action_space.low
	ra = getra_(args.ra_piece,min_action,max_action)
	print(ra)
	avg_reward = 0.
	total_step = 0
	sim_step = 0
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			t_action = t_model.select_action(np.array(state))
			a_action = a_model.select_action(np.array(state))
			if getDistance(t_action,a_action) < ra:
				sim_step += 1
			total_step += 1
			state, reward, done, _ = eval_env.step(t_action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} similarity: {sim_step/total_step}")
	print("---------------------------------------")
	return avg_reward


def test(args, target_model,attacked_model):
	if args.env_name == "ControlSlideEnv":
		print(args.env_name)
		env = ControlSlideEnv()
	elif args.env_name == "CarFindFlagEnv":
		print(args.env_name)
		env = CarFindFlagEnv()
	elif args.env_name == "ControlSlideALEnv":
		print(args.env_name)
		env = ControlSlideALEnv()
	elif args.env_name == "CarFindFlagEEnv":
		print(args.env_name)
		env = CarFindFlagEEnv()
	elif args.env_name == "CarFindFlagMEnv":
		print(args.env_name)
		env = CarFindFlagMEnv()
	elif args.env_name == "CarFindFlag3MEnv":
		print(args.env_name)
		env = CarFindFlag3MEnv()
	else:
		env = gym.make(args.env_name)
	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		t_model = TD3.TD3(**kwargs)
		a_model = TD3.TD3(**kwargs)
	elif args.policy == "OurDDPG":
		t_model = OurDDPG.DDPG(**kwargs)
		a_model = OurDDPG.DDPG(**kwargs)
	elif args.policy == "DDPG":
		t_model = DDPG.DDPG(**kwargs)
		a_model = DDPG.DDPG(**kwargs)

	t_model.load(target_model)
	a_model.load(attacked_model)
	for i in range(1):
		eval_policy(t_model,a_model, args.env_name, args.seed,10000)

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--mode", default="test")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env_name", default="CarFindFlagMEnv")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=2000, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=10000, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e9, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.3)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=64, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.02)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.05)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", default=True)        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument('--target_model', default="./TargetModel/")
	parser.add_argument('--attacked_model', default="./AttackedModel/")
	parser.add_argument('--attack_method', default="black", help='white or black')
	parser.add_argument("--directory", default="./results")
	parser.add_argument('--ra_piece', default=9, type=int)
	args = parser.parse_args()
	attacker = None
	args.directory = get_output_folder(args.directory, args.env_name)
	print(args.directory)
	if not os.path.exists(args.directory):
		os.makedirs(args.directory)
	################## save args ########################
	argsdict = args.__dict__
	with open(args.directory + 'setting.txt', 'w') as f:
		for eachAcg in argsdict:
			f.writelines(str(eachAcg) + ':' + str(argsdict[eachAcg]) + '\n')

	target_model = args.target_model + args.env_name + "/" + args.policy + "/target"
	for i in [7, 8, 9, 10]:
		print(i)
		attacked_model = args.attacked_model + args.env_name + "/" + args.policy +"/" + args.attack_method +"/" + str(i)
		if args.mode == 'test':
			test(args, target_model,attacked_model)