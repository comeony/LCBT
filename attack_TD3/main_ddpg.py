import numpy as np
import torch
import gym
import argparse
import os
import time

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
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
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

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward
	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward

def train(args,attacker = None):


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
	print("state_dim",state_dim)
	print("action_dim",action_dim)
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
		policy = TD3.TD3(**kwargs)
	elif args.policy == "OurDDPG":
		policy = OurDDPG.DDPG(**kwargs)
	elif args.policy == "DDPG":
		policy = DDPG.DDPG(**kwargs)

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env_name, args.seed)]

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	e_r = []
	tarj = []

	all_time = [[0, 0.0, 0.0]]
	all_time_start = time.time()
	for t in range(int(args.max_timesteps)):

		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
					policy.select_action(np.array(state))
					+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

			if t % 10000 == 0:
				args.expl_noise = max(args.expl_noise - 0.04, 0.02)
				print("args.expl_noise:", args.expl_noise)

		attacker_time_begin = time.time()
		tarAction = deepcopy(action)
		if args.ATTACK:
			tarAction, wh = attacker.antiAction(action, episode_timesteps - 1, state)
		tarAction = np.array(tarAction)
		attacker_time_end = time.time()
		all_time[-1][1] = all_time[-1][1] + attacker_time_end - attacker_time_begin

		# Perform action
		next_state, reward, done, _ = env.step(tarAction)
		done_bool = float(done) if episode_timesteps < env.max_steps else 0
		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		if args.ATTACK:
			attacker_time_begin = time.time()
			tarj.append([tarAction, reward, state, next_state, wh])
			attacker_time_end = time.time()
			all_time[-1][1] = all_time[-1][1] + attacker_time_end - attacker_time_begin

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		if done:
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			if (episode_num + 1) % 100 == 0:
				evaluations.append(eval_policy(policy, args.env_name, args.seed))
				np.save(args.directory + "eval.npy", evaluations)
				np.save(args.directory + "reward.npy", np.array(e_r))
				print(
					f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")

			# Evaluate episode
			if (episode_num + 1) % args.eval_freq == 0:
				if args.save_model: policy.save(args.directory + "{}_{}".format(episode_num, episode_reward))
				if args.ATTACK:
					print("similarity save")
					np.save(args.directory + "sim.npy", np.array(attacker.similarity))

			# Reset environment
			e_r.append(episode_reward)
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1
			if args.ATTACK:
				attacker_time_begin = time.time()
				attacker.update(tarj)
				tarj = []
				attacker_time_end = time.time()
				all_time[-1][1] = all_time[-1][1] + attacker_time_end - attacker_time_begin

			all_time_end = time.time()
			all_time[-1][2] = all_time[-1][2] + all_time_end - all_time_start
			all_time[-1][0] = t + 1
			all_time += [all_time[-1][:]]
			if args.ATTACK and (episode_num + 1) % 100 == 0:
				np.save(args.directory + "times.npy", np.array(all_time))
				t = np.load(args.directory + "times.npy")
			all_time_start = time.time()


def test(args):
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
		policy = TD3.TD3(**kwargs)
	elif args.policy == "OurDDPG":
		policy = OurDDPG.DDPG(**kwargs)
	elif args.policy == "DDPG":
		policy = DDPG.DDPG(**kwargs)

	test_target = args.attack_target_model + args.env_name + "/" + args.policy + "/target5"
	policy.load(test_target)
	for i in range(1):
		eval_policy(policy, args.env_name, args.seed,100)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="DDPG")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--mode", default="test")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env_name", default="CarFindFlag3MEnv")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=2000, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=10000, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=2e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.3)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=64, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.02)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.05)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", default=True)  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--directory", default="./results")
    # attack

    parser.add_argument("--max_episode_length", default=10, type=int)  # Frequency of delayed policy updates
    parser.add_argument('--ATTACK', default=True, type=bool, help='Attack or not')
    parser.add_argument('--attack_method', default="white", help='white or black')
    parser.add_argument('--ls', default=1.0, type=float)
    parser.add_argument('--p', default=pow(2, -1.0 / 5.0), type=float)
    parser.add_argument('--rs_piece', default=9, type=int, help="The number of shards per dimension in the state")
    parser.add_argument('--ra_piece', default=9, type=int)
    parser.add_argument('--attack_target_model', default="./TargetModel/")
    parser.add_argument('--delta', default=0.05, type=float)
    parser.add_argument('--isWeak', default=False, type=bool)
    parser.add_argument('--lrs', default=2, type=int)
    parser.add_argument("--attack_random_seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument('--multiples_of_v', default=1, type=int)
    parser.add_argument('--alpha', default=5, type=float)
    parser.add_argument('--beginAttackK', default=100, type=int)
    args = parser.parse_args()
    attacker = None
    args.directory = get_output_folder(args.directory, args.env_name)
    print(args.directory)
    print(args.p)
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
    ################## save args ########################
    argsdict = args.__dict__
    with open(args.directory + 'setting.txt', 'w') as f:
        for eachAcg in argsdict:
            f.writelines(str(eachAcg) + ':' + str(argsdict[eachAcg]) + '\n')

    #####################################################
    if args.ATTACK:
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

        nb_states = env.observation_space.shape[0]
        nb_actions = env.action_space.shape[0]
        max_state = env.observation_space.high
        min_state = env.observation_space.low
        max_action = env.action_space.high
        min_action = env.action_space.low
        max_reward = env.reward_range

        kwargs = {
            "state_dim": nb_states,
            "action_dim": nb_actions,
            "max_action": float(max_action[0]),
            "discount": args.discount,
            "tau": args.tau,
        }
        if args.policy == "TD3":
            # Target policy smoothing is scaled wrt the action scale
            kwargs["policy_noise"] = args.policy_noise * max_action
            kwargs["noise_clip"] = args.noise_clip * max_action
            kwargs["policy_freq"] = args.policy_freq
            targetAgent = TD3.TD3(**kwargs)
        elif args.policy == "OurDDPG":
            targetAgent = OurDDPG.DDPG(**kwargs)
        elif args.policy == "DDPG":
            targetAgent = DDPG.DDPG(**kwargs)
        attack_target_model = args.attack_target_model + args.env_name + "/" + args.policy + "/target"
        print(attack_target_model)
        targetAgent.load(attack_target_model)
        # targetAgent, s_dim, a_dim, min_a, max_a, min_s, max_s,args
        attacker_policy = lambda x: targetAgent.select_action(x)
        attacker = Attacker(attacker_policy, nb_states, nb_actions, min_action, max_action, min_state, max_state, args)

    if args.mode == 'train':
        train(args,attacker)
    elif args.mode == 'test':
        test(args)