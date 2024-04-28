import torch
import numpy as np
from attacker.Attacker import Attacker
import argparse
from PPO import PPO
from attacker.Attacker import Attacker
from env.ControlSlide import ControlSlideEnv
from env.CarFindFlag import CarFindFlagEnv
from env.CarFindFlag_e import CarFindFlagEEnv
from env.CarFindFlag_m import CarFindFlagMEnv
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

#################################### Testing ###################################
def test(args, attacked_path):
    env_name = args.env_name
    has_continuous_action_space = True
    max_ep_len = args.max_episode_length           # max timesteps in one episode
    action_std = 0.00001            # set same std for action distribution which was used while saving

    total_test_episodes = 10000    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################

    if env_name == "ControlSlideEnv":
        env = ControlSlideEnv()
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
    a_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    t_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory
    a_agent.eval()
    t_agent.eval()
    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)
    target_path = args.target_model + args.env_name + "/target_model.pth"

    t_agent.load(target_path)
    a_agent.load(attacked_path)

    max_action = env.action_space.high
    min_action = env.action_space.low
    ra = getra_(args.ra_piece,min_action,max_action)
    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0
    total_step = 0
    sim_step = 0
    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()

        for t in range(1, max_ep_len+1):
            t_action = t_agent.select_action(state,Train=False)
            a_action = a_agent.select_action(state, Train=False)
            state, reward, done, _ = env.step(t_action)
            ep_reward += reward
            if getDistance(t_action, a_action) < ra:
                sim_step += 1
            total_step += 1
            if done:
                break

        test_running_reward +=  ep_reward

    env.close()
    print('similarity: {}'.format(sim_step / total_step))
    print("============================================================================================")



parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')
# ControlSlideEnv:
parser.add_argument('--mode', default='test', type=str, help='support option: train/test')

parser.add_argument('--env_name', default='ControlSlideEnv', type=str, help='environment(ControlSlideEnv CarFindFlagEnv)')
parser.add_argument('--lr_actor', default=0.0003, type=float)
parser.add_argument('--lr_critic', default=0.001, type=float)
parser.add_argument('--random_seed', default=1, type=int)
parser.add_argument('--action_std', default=0.6, type=float,help='starting std for action distribution (Multivariate Normal)')
parser.add_argument('--action_std_decay_rate', default=0.05, type=float,help='linearly decay action_std (action_std = action_std - action_std_decay_rate)')
parser.add_argument('--min_action_std', default=0.1, type=float,help='minimum action_std (stop decay after action_std <= min_action_std)')
parser.add_argument('--action_std_decay_freq', default=5000, type=int,help='action_std decay frequency (in num timesteps)')
parser.add_argument('--K_epochs', default=80, type=int,help='update policy for K epochs in one PPO update')
parser.add_argument('--max_episode_length', default=10, type=int, help='')
parser.add_argument('--max_training_epochs', default=5000000, type=int, help='')
parser.add_argument('--target_model', default="./TargetModel/")
parser.add_argument('--attacked_model', default="./AttackedModel/")
parser.add_argument('--attack_method', default="black", help='white or black')
parser.add_argument('--ra_piece', default=32, type=int)
args = parser.parse_args()
for i in [2, 3, 4]:
    print(i)
    attacked_path = args.attacked_model + args.env_name + "/" + args.attack_method + "/" + str(i) + "_model.pth"
    if args.mode == 'test':
        test(args, attacked_path)
    else:
        print('error')