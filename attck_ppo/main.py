from train import train
from test import test
from attacker.Attacker import Attacker
import argparse
from PPO import PPO
from attacker.Attacker import Attacker
from env.ControlSlide import ControlSlideEnv
parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')
# ControlSlideEnv:
parser.add_argument('--mode', default='train', type=str, help='support option: train/test')

parser.add_argument('--env_name', default='ControlSlideEnv', type=str, help='environment(ControlSlideEnv CarFindFlagEnv)')
parser.add_argument('--lr_actor', default=0.0003, type=float)
parser.add_argument('--lr_critic', default=0.001, type=float)
parser.add_argument('--random_seed', default=0, type=int)
parser.add_argument('--action_std', default=0.4, type=float,help='starting std for action distribution (Multivariate Normal)')
parser.add_argument('--action_std_decay_rate', default=0.04, type=float,help='linearly decay action_std (action_std = action_std - action_std_decay_rate)')
parser.add_argument('--min_action_std', default=0.01, type=float,help='minimum action_std (stop decay after action_std <= min_action_std)')
parser.add_argument('--action_std_decay_freq', default=2500, type=int,help='action_std decay frequency (in num timesteps)')
parser.add_argument('--K_epochs', default=80, type=int,help='update policy for K epochs in one PPO update')
parser.add_argument('--max_episode_length', default=10, type=int, help='')
parser.add_argument('--max_training_epochs', default=5000000, type=int, help='')

# attack
parser.add_argument('--ATTACK', default=True, type=bool, help='Attack or not')
parser.add_argument('--attack_method', default="black", help='white or black')
parser.add_argument('--ls', default=1.0, type=float)
parser.add_argument('--p', default=0.5, type=float)
parser.add_argument('--rs_piece', default=32, type=int, help="The number of shards per dimension in the state")
parser.add_argument('--ra_piece', default=64, type=int)
parser.add_argument('--attack_target_model', default="./TargetModel/")
parser.add_argument('--delta', default=0.05, type=float)
parser.add_argument('--isWeak', default=False)
parser.add_argument('--multiples_of_v', default=4, type=int)
parser.add_argument('--lrs', default=1, type=int)
args = parser.parse_args()
attacker = None
if args.env_name == "ControlSlideEnv":
    print(args.env_name)
    env = ControlSlideEnv()
    print(env.bound)


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
if args.ATTACK:
    targetAgent = PPO(nb_states, nb_actions, 0.0, 0.0, 0.0, 0, 0.0, True, 0.001)
    attack_target_model = args.attack_target_model + args.env_name + "/target7_model.pth"
    print(attack_target_model)
    targetAgent.load(attack_target_model)
    targetAgent.eval()
    # targetAgent, s_dim, a_dim, min_a, max_a, min_s, max_s,args
    attacker_policy = lambda x: targetAgent.select_action(x, Train=False)
    attacker = Attacker(attacker_policy, nb_states, nb_actions, min_action, max_action, min_state, max_state, args)
if args.mode == 'train':
    train(args,attacker)
elif args.mode == 'test':
    test(args)
else:
    print('error')
