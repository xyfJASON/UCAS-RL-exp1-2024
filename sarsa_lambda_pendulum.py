import os
import yaml
import argparse
import datetime

import numpy as np

from learners.sarsa_lambda import SarsaLambda
from envs.pendulum import PendulumEnv, StateQuantizer, ActionQuantizer


def get_parser():
    parser = argparse.ArgumentParser()
    # system
    parser.add_argument('--logdir', type=str, help='log directory')
    parser.add_argument('--no-test-gif', action='store_true', help='do not save test gif')
    # hyper-parameters
    parser.add_argument('--alpha', type=float, default=0.1, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.98, help='discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1, help='epsilon-greedy policy')
    parser.add_argument('--lambd', type=float, default=0.9, help='lambda for sarsa(lambda)')
    parser.add_argument('--episodes', type=int, default=100, help='number of episodes')
    parser.add_argument('--episode_length', type=int, default=10000, help='length of each episode')
    parser.add_argument('--num_disc_alpha', type=int, default=20, help='discretization of alpha')
    parser.add_argument('--num_disc_alpha_dot', type=int, default=20, help='discretization of alpha_dot')
    parser.add_argument('--num_u', type=int, default=3, help='discretization of u')
    return parser


def main():
    # Arguments
    args = get_parser().parse_args()
    if args.logdir is None:
        args.logdir = f"./logs/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    os.makedirs(args.logdir, exist_ok=True)
    with open(f'{args.logdir}/args.yaml', 'w') as f:
        args_dict = args.__dict__.copy()
        args_dict.pop('logdir')
        yaml.dump(args_dict, f, default_flow_style=False)

    # Initialize
    env = PendulumEnv()
    learner = SarsaLambda(
        env=env,
        state_quantizer=StateQuantizer(
            num_disc_alpha=args.num_disc_alpha,
            num_disc_alpha_dot=args.num_disc_alpha_dot,
        ),
        action_quantizer=ActionQuantizer(num_u=args.num_u),
        lambd=args.lambd,
    )

    # Train
    learner.train(
        episodes=args.episodes,
        episode_length=args.episode_length,
        epsilon=args.epsilon,
        learning_rate=args.alpha,
        discount_factor=args.gamma,
    )

    # Save Q-table
    np.save(os.path.join(args.logdir, 'q_table.npy'), learner.Q)

    # Test
    states, actions, rewards = learner.test(episode_length=1000)
    np.save(os.path.join(args.logdir, 'test_rewards.npy'), rewards)
    env.plot_curve(states, actions, rewards, os.path.join(args.logdir, 'test.png'))
    if not args.no_test_gif:
        env.animate(states, actions, os.path.join(args.logdir, 'test.gif'))


if __name__ == '__main__':
    main()
