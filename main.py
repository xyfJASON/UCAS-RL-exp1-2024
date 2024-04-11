import os
import math
import yaml
import argparse
import datetime
from tqdm import tqdm

import numpy as np

from agent import Agent
from env import Environment
from action import DiscreteActionSpace
from state import State, DiscreteStateSpace
from utils import plot, animate_pendulum


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.1, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.98, help='discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1, help='epsilon-greedy policy')
    parser.add_argument('--episodes', type=int, default=1000, help='number of episodes')
    parser.add_argument('--episode_steps', type=int, default=1000, help='number of steps for each episode')
    parser.add_argument('--num_disc_alpha', type=int, default=20, help='discretization of alpha')
    parser.add_argument('--num_disc_alpha_dot', type=int, default=20, help='discretization of alpha_dot')
    parser.add_argument('--num_actions', type=int, default=3, help='number of actions')
    return parser


def main():
    # Arguments
    args = get_parser().parse_args()
    logdir = f"./logs/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    os.makedirs(logdir, exist_ok=True)
    with open(f'{logdir}/args.yaml', 'w') as f:
        yaml.dump(args.__dict__, f, default_flow_style=False)

    # Initialize
    env = Environment()
    agent = Agent(
        epsilon=args.epsilon,
        state_space=DiscreteStateSpace(
            num_disc_alpha=args.num_disc_alpha,
            num_disc_alpha_dot=args.num_disc_alpha_dot,
        ),
        action_space=DiscreteActionSpace(num_actions=args.num_actions),
    )

    # Train
    for _ in tqdm(range(args.episodes)):
        state = State(alpha=-math.pi, alpha_dot=0)
        for _ in range(args.episode_steps):
            # agent: s_t => a_t
            action = agent.apply_epsilon_greedy_policy(state)
            # env: s_t, a_t => r_{t+1}, s_{t+1}
            next_state, reward = env.update(state, action)
            # q-learning: update Q(s_t, a_t)
            state_idx = agent.state_to_idx(state)
            action_idx = agent.action_to_idx(action)
            next_state_idx = agent.state_to_idx(next_state)
            next_action_idx = np.argmax(agent.Q[next_state_idx])
            agent.Q[state_idx, action_idx] += args.alpha * (
                reward + args.gamma * agent.Q[next_state_idx, next_action_idx] -
                agent.Q[state_idx, action_idx]
            )
            # next state: s_{t+1}
            state = next_state

    # Save Q-table
    agent.saveq(os.path.join(logdir, 'q_table.npy'))

    # Test
    state = State(alpha=-math.pi, alpha_dot=0)
    states, actions, rewards = [], [], []
    for _ in range(1000):
        action = agent.apply_greedy_policy(state)
        state, reward = env.update(state, action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
    plot(states, actions, rewards, os.path.join(logdir, 'test.png'))
    animate_pendulum(states, actions, os.path.join(logdir, 'test.mp4'))


if __name__ == '__main__':
    main()
