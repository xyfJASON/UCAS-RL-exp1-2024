import tqdm
import numpy as np

from envs.base import BaseEnv
from learners.base import BaseLearner


class QLearning(BaseLearner):
    def __init__(self, env: BaseEnv):
        self.env = env

        # Q table: Q(s,a)
        self.num_states = getattr(self.env, 'num_states')
        self.num_actions = getattr(self.env, 'num_actions')
        self.Q = np.zeros((self.num_states, self.num_actions))

    def greedy_policy(self, state: int) -> int:
        return np.argmax(self.Q[state])

    def epsilon_greedy_policy(self, state: int, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)
        return self.greedy_policy(state)

    def train(
            self,
            episodes: int,
            episode_length: int,
            learning_rate: float = 0.1,
            epsilon: float = 0.1,
            discount_factor: float = 0.98,
            with_tqdm: bool = True,
    ):
        rewards = []
        for _ in tqdm.trange(episodes, disable=not with_tqdm):
            self.env.reset()
            for _ in range(episode_length):
                if self.env.is_terminated:
                    break
                state = self.env.get_state()
                action = self.epsilon_greedy_policy(state, epsilon=epsilon)
                new_state, reward = self.env.update(action)
                self.Q[state, action] += learning_rate * (
                    reward + discount_factor * np.max(self.Q[new_state]) - self.Q[state, action]
                )
                rewards.append(reward)
        return rewards

    def test(self, episode_length: int = 1000):
        self.env.reset()
        states, actions, rewards = [], [], []
        for _ in range(episode_length):
            self.env.update(self.greedy_policy(self.env.get_state()))
            state, action, reward = self.env.get_info()
            states.append(state)
            actions.append(action)
            rewards.append(reward)
        return states, actions, rewards
