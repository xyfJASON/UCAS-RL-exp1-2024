import tqdm
import numpy as np
from typing import Any

from envs.base import BaseEnv, BaseQuantizer
from learners.base import BaseLearner


class QLearning(BaseLearner):
    def __init__(self, env: BaseEnv, state_quantizer: BaseQuantizer, action_quantizer: BaseQuantizer):
        self.env = env
        self.state_quantizer = state_quantizer
        self.action_quantizer = action_quantizer

        # Q table: Q(s,a)
        self.num_states = self.state_quantizer.size
        self.num_actions = self.action_quantizer.size
        self.Q = np.zeros((self.num_states, self.num_actions))

    def greedy_policy(self, state: Any) -> Any:
        state_idx = self.state_quantizer.element_to_idx(state)
        action_idx = np.argmax(self.Q[state_idx])
        action = self.action_quantizer.idx_to_element(action_idx)
        return action

    def epsilon_greedy_policy(self, state: Any, epsilon: float) -> Any:
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(self.num_actions)
        else:
            state_idx = self.state_quantizer.element_to_idx(state)
            action_idx = np.argmax(self.Q[state_idx])
        action = self.action_quantizer.idx_to_element(action_idx)
        return action

    def train(
            self,
            episodes: int,
            episode_length: int,
            learning_rate: float = 0.1,
            epsilon: float = 0.1,
            discount_factor: float = 0.98,
            with_tqdm: bool = True,
    ):
        for _ in tqdm.trange(episodes, disable=not with_tqdm):
            self.env.reset()
            for _ in range(episode_length):
                if self.env.is_terminated:
                    break
                state = self.env.get_state()
                action = self.epsilon_greedy_policy(state, epsilon=epsilon)
                new_state, reward = self.env.update(action)
                self.update_Q(state, action, new_state, reward, learning_rate, discount_factor)

    def update_Q(
            self, state: Any, action: Any, new_state: Any, reward: float,
            learning_rate: float, discount_factor: float,
    ):
        state_idx = self.state_quantizer.element_to_idx(state)
        action_idx = self.action_quantizer.element_to_idx(action)
        new_state_idx = self.state_quantizer.element_to_idx(new_state)
        self.Q[state_idx, action_idx] += learning_rate * (
            reward + discount_factor * np.max(self.Q[new_state_idx]) - self.Q[state_idx, action_idx]
        )

    def test(self, episode_length: int = 1000):
        self.env.reset()
        states, actions, rewards = [], [], []
        for _ in range(episode_length):
            state = self.env.get_state()
            action = self.greedy_policy(state)
            state, reward = self.env.update(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
        return states, actions, rewards
