from dataclasses import dataclass
from typing import Tuple

import numpy as np

from envs.base import BaseEnv


@dataclass
class State:
    alpha: float
    alpha_dot: float

    def __post_init__(self):
        if not (-np.pi <= self.alpha < np.pi):
            raise ValueError(f'alpha must be in [-π, π). Got {self.alpha}.')
        if not (-15 * np.pi <= self.alpha_dot <= 15 * np.pi):
            raise ValueError(f'alpha_dot must be in [-15π, 15π]. Got {self.alpha_dot}.')


@dataclass
class Action:
    u: float

    def __post_init__(self):
        if not (-3 <= self.u <= 3):
            raise ValueError(f'u must be in [-3, 3]. Got {self.u}.')


class StateQuantizer:
    def __init__(self, alpha_table: np.ndarray, alpha_dot_table: np.ndarray):
        self.alpha_table = alpha_table
        self.alpha_dot_table = alpha_dot_table

    def element_to_idx(self, state: State) -> int:
        alpha_idx = np.argmin(np.abs(self.alpha_table - state.alpha))
        alpha_dot_idx = np.argmin(np.abs(self.alpha_dot_table - state.alpha_dot))
        state_idx = alpha_idx * len(self.alpha_dot_table) + alpha_dot_idx
        return state_idx

    def idx_to_element(self, state_idx: int) -> State:
        alpha_idx, alpha_dot_idx = divmod(state_idx, len(self.alpha_dot_table))
        alpha = self.alpha_table[alpha_idx].item()
        alpha_dot = self.alpha_dot_table[alpha_dot_idx].item()
        return State(alpha, alpha_dot)


class ActionQuantizer:
    def __init__(self, action_table: np.ndarray):
        self.action_table = action_table

    def element_to_idx(self, action: Action) -> int:
        return np.argmin(np.abs(self.action_table - action.u))

    def idx_to_element(self, action_idx: int) -> Action:
        return Action(self.action_table[action_idx].item())


class PendulumEnv(BaseEnv):
    def __init__(self):
        self.m = 0.055
        self.g = 9.81
        self.l = 0.042
        self.J = 1.91e-4
        self.b = 3e-6
        self.K = 0.0536
        self.R = 9.5
        self.Ts = 0.005

        # info
        self.state = State(-np.pi, 0.0)
        self.action = None
        self.reward = None

    def get_state(self) -> State:
        return self.state

    def get_info(self) -> Tuple[State, Action, float]:
        return self.state, self.action, self.reward

    def reset(self):
        self.state = State(-np.pi, 0)
        self.action = None
        self.reward = None

    def update(self, action: Action) -> Tuple[State, float]:
        self.action = action

        new_alpha = self.state.alpha + self.Ts * self.state.alpha_dot
        new_alpha_dot = (
            self.state.alpha_dot + self.Ts *
            self._dynamic_fn(self.state.alpha, self.state.alpha_dot, action.u)
        )
        new_alpha = (new_alpha + np.pi) % (2 * np.pi) - np.pi
        new_alpha_dot = np.clip(new_alpha_dot, -15 * np.pi, 15 * np.pi)
        self.state = State(new_alpha, new_alpha_dot)

        self.reward = -(
            5 * self.state.alpha * self.state.alpha +
            0.1 * self.state.alpha_dot * self.state.alpha_dot +
            action.u * action.u
        )
        return self.state, self.reward

    @property
    def is_terminated(self) -> bool:
        return False

    def _dynamic_fn(self, alpha, alpha_dot, u):
        return 1 / self.J * (
            self.m * self.g * self.l * np.sin(alpha) -
            self.b * alpha_dot -
            self.K * self.K / self.R * alpha_dot +
            self.K / self.R * u
        )


class PendulumDiscObsEnv(PendulumEnv):
    """Pendulum environment with discrete observations."""
    def __init__(
            self,
            num_disc_alpha: int = 100,
            num_disc_alpha_dot: int = 100,
            num_actions: int = 3,
    ):
        super().__init__()

        self.num_states = num_disc_alpha * num_disc_alpha_dot
        self.num_actions = num_actions

        self.state_quantizer = StateQuantizer(
            alpha_table=np.arange(-np.pi, np.pi, 2 * np.pi / num_disc_alpha),
            alpha_dot_table=np.linspace(-15 * np.pi, 15 * np.pi, num_disc_alpha_dot),
        )
        self.action_quantizer = ActionQuantizer(
            action_table=np.linspace(-3, 3, num_actions),
        )

        self.state_idx = self.state_quantizer.element_to_idx(self.state)

    def get_state(self) -> int:
        return self.state_idx

    def reset(self):
        super().reset()
        self.state_idx = self.state_quantizer.element_to_idx(self.state)

    def update(self, action_idx: int) -> Tuple[int, float]:
        action = self.action_quantizer.idx_to_element(action_idx)
        _, reward = super().update(action)
        self.state_idx = self.state_quantizer.element_to_idx(self.state)
        return self.state_idx, reward
