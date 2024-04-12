from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

from envs.base import BaseEnv


@dataclass
class State:
    alpha: float
    alpha_dot: float

    def __post_init__(self):
        if not -np.pi <= self.alpha < np.pi:
            raise ValueError(f'alpha must be in [-π, π). Got {self.alpha}.')
        if not -15 * np.pi <= self.alpha_dot <= 15 * np.pi:
            raise ValueError(f'alpha_dot must be in [-15π, 15π]. Got {self.alpha_dot}.')


@dataclass
class Action:
    u: float

    def __post_init__(self):
        if not -3 <= self.u <= 3:
            raise ValueError(f'u must be in [-3, 3]. Got {self.u}.')


class StateQuantizer:
    def __init__(self, alpha_table: np.ndarray, alpha_dot_table: np.ndarray):
        self.alpha_table = alpha_table
        self.alpha_dot_table = alpha_dot_table

    def state_to_idx(self, state: State) -> int:
        alpha_idx = np.argmin(np.abs(self.alpha_table - state.alpha))
        alpha_dot_idx = np.argmin(np.abs(self.alpha_dot_table - state.alpha_dot))
        state_idx = alpha_idx * len(self.alpha_dot_table) + alpha_dot_idx
        return state_idx

    def idx_to_state(self, state_idx: int) -> State:
        alpha_idx, alpha_dot_idx = divmod(state_idx, len(self.alpha_dot_table))
        alpha = self.alpha_table[alpha_idx].item()
        alpha_dot = self.alpha_dot_table[alpha_dot_idx].item()
        return State(alpha, alpha_dot)


class ActionQuantizer:
    def __init__(self, u_table: np.ndarray):
        self.u_table = u_table

    def action_to_idx(self, action: Action) -> int:
        return np.argmin(np.abs(self.u_table - action.u))

    def idx_to_action(self, action_idx: int) -> Action:
        return Action(self.u_table[action_idx].item())


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

    def _dynamic_fn(self, alpha: float, alpha_dot: float, u: float):
        return 1 / self.J * (
            self.m * self.g * self.l * np.sin(alpha) -
            self.b * alpha_dot -
            self.K * self.K / self.R * alpha_dot +
            self.K / self.R * u
        )

    @staticmethod
    def plot_curve(states: List[State], actions: List[Action], rewards: List[float], save_path: str):
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 4, 1)
        plt.plot(rewards)
        plt.title('reward')

        plt.subplot(1, 4, 2)
        plt.plot([state.alpha for state in states])
        plt.ylim(-np.pi - 0.5, np.pi + 0.5)
        plt.title('alpha')

        plt.subplot(1, 4, 3)
        plt.plot([state.alpha_dot for state in states])
        plt.ylim(-15 * np.pi - 5, 15 * np.pi + 5)
        plt.title('alpha_dot')

        plt.subplot(1, 4, 4)
        plt.plot([action.u for action in actions])
        plt.ylim(-3.5, 3.5)
        plt.title('action')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def animate(states: List[State], actions: List[Action], save_path: str):
        xs = [np.sin(s.alpha) for s in states]
        ys = [np.cos(s.alpha) for s in states]

        fig = plt.figure(figsize=(4, 3), dpi=200)
        ax = fig.add_subplot(autoscale_on=False, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
        ax.set_aspect('equal')
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.grid()
        line, = ax.plot([], [], 'o-', lw=2)
        text = ax.text(0, -0.2, '')

        def draw(i):
            line.set_data([0, xs[i]], [0, ys[i]])
            text.set_text(str(actions[i].u))
            return line, text

        animator = ani.FuncAnimation(fig, draw, frames=len(xs), interval=5)
        animator.save(save_path, fps=50)
        plt.close()


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
            u_table=np.linspace(-3, 3, num_actions),
        )

        self.state_idx = self.state_quantizer.state_to_idx(self.state)

    def get_state(self) -> int:
        return self.state_idx

    def reset(self):
        super().reset()
        self.state_idx = self.state_quantizer.state_to_idx(self.state)

    def update(self, action_idx: int) -> Tuple[int, float]:
        action = self.action_quantizer.idx_to_action(action_idx)
        _, reward = super().update(action)
        self.state_idx = self.state_quantizer.state_to_idx(self.state)
        return self.state_idx, reward
