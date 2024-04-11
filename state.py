import math
import numpy as np


class State:
    def __init__(self, alpha: float, alpha_dot: float):
        self.alpha = alpha
        self.alpha_dot = alpha_dot
        if not -math.pi <= self.alpha < math.pi:
            raise ValueError('alpha should be in range [-pi, pi)')
        if not -15 * math.pi <= self.alpha_dot <= 15 * math.pi:
            raise ValueError('alpha_dot should be in range [-15pi, 15pi]')


class DiscreteStateSpace:
    def __init__(self, num_disc_alpha: int = 100, num_disc_alpha_dot: int = 100):
        self.alpha_table = np.arange(-np.pi, np.pi, 2 * np.pi / num_disc_alpha)
        self.alpha_dot_table = np.arange(-15 * np.pi, 15 * np.pi, 30 * np.pi / num_disc_alpha_dot)
        self.num_states = len(self.alpha_table) * len(self.alpha_dot_table)

    def get_state_idx(self, alpha_idx: int, alpha_dot_idx: int):
        return alpha_idx * len(self.alpha_dot_table) + alpha_dot_idx
