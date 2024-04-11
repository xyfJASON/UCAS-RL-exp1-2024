import math

import const
from action import Action
from state import State


class Environment:

    def update(self, state: State, action: Action):
        new_alpha = state.alpha + const.Ts * state.alpha_dot
        new_alpha_dot = (
            state.alpha_dot + const.Ts *
            self._dynamic_fn(state.alpha, state.alpha_dot, action.u)
        )
        while new_alpha >= math.pi:
            new_alpha -= 2 * math.pi
        while new_alpha < -math.pi:
            new_alpha += 2 * math.pi
        if new_alpha_dot >= 15 * math.pi:
            new_alpha_dot = 15 * math.pi
        if new_alpha_dot <= -15 * math.pi:
            new_alpha_dot = -15 * math.pi
        new_state = State(new_alpha, new_alpha_dot)

        reward = -(
            5 * state.alpha * state.alpha +
            0.1 * state.alpha_dot * state.alpha_dot +
            action.u * action.u
        )
        return new_state, reward

    @staticmethod
    def _dynamic_fn(alpha: float, alpha_dot: float, u: float):
        return 1 / const.J * (
            const.m * const.g * const.l * math.sin(alpha) -
            const.b * alpha_dot - 
            const.K * const.K / const.R * alpha_dot +
            const.K / const.R * u
        )
