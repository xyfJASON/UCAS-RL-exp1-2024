import numpy as np

from action import Action, DiscreteActionSpace
from state import State, DiscreteStateSpace


class Agent:
    def __init__(
            self,
            epsilon: float,
            state_space: DiscreteStateSpace,
            action_space: DiscreteActionSpace,
    ):
        self.epsilon = epsilon
        self.state_space = state_space
        self.action_space = action_space

        # action-value function Q(s,a)
        self.Q = np.zeros((self.state_space.num_states, self.action_space.num_actions))

    def saveq(self, filename):
        np.save(filename, self.Q)

    def loadq(self, filename):
        self.Q = np.load(filename)

    def apply_epsilon_greedy_policy(self, state: State):
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.action_space.num_actions)
        else:
            state_idx = self.state_to_idx(state)
            action_idx = np.argmax(self.Q[state_idx])
        return self.idx_to_action(action_idx)

    def apply_greedy_policy(self, state: State):
        state_idx = self.state_to_idx(state)
        action_idx = np.argmax(self.Q[state_idx])
        return self.idx_to_action(action_idx)

    def state_to_idx(self, state: State):
        alpha_idx = np.argmin(np.abs(self.state_space.alpha_table - state.alpha))
        alpha_dot_idx = np.argmin(np.abs(self.state_space.alpha_dot_table - state.alpha_dot))
        return self.state_space.get_state_idx(alpha_idx, alpha_dot_idx)

    def action_to_idx(self, action: Action):
        u_idx = np.argmin(np.abs(self.action_space.u_table - action.u))
        return u_idx

    def idx_to_action(self, action_idx: int):
        u = self.action_space.u_table[action_idx].item()
        return Action(u)
