import numpy as np


class Action:
    def __init__(self, u: float):
        self.u = u
        if not -3. <= self.u <= 3.:
            raise ValueError('u should be in range [-3, 3]')


class DiscreteActionSpace:
    def __init__(self, num_actions: int = 3):
        self.num_actions = num_actions
        self.u_table = np.linspace(-3, 3, num_actions)
