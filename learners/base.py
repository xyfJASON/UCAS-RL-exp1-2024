from abc import ABC, abstractmethod


class BaseLearner(ABC):

    @abstractmethod
    def train(self, num_episodes: int, episode_length: int):
        pass

    @abstractmethod
    def test(self, episode_length: int):
        pass
