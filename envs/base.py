from abc import ABC, abstractmethod
from typing import Any, Tuple


class BaseEnv(ABC):

    @abstractmethod
    def get_state(self) -> Any:
        """Get the current state of the environment.

        Returns:
            state: The current state of the environment.

        """
        pass

    @abstractmethod
    def get_info(self) -> Tuple[Any, Any, float]:
        """Get the information of the environment.

        Returns:
            state: The current state of the environment.
            action: The most recent action taken by the agent.
            reward: The reward of the action.

        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the environment."""
        pass

    @abstractmethod
    def update(self, action: Any) -> Tuple[Any, float]:
        """Update the environment based on the action.

        Args:
            action: The action taken by the agent.

        Returns:
            new_state: The new state of the environment.
            reward: The reward of the action.

        """
        pass

    @property
    @abstractmethod
    def is_terminated(self) -> bool:
        """Check if the environment is terminated.

        Returns:
            is_terminated: True if the environment is terminated, False otherwise.

        """
        pass
