from abc import ABC, abstractmethod
from gym import spaces


class RendererInterface(ABC):
    @abstractmethod
    def render(self, env, **kwargs):
        """Render an observation from the environment using the renderer."""
        pass

    @property
    @abstractmethod
    def observation_space(self) -> spaces.Space:
        """Return the observation space for this renderer."""
        pass

    def close(self):
        """Close the renderer."""
        pass
