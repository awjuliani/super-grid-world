from abc import ABC, abstractmethod


class RendererInterface(ABC):
    @abstractmethod
    def render(self, env, **kwargs):
        """Render an observation from the environment using the renderer."""
        pass
