import numpy as np
from typing import Dict, List, Tuple, Any
from sgw.renderers.rend_interface import RendererInterface
from gym import spaces


class GridSymbolicRenderer(RendererInterface):
    # Object types and their corresponding channel indices
    OBJECT_CHANNELS = {
        "agent": 0,  # Current agent
        "other_agents": 1,  # Other agents
        "rewards": 2,
        "keys": 3,
        "doors": 4,
        "walls": 5,
        "warps": 6,
        "trees": 7,
        "fruits": 8,
    }

    def __init__(self, grid_shape: Tuple[int, int], window_size: int = None):
        self.grid_shape = grid_shape
        self.window_size = window_size
        self.num_channels = len(self.OBJECT_CHANNELS)

    @property
    def observation_space(self) -> spaces.Space:
        """Return the observation space for symbolic observations."""
        if self.window_size is None:
            shape = (self.grid_shape[0], self.grid_shape[1], self.num_channels)
        else:
            window_dim = 2 * self.window_size + 1
            shape = (window_dim, window_dim, self.num_channels)
        return spaces.Box(0, 1, shape=shape)

    def render_full(self, env: Any, agent_idx: int = 0) -> np.ndarray:
        """
        Returns a symbolic representation of the environment in a numpy tensor.
        The tensor shape is dynamic based on the number of object types.
        Each channel corresponds to a different object type as defined in OBJECT_CHANNELS.
        """
        grid = np.zeros([self.grid_shape[0], self.grid_shape[1], self.num_channels])

        # Set agents' positions
        for i, agent in enumerate(env.agents):
            if i == agent_idx:
                grid[agent.pos[0], agent.pos[1], self.OBJECT_CHANNELS["agent"]] = (
                    agent.looking + 1
                )  # Store orientation (+1 to avoid 0)
            else:
                grid[
                    agent.pos[0], agent.pos[1], self.OBJECT_CHANNELS["other_agents"]
                ] = (
                    i + 1
                )  # Store agent index (+1 to avoid 0)

        # Render all object types from the environment
        for obj_type, objects in env.objects.items():
            if obj_type in self.OBJECT_CHANNELS:
                channel_idx = self.OBJECT_CHANNELS[obj_type]
                for obj in objects:
                    if obj_type == "rewards":
                        value = obj.value
                        if isinstance(value, list):
                            if value[1] == 1:  # Only include if active
                                grid[obj.pos[0], obj.pos[1], channel_idx] = value[0]
                        else:
                            grid[obj.pos[0], obj.pos[1], channel_idx] = value
                    else:
                        grid[obj.pos[0], obj.pos[1], channel_idx] = 1

        return grid

    def render_walls(self, walls: List[List[int]]) -> np.ndarray:
        """Returns a numpy array of the walls in the environment."""
        grid = np.zeros([self.grid_shape[0], self.grid_shape[1]])
        for block in walls:
            grid[block[0], block[1]] = 1
        return grid

    def render_window(self, env: Any, size: int, agent_idx: int = 0) -> np.ndarray:
        """
        Returns a windowed symbolic observation centered on the agent.
        Window size is determined by vision range: 2 * range + 1
        """
        obs = self.render_full(env, agent_idx)
        window_dim = 2 * size + 1
        pad_size = size

        # Pad the observation with walls
        padded = np.zeros(
            (obs.shape[0] + 2 * pad_size, obs.shape[1] + 2 * pad_size, obs.shape[2])
        )
        padded[pad_size:-pad_size, pad_size:-pad_size] = obs
        wall_channel = self.OBJECT_CHANNELS["walls"]
        padded[:, :, wall_channel] = np.where(
            padded[:, :, wall_channel] == 0, 1, padded[:, :, wall_channel]
        )

        # Extract window centered on agent
        x, y = env.agents[agent_idx].pos
        window = padded[x : x + window_dim, y : y + window_dim, :]
        return window

    def render(self, env: Any, agent_idx: int = 0, **kwargs) -> np.ndarray:
        """
        Renders a symbolic observation from the environment.
        """
        if self.window_size is not None:
            return self.render_window(env, self.window_size, agent_idx)
        return self.render_full(env, agent_idx)

    @classmethod
    def add_object_type(cls, object_type: str) -> None:
        """
        Adds a new object type to the renderer.
        The new object type will be assigned the next available channel index.

        Args:
            object_type (str): The name of the new object type to add.
        """
        if object_type not in cls.OBJECT_CHANNELS:
            next_channel = len(cls.OBJECT_CHANNELS)
            cls.OBJECT_CHANNELS[object_type] = next_channel
