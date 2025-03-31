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
        "linked_doors": 5,
        "pressure_plates": 6,
        "levers": 7,
        "walls": 8,
        "warps": 9,
        "trees": 10,
        "fruits": 11,
        "signs": 12,
        "boxes": 13,
        "pushable_boxes": 14,
    }

    def __init__(self, grid_shape: Tuple[int, int], window_size: int = None):
        self.grid_shape = grid_shape
        self.window_size = window_size
        self.num_channels = len(self.OBJECT_CHANNELS)
        if set(self.OBJECT_CHANNELS.values()) != set(range(self.num_channels)):
            raise ValueError("OBJECT_CHANNELS indices are not contiguous!")

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
        for obj_type_key, objects in env.objects.items():
            channel_key = obj_type_key
            if channel_key in self.OBJECT_CHANNELS:
                channel_idx = self.OBJECT_CHANNELS[channel_key]
                for obj in objects:
                    if (
                        0 <= obj.pos[0] < self.grid_shape[0]
                        and 0 <= obj.pos[1] < self.grid_shape[1]
                    ):
                        value_to_set = 1

                        if channel_key == "rewards":
                            value = (
                                obj.value[0]
                                if isinstance(obj.value, list)
                                else obj.value
                            )
                            value_to_set = value
                        elif channel_key == "linked_doors":
                            value_to_set = 2 if obj.is_open else 1
                        elif channel_key == "levers":
                            value_to_set = 2 if obj.activated else 1
                        elif channel_key == "pressure_plates":
                            pass

                        grid[obj.pos[0], obj.pos[1], channel_idx] = value_to_set

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
            cls.num_channels = len(cls.OBJECT_CHANNELS)
            if set(cls.OBJECT_CHANNELS.values()) != set(range(cls.num_channels)):
                raise ValueError(
                    "OBJECT_CHANNELS indices became non-contiguous after adding!"
                )
