import numpy as np
from typing import Any, Tuple
from sgw.renderers.rend_interface import RendererInterface
from gym import spaces


class GridASCIIRenderer(RendererInterface):
    # Object types and their corresponding ASCII characters
    ASCII_MAP = {
        "empty": " ",
        "agent": "A",
        "other_agents": "a",
        "walls": "B",
        "rewards_positive": "R",
        "rewards_negative": "L",
        "keys": "K",
        "doors": "D",
        "warps": "W",
        "other": "O",
        "trees": "T",
        "fruits": "F",
        "signs": "S",
        "boxes": "X",
        "pushable_boxes": "P",
    }

    # Object types and their corresponding grid values
    GRID_VALUES = {obj_type: i for i, obj_type in enumerate(ASCII_MAP.keys())}

    def __init__(self, grid_shape: Tuple[int, int]):
        self.grid_shape = grid_shape

    @property
    def observation_space(self) -> spaces.Space:
        """Return the observation space for ASCII observations."""
        return spaces.Discrete(1)

    def make_ascii_obs(self, env: Any, agent_idx: int = 0) -> str:
        """
        Returns an ASCII string representation of the environment.
        Each object type is represented by a unique ASCII character as defined in ASCII_MAP.
        """
        grid = np.zeros((self.grid_shape[1], self.grid_shape[0]), dtype=int)

        # Set agents' positions
        for i, agent in enumerate(env.agents):
            grid[agent.pos[0], agent.pos[1]] = (
                self.GRID_VALUES["agent"]
                if i == agent_idx
                else self.GRID_VALUES["other_agents"]
            )

        # Render all object types from the environment
        for obj_type, objects in env.objects.items():
            if obj_type == "rewards":
                # Special handling for rewards to distinguish positive/negative
                for reward in objects:
                    value = reward.value
                    if isinstance(value, list):
                        value = value[0]
                    grid_type = "rewards_positive" if value > 0 else "rewards_negative"
                    if grid_type in self.GRID_VALUES:
                        grid[reward.pos[0], reward.pos[1]] = self.GRID_VALUES[grid_type]
            elif obj_type in self.GRID_VALUES:
                # Standard handling for other object types
                for obj in objects:
                    grid[obj.pos[0], obj.pos[1]] = self.GRID_VALUES[obj_type]

        # Convert grid to ASCII string
        return "\n".join(
            "".join(
                self.ASCII_MAP.get(
                    [k for k, v in self.GRID_VALUES.items() if v == int(cell)][0], "?"
                )
                for cell in row
            )
            for row in grid
        )

    def render(self, env: Any, agent_idx: int = 0, **kwargs) -> str:
        """Render the environment as an ASCII string."""
        return self.make_ascii_obs(env, agent_idx)

    @classmethod
    def add_object_type(cls, object_type: str, ascii_char: str) -> None:
        """
        Adds a new object type to the renderer with its ASCII representation.

        Args:
            object_type (str): The name of the new object type to add.
            ascii_char (str): Single character to represent this object type.
        """
        if object_type not in cls.ASCII_MAP:
            cls.ASCII_MAP[object_type] = ascii_char
            cls.GRID_VALUES = {
                obj_type: i for i, obj_type in enumerate(cls.ASCII_MAP.keys())
            }
