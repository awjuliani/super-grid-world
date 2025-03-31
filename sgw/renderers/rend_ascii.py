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
        "linked_doors": "=",
        "linked_doors_open": "_",
        "pressure_plates": "P",
        "levers": "L",
        "levers_inactive": "l",
        "levers_active": "L",
        "warps": "W",
        "other": "O",
        "trees": "T",
        "fruits": "F",
        "signs": "S",
        "boxes": "X",
        "pushable_boxes": "C",
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
        """
        # Initialize grid with empty value
        grid = np.full(
            (self.grid_shape[0], self.grid_shape[1]),
            self.GRID_VALUES["empty"],
            dtype=int,
        )

        # Render all object types from the environment
        # Process objects in a defined order (e.g., static first, then dynamic)
        render_order = [
            "walls",
            "pressure_plates",  # Floor items
            "rewards",
            "keys",
            "doors",
            "linked_doors",
            "warps",
            "other",
            "trees",
            "fruits",
            "signs",
            "boxes",
            "pushable_boxes",
            "levers",  # Items on top
        ]

        for obj_type_key in render_order:
            if obj_type_key in env.objects:
                objects = env.objects[obj_type_key]
                if not objects:
                    continue  # Skip if list is empty

                if obj_type_key == "rewards":
                    for reward in objects:
                        value = (
                            reward.value[0]
                            if isinstance(reward.value, list)
                            else reward.value
                        )
                        grid_type = (
                            "rewards_positive" if value > 0 else "rewards_negative"
                        )
                        if grid_type in self.GRID_VALUES:
                            grid[reward.pos[0], reward.pos[1]] = self.GRID_VALUES[
                                grid_type
                            ]
                elif obj_type_key == "linked_doors":
                    for door in objects:
                        grid_type = (
                            "linked_doors_open" if door.is_open else "linked_doors"
                        )
                        if grid_type in self.GRID_VALUES:
                            grid[door.pos[0], door.pos[1]] = self.GRID_VALUES[grid_type]
                elif obj_type_key == "levers":
                    for lever in objects:
                        grid_type = (
                            "levers_active" if lever.activated else "levers_inactive"
                        )
                        if grid_type in self.GRID_VALUES:
                            grid[lever.pos[0], lever.pos[1]] = self.GRID_VALUES[
                                grid_type
                            ]
                elif obj_type_key in self.GRID_VALUES:  # Handle standard types
                    for obj in objects:
                        # Check position bounds
                        if (
                            0 <= obj.pos[0] < self.grid_shape[0]
                            and 0 <= obj.pos[1] < self.grid_shape[1]
                        ):
                            grid[obj.pos[0], obj.pos[1]] = self.GRID_VALUES[
                                obj_type_key
                            ]
                elif obj_type_key == "pressure_plates":  # Use specific key if needed
                    grid_type = "pressure_plates"
                    for plate in objects:
                        if grid_type in self.GRID_VALUES:
                            grid[plate.pos[0], plate.pos[1]] = self.GRID_VALUES[
                                grid_type
                            ]
                elif obj_type_key == "pushable_boxes":  # Use specific key
                    grid_type = "pushable_boxes"
                    for box in objects:
                        if grid_type in self.GRID_VALUES:
                            grid[box.pos[0], box.pos[1]] = self.GRID_VALUES[grid_type]
                # Add other specific handlers if needed

        # Set agents' positions last so they appear on top
        for i, agent in enumerate(env.agents):
            grid_type = "agent" if i == agent_idx else "other_agents"
            grid[agent.pos[0], agent.pos[1]] = self.GRID_VALUES[grid_type]

        # Convert grid to ASCII string using the updated map
        # Create inverse map for quick lookup
        value_to_char = {v: k for k, v in self.GRID_VALUES.items()}
        ascii_rows = []
        for row in grid:
            char_row = []
            for cell_value in row:
                # Find the key (e.g., "walls") corresponding to the cell_value
                type_key = None
                for k, v in self.GRID_VALUES.items():
                    if v == cell_value:
                        type_key = k
                        break
                # Get the ASCII char using the key
                char_row.append(self.ASCII_MAP.get(type_key, "?"))  # Default to '?'
            ascii_rows.append("".join(char_row))

        return "\n".join(ascii_rows)

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
