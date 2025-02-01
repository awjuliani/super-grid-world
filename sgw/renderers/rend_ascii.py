import numpy as np
from typing import Dict, List, Any
from sgw.renderers.rend_interface import RendererInterface
from gym import spaces


class GridASCIIRenderer(RendererInterface):
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.ascii_map = {
            0: " ",  # empty
            1: "A",  # agent
            2: "B",  # block/wall
            3: "R",  # reward
            4: "L",  # lava/negative reward
            5: "K",  # key
            6: "D",  # door
            7: "W",  # warp
            8: "O",  # other
        }

    @property
    def observation_space(self) -> spaces.Space:
        """Return the observation space for ASCII observations."""
        return spaces.Discrete(1)

    def make_ascii_obs(self, env: Any) -> str:
        """
        Returns an ASCII string representation of the environment.
        Legend:
        _ = empty
        A = agent
        B = block/wall
        R = reward
        L = lava/negative reward
        K = key
        D = door
        W = warp
        O = other
        """
        grid = np.zeros((self.grid_size, self.grid_size))

        # Set agent position
        grid[env.agent_pos[0], env.agent_pos[1]] = 1

        # Set walls
        for block in env.objects["walls"]:
            grid[block[0], block[1]] = 2

        # Set rewards
        for reward_pos, reward_val in env.objects["rewards"].items():
            if isinstance(reward_val, list):
                reward_val = reward_val[0]
            if reward_val > 0:
                grid[reward_pos[0], reward_pos[1]] = 3
            else:
                grid[reward_pos[0], reward_pos[1]] = 4

        # Set keys
        for key_pos in env.objects["keys"]:
            grid[key_pos[0], key_pos[1]] = 5

        # Set doors
        for door_pos in env.objects["doors"]:
            grid[door_pos[0], door_pos[1]] = 6

        # Set warps
        for warp_pos in env.objects["warps"]:
            grid[warp_pos[0], warp_pos[1]] = 7

        # Set other objects
        for other_pos, other_name in env.objects["other"].items():
            grid[other_pos[0], other_pos[1]] = 8

        # Convert numerical grid to ASCII characters
        ascii_grid = np.vectorize(self.ascii_map.get)(grid)

        # Join with newlines to create final string
        return "\n".join(["".join(row) for row in ascii_grid])

    def render(self, env: Any, **kwargs) -> str:
        # Render method for ASCII observation.
        return self.make_ascii_obs(env)
