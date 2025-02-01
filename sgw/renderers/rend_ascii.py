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
        grid[env.agent.pos[0], env.agent.pos[1]] = 1

        # Set walls
        for wall in env.objects["walls"]:
            grid[wall.pos[0], wall.pos[1]] = 2

        # Set rewards
        for reward in env.objects["rewards"]:
            value = reward.value
            if isinstance(value, list):
                value = value[0]
            if value > 0:
                grid[reward.pos[0], reward.pos[1]] = 3
            else:
                grid[reward.pos[0], reward.pos[1]] = 4

        # Set keys
        for key in env.objects["keys"]:
            grid[key.pos[0], key.pos[1]] = 5

        # Set doors
        for door in env.objects["doors"]:
            grid[door.pos[0], door.pos[1]] = 6

        # Set warps
        for warp in env.objects["warps"]:
            grid[warp.pos[0], warp.pos[1]] = 7

        # Set other objects
        for other in env.objects["other"]:
            grid[other.pos[0], other.pos[1]] = 8

        # Convert grid to ASCII string
        return "\n".join(
            "".join(self.ascii_map[int(cell)] for cell in row) for row in grid
        )

    def render(self, env: Any, **kwargs) -> str:
        # Render method for ASCII observation.
        return self.make_ascii_obs(env)
