import numpy as np
from typing import Dict, List, Tuple, Any
from sgw.renderers.rend_interface import RendererInterface
from gym import spaces


class GridSymbolicRenderer(RendererInterface):
    def __init__(self, grid_size: int, window_size: int = None):
        self.grid_size = grid_size
        self.window_size = window_size

    @property
    def observation_space(self) -> spaces.Space:
        """Return the observation space for symbolic observations."""
        if self.window_size is None:
            shape = (self.grid_size, self.grid_size, 6)
        elif self.window_size == 3:
            shape = (3, 3, 6)
        else:  # window_size == 5
            shape = (5, 5, 6)
        return spaces.Box(0, 1, shape=shape)

    def render_full(self, env: Any) -> np.ndarray:
        """
        Returns a symbolic representation of the environment in a numpy tensor.
        Tensor shape is (grid_size, grid_size, 6)
        6 channels are:
            0: agent
            1: rewards
            2: keys
            3: doors
            4: walls
            5: warps
        """
        grid = np.zeros([self.grid_size, self.grid_size, 6])

        # Set agent's position
        grid[env.agent.pos[0], env.agent.pos[1], 0] = 1

        # Set rewards
        for reward in env.objects["rewards"]:
            value = reward.value
            if isinstance(value, list):
                if value[1] == 1:  # Only include if active
                    grid[reward.pos[0], reward.pos[1], 1] = value[0]
            else:
                grid[reward.pos[0], reward.pos[1], 1] = value

        # Set keys
        for key in env.objects["keys"]:
            grid[key.pos[0], key.pos[1], 2] = 1

        # Set doors
        for door in env.objects["doors"]:
            grid[door.pos[0], door.pos[1], 3] = 1

        # Set warps
        for warp in env.objects["warps"]:
            grid[warp.pos[0], warp.pos[1], 5] = 1

        # Set walls
        if env.visible_walls:
            walls = self.render_walls([wall.pos for wall in env.objects["walls"]])
            grid[:, :, 4] = walls

        return grid

    def render_walls(self, walls: List[List[int]]) -> np.ndarray:
        """
        Returns a numpy array of the walls in the environment.
        """
        grid = np.zeros([self.grid_size, self.grid_size])
        for block in walls:
            grid[block[0], block[1]] = 1
        return grid

    def render_window(self, env: Any, size: int) -> np.ndarray:
        """
        Returns a windowed symbolic observation centered on the agent.
        """
        if size not in [3, 5]:
            raise ValueError("Window size must be 3 or 5")

        obs = self.render_full(env)
        pad_size = (size - 1) // 2
        full_window = np.pad(
            obs,
            ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        full_window[:, :, 4] = np.where(
            full_window[:, :, 4] == 0, 1, full_window[:, :, 4]
        )

        window = full_window[
            env.agent.pos[0] : env.agent.pos[0] + size,
            env.agent.pos[1] : env.agent.pos[1] + size,
            :,
        ]

        return window

    def render(self, env: Any, **kwargs) -> np.ndarray:
        """Render an observation from the environment using the renderer."""
        if self.window_size is not None:
            return self.render_window(env, self.window_size)
        return self.render_full(env)
