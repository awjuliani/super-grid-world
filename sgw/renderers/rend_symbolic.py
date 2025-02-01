import numpy as np
from typing import Dict, List, Tuple, Any
from sgw.renderers.rend_interface import RendererInterface


class GridSymbolicRenderer(RendererInterface):
    def __init__(self, grid_size: int):
        self.grid_size = grid_size

    def make_symbolic_obs(
        self,
        agent_pos: List[int],
        objects: Dict[str, Any],
        visible_walls: bool,
    ) -> np.ndarray:
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
        grid[agent_pos[0], agent_pos[1], 0] = 1

        # Set rewards
        reward_list = [
            (loc, reward)
            for loc, reward in objects["rewards"].items()
            if type(reward) != list or reward[1] == 1
        ]
        for loc, reward in reward_list:
            if type(reward) == list:
                reward = reward[0]
            grid[loc[0], loc[1], 1] = reward

        # Set keys
        key_locs = objects["keys"]
        for loc in key_locs:
            grid[loc[0], loc[1], 2] = 1

        # Set doors
        door_locs = objects["doors"]
        for loc in door_locs:
            grid[loc[0], loc[1], 3] = 1

        # Set warps
        warp_locs = objects["warps"].keys()
        for loc in warp_locs:
            grid[loc[0], loc[1], 5] = 1

        # Set walls
        if visible_walls:
            walls = self.render_walls(objects["walls"])
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

    def make_symbolic_window_obs(
        self,
        agent_pos: List[int],
        objects: Dict[str, Any],
        visible_walls: bool,
        size: int = 5,
    ) -> np.ndarray:
        """
        Returns a windowed symbolic observation centered on the agent.
        """
        if size not in [3, 5]:
            raise ValueError("Window size must be 3 or 5")

        obs = self.make_symbolic_obs(agent_pos, objects, visible_walls)
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
            agent_pos[0] : agent_pos[0] + size,
            agent_pos[1] : agent_pos[1] + size,
            :,
        ]

        return window

    def render(self, env, **kwargs):
        # Common interface render method.
        # If 'window' is True, optionally using provided 'size' parameter, use windowed observation.
        if kwargs.get("window", False):
            size = kwargs.get("size", 5)
            return self.make_symbolic_window_obs(
                env.agent_pos, env.objects, env.visible_walls, size
            )
        return self.make_symbolic_obs(env.agent_pos, env.objects, env.visible_walls)
