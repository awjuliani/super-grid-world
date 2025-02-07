import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Callable
from gym import spaces

from sgw.renderers.rend_interface import RendererInterface
from sgw.utils.base_utils import resize_obs
from sgw.object import Wall, Reward, Marker, Key, Door, Warp, Other


class Grid2DRenderer(RendererInterface):
    # Color constants
    BACKGROUND_COLOR = (235, 235, 235)
    GRID_LINE_COLOR = (210, 210, 210)
    WALL_COLOR_INNER = (165, 165, 165)
    WALL_COLOR_OUTER = (125, 125, 125)
    POSITIVE_REWARD_FILL = (100, 100, 255)
    POSITIVE_REWARD_BORDER = (50, 50, 200)
    NEGATIVE_REWARD_FILL = (255, 100, 100)
    NEGATIVE_REWARD_BORDER = (200, 50, 50)
    KEY_FILL = (255, 215, 0)
    KEY_BORDER = (200, 160, 0)
    DOOR_FILL = (0, 150, 0)
    DOOR_BORDER = (0, 100, 0)
    WARP_FILL = (130, 0, 250)
    WARP_BORDER = (80, 0, 200)
    AGENT_COLOR = (0, 0, 0)
    TEMPLATE_COLOR = (150, 150, 150)
    FOG_COLOR = (0, 0, 0, 150)  # Semi-transparent black for fog of war

    def __init__(
        self,
        grid_size: int,
        block_size: int = 32,
        block_border: int = 3,
        window_size: int = None,
        resolution: int = 256,
        torch_obs: bool = False,
    ):
        self.grid_size = grid_size
        self.block_size = block_size
        self.block_border = block_border
        self.window_size = window_size
        self.resolution = resolution
        self.torch_obs = torch_obs
        self.cached_image = None
        self.cached_objects = None
        self.img_size = self.block_size * self.grid_size

        # Register default object renderers
        self.object_renderers: Dict[str, Callable[[np.ndarray, List[Any]], None]] = {}
        self.register_renderer("walls", self._render_walls)
        self.register_renderer("rewards", self._render_rewards)
        self.register_renderer("markers", self._render_markers)
        self.register_renderer("keys", self._render_keys)
        self.register_renderer("doors", self._render_doors)
        self.register_renderer("warps", self._render_warps)
        self.register_renderer("other", self._render_other)

    @property
    def observation_space(self) -> spaces.Space:
        """Return the observation space for visual observations."""
        if self.torch_obs:
            return spaces.Box(0, 1, shape=(3, 64, 64))
        return spaces.Box(0, 1, shape=(self.resolution, self.resolution, 3))

    def register_renderer(
        self, key: str, renderer_fn: Callable[[np.ndarray, List[Any]], None]
    ) -> None:
        """Register a renderer for a new object type. This makes it easy to extend rendering capabilities."""
        self.object_renderers[key] = renderer_fn

    def get_square_edges(
        self, y: int, x: int
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        true_start = self.block_border + 1
        block_end = self.block_size - self.block_border - 1

        x_unit = x * self.block_size
        y_unit = y * self.block_size

        return (
            (y_unit + true_start, x_unit + true_start),
            (y_unit + block_end, x_unit + block_end),
        )

    def _create_base_image(self) -> np.ndarray:
        # Create a base image filled with the background color
        img = np.ones((self.img_size, self.img_size, 3), np.uint8) * np.array(
            self.BACKGROUND_COLOR, dtype=np.uint8
        )
        return img

    def _render_gridlines(self, img: np.ndarray) -> None:
        # Draw grid lines over the image
        for i in range(0, self.img_size + 1, self.block_size):
            cv.line(img, (0, i), (self.img_size, i), self.GRID_LINE_COLOR, 1)
            cv.line(img, (i, 0), (i, self.img_size), self.GRID_LINE_COLOR, 1)
        cv.line(
            img,
            (self.img_size - 1, 0),
            (self.img_size - 1, self.img_size - 1),
            self.GRID_LINE_COLOR,
            1,
        )
        cv.line(
            img,
            (0, self.img_size - 1),
            (self.img_size - 1, self.img_size - 1),
            self.GRID_LINE_COLOR,
            1,
        )

    def _render_walls(self, img: np.ndarray, walls: List[Wall]) -> None:
        # Render wall objects on the image
        for wall in walls:
            y, x = wall.pos
            start, end = self.get_square_edges(x, y)
            cv.rectangle(img, start, end, self.WALL_COLOR_INNER, -1)
            cv.rectangle(img, start, end, self.WALL_COLOR_OUTER, self.block_border - 1)

    def _render_rewards(self, img: np.ndarray, rewards: List[Reward]) -> None:
        # Render reward objects
        for reward in rewards:
            draw, factor, reward_value = self._process_reward(reward.value)
            if draw:
                self._draw_reward(img, reward.pos, factor, reward_value)

    def _draw_reward(
        self, img: np.ndarray, pos: Tuple[int, int], factor: float, reward_value: float
    ) -> None:
        fill_color, border_color = self._get_reward_colors(reward_value)
        start, end = self.get_square_edges(pos[1], pos[0])
        size_reduction = int(2 * factor)
        adjusted_start = (start[0] + size_reduction, start[1] + size_reduction)
        adjusted_end = (end[0] - size_reduction, end[1] - size_reduction)
        cv.rectangle(img, adjusted_start, adjusted_end, fill_color, -1)
        cv.rectangle(
            img, adjusted_start, adjusted_end, border_color, self.block_border - 1
        )

    def _process_reward(self, reward: Any) -> Tuple[bool, float, float]:
        if isinstance(reward, list):
            draw = reward[1]
            factor = 1 if reward[2] else 1.5
            reward_value = reward[0]
        else:
            draw = True
            factor = 1
            reward_value = reward
        return draw, factor, reward_value

    def _get_reward_colors(
        self, reward: float
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        if reward > 0:
            return self.POSITIVE_REWARD_FILL, self.POSITIVE_REWARD_BORDER
        else:
            return self.NEGATIVE_REWARD_FILL, self.NEGATIVE_REWARD_BORDER

    def _render_markers(self, img: np.ndarray, markers: List[Marker]) -> None:
        for marker in markers:
            fill_color = tuple(int(np.clip(c, 0, 1) * 255) for c in marker.color)
            start, end = self.get_square_edges(marker.pos[1], marker.pos[0])
            cv.rectangle(img, start, end, fill_color, -1)

    def _render_keys(self, img: np.ndarray, keys: List[Key]) -> None:
        for key in keys:
            center = (
                key.pos[1] * self.block_size + self.block_size // 2,
                key.pos[0] * self.block_size + self.block_size // 2,
            )
            pts = np.array(
                [
                    [center[0], center[1] - 4],
                    [center[0] + 4, center[1]],
                    [center[0], center[1] + 4],
                    [center[0] - 4, center[1]],
                ],
                np.int32,
            )
            cv.fillPoly(img, [pts], self.KEY_FILL)
            cv.polylines(img, [pts], True, self.KEY_BORDER, 1)

    def _render_doors(self, img: np.ndarray, doors: List[Door]) -> None:
        for door in doors:
            start, end = self.get_square_edges(door.pos[1], door.pos[0])
            if door.orientation == "h":
                start = (start[0] - 2, start[1] + 5)
                end = (end[0] + 2, end[1] - 5)
            elif door.orientation == "v":
                start = (start[0] + 5, start[1] - 2)
                end = (end[0] - 5, end[1] + 2)
            else:
                raise ValueError("Invalid door orientation")
            cv.rectangle(img, start, end, self.DOOR_FILL, -1)
            cv.rectangle(img, start, end, self.DOOR_BORDER, self.block_border - 1)

    def _render_warps(self, img: np.ndarray, warps: List[Warp]) -> None:
        for warp in warps:
            center = (
                warp.pos[1] * self.block_size + self.block_size // 2,
                warp.pos[0] * self.block_size + self.block_size // 2,
            )
            radius = self.block_size // 4
            cv.circle(img, center, radius, self.WARP_FILL, -1)
            cv.circle(img, center, radius, self.WARP_BORDER, self.block_border - 1)

    def _render_other(self, img: np.ndarray, others: List[Other]) -> None:
        for other in others:
            center = (
                other.pos[1] * self.block_size + self.block_size // 2,
                other.pos[0] * self.block_size + self.block_size // 2,
            )
            letter = other.name[0].upper() if other.name else "?"
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 0.75
            thickness = 2
            (text_width, text_height), _ = cv.getTextSize(
                letter, font, font_scale, thickness
            )
            text_pos = (center[0] - text_width // 2, center[1] + text_height // 2)
            cv.putText(
                img, letter, text_pos, font, font_scale, self.AGENT_COLOR, thickness
            )

    def _create_new_frame(self, env: Any) -> np.ndarray:
        # Create the background image and render static elements
        img = self._create_base_image()
        self._render_gridlines(img)

        # Render all objects using registered renderers
        for key, renderer in self.object_renderers.items():
            if key in env.objects:
                renderer(img, env.objects[key])
        return img

    def render_agent(
        self, img: np.ndarray, agent_pos: Tuple[int, int], agent_dir: int
    ) -> None:
        agent_dir_val = int(agent_dir) if not isinstance(agent_dir, int) else agent_dir
        agent_size = self.block_size // 2
        agent_offset = self.block_size // 4
        x_offset = agent_pos[1] * self.block_size + agent_offset
        y_offset = agent_pos[0] * self.block_size + agent_offset

        triangle_pts = {
            0: [(0, 1), (1, 1), (0.5, 0)],  # facing up
            1: [(0, 0), (0, 1), (1, 0.5)],  # facing right
            2: [(0, 0), (1, 0), (0.5, 1)],  # facing down
            3: [(1, 0), (1, 1), (0, 0.5)],  # facing left
        }

        pts = np.array(
            [
                (x_offset + pt[0] * agent_size, y_offset + pt[1] * agent_size)
                for pt in triangle_pts[agent_dir_val]
            ],
            dtype=np.int32,
        )

        cv.fillConvexPoly(img, pts, self.AGENT_COLOR)

    def _should_update_cache(self, env: Any) -> bool:
        objects_changed = self.cached_objects != env.objects
        return objects_changed or self.cached_image is None

    def _update_cache(self, env: Any) -> None:
        self.cached_objects = {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in env.objects.items()
        }

    def create_visibility_mask(self, agent_pos, vision_range):
        """Create a mask showing what's visible to the agent."""
        mask = np.zeros((self.img_size, self.img_size, 4), dtype=np.uint8)
        mask[:, :, 3] = self.FOG_COLOR[3]  # Set alpha channel for fog of war

        # Calculate visible region in pixels - using same logic as render_window
        x, y = agent_pos
        window_size = (2 * vision_range + 1) * self.block_size
        x_center = (y * self.block_size) + (
            self.block_size // 2
        )  # Swap x,y to match render_window
        y_center = (x * self.block_size) + (self.block_size // 2)
        half_window = window_size // 2

        x_start = x_center - half_window
        x_end = x_center + half_window
        y_start = y_center - half_window
        y_end = y_center + half_window

        # Ensure window boundaries stay within image
        x_start = max(0, x_start)
        x_end = min(self.img_size, x_end)
        y_start = max(0, y_start)
        y_end = min(self.img_size, y_end)

        # Set visible region to transparent
        mask[y_start:y_end, x_start:x_end, 3] = 0
        return mask

    def render_frame(self, env: Any, is_state_view: bool = False) -> np.ndarray:
        if self._should_update_cache(env):
            self._update_cache(env)
            img = self._create_new_frame(env)
            self.cached_image = img.copy()
        else:
            img = self.cached_image.copy()

        self.render_agent(img, env.agent.pos, env.agent.looking)

        # If this is the state view and we have limited vision, apply the fog of war
        if is_state_view and env.agent.field_of_view is not None:
            # Convert to RGBA
            rgba_img = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
            rgba_img[:, :, :3] = img
            rgba_img[:, :, 3] = 255  # Full opacity

            # Create and apply visibility mask
            mask = self.create_visibility_mask(env.agent.pos, env.agent.field_of_view)

            # Blend the mask with the image
            alpha = mask[:, :, 3:4].astype(float) / 255
            rgba_img = rgba_img * (1 - alpha) + mask * alpha

            return rgba_img.astype(np.uint8)

        return resize_obs(img, self.resolution, self.torch_obs)

    def render_window(self, env: Any, w_size: int = 2) -> np.ndarray:
        if self._should_update_cache(env):
            self._update_cache(env)
            img = self._create_new_frame(env)
            self.cached_image = img.copy()
        else:
            img = self.cached_image.copy()

        self.render_agent(img, env.agent.pos, env.agent.looking)
        template_size = self.img_size
        padded_size = template_size + (2 * self.block_size)
        template = np.ones((padded_size, padded_size, 3), dtype=np.uint8) * np.array(
            self.TEMPLATE_COLOR, dtype=np.uint8
        )
        template[
            self.block_size : self.block_size + template_size,
            self.block_size : self.block_size + template_size,
        ] = img

        x, y = env.agent.pos
        # Calculate window size based on vision range
        window_size = (2 * w_size + 1) * self.block_size
        x_center = (x * self.block_size) + self.block_size + (self.block_size // 2)
        y_center = (y * self.block_size) + self.block_size + (self.block_size // 2)
        half_window = window_size // 2
        x_start = x_center - half_window
        x_end = x_center + half_window
        y_start = y_center - half_window
        y_end = y_center + half_window

        # Ensure window boundaries stay within padded template
        x_start = max(0, x_start)
        x_end = min(padded_size, x_end)
        y_start = max(0, y_start)
        y_end = min(padded_size, y_end)

        window = template[x_start:x_end, y_start:y_end]
        return resize_obs(window, self.resolution, self.torch_obs)

    def render(self, env: Any, is_state_view: bool = False) -> np.ndarray:
        if self.window_size is not None and not is_state_view:
            img = self.render_window(env, self.window_size)
        else:
            img = self.render_frame(env, is_state_view)
        return img
