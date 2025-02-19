import numpy as np
import cv2 as cv
from typing import Tuple, List, Dict, Any, Callable
from gym import spaces

from sgw.renderers.rend_interface import RendererInterface
from sgw.utils.base_utils import resize_obs
from sgw.object import Wall, Reward, Marker, Key, Door, Warp, Other
from sgw.enums import ControlType


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
    OTHER_AGENT_COLOR = (100, 100, 100)
    TEMPLATE_COLOR = (150, 150, 150)
    FOG_COLOR = (0, 0, 0, 150)  # Semi-transparent black for fog of war
    TREE_FILL = (34, 139, 34)  # Forest green
    TREE_BORDER = (0, 100, 0)  # Dark green
    FRUIT_FILL = (255, 69, 0)  # Red-orange
    FRUIT_BORDER = (200, 50, 0)
    SIGN_FILL = (139, 69, 19)  # Saddle brown
    SIGN_BORDER = (101, 67, 33)  # Dark brown

    def __init__(
        self,
        grid_shape: Tuple[int, int],
        block_size: int = 32,
        block_border: int = 3,
        window_size: int = None,
        resolution: int = 256,
        torch_obs: bool = False,
    ):
        self.grid_shape = grid_shape
        self.block_size = block_size
        self.block_border = block_border
        self.window_size = window_size
        self.resolution = resolution
        self.torch_obs = torch_obs
        self.cached_image = None
        self.cached_objects = None
        self.img_height = self.block_size * self.grid_shape[0]
        self.img_width = self.block_size * self.grid_shape[1]

        # Register default object renderers
        self.object_renderers: Dict[str, Callable[[np.ndarray, List[Any]], None]] = {}
        self.register_renderer("walls", self._render_walls)
        self.register_renderer("rewards", self._render_rewards)
        self.register_renderer("markers", self._render_markers)
        self.register_renderer("keys", self._render_keys)
        self.register_renderer("doors", self._render_doors)
        self.register_renderer("warps", self._render_warps)
        self.register_renderer("other", self._render_other)
        self.register_renderer("trees", self._render_trees)
        self.register_renderer("fruits", self._render_fruits)
        self.register_renderer("signs", self._render_signs)

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
        self, pos: Tuple[int, int]
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        row, col = pos
        # OpenCV uses (x,y) coordinates where x is horizontal (column) and y is vertical (row)
        # Convert from grid coordinates (row, col) to screen coordinates (x, y)
        x_unit = col * self.block_size
        y_unit = row * self.block_size
        true_start = self.block_border + 1
        block_end = self.block_size - self.block_border - 1
        start = (x_unit + true_start, y_unit + true_start)
        end = (x_unit + block_end, y_unit + block_end)
        return start, end

    def _create_base_image(self) -> np.ndarray:
        # Create a base image filled with the background color
        img = np.ones((self.img_height, self.img_width, 3), np.uint8) * np.array(
            self.BACKGROUND_COLOR, dtype=np.uint8
        )
        return img

    def _render_gridlines(self, img: np.ndarray) -> None:
        # Draw horizontal grid lines
        for i in range(0, self.img_height + 1, self.block_size):
            cv.line(img, (0, i), (self.img_width, i), self.GRID_LINE_COLOR, 1)
        # Draw vertical grid lines
        for i in range(0, self.img_width + 1, self.block_size):
            cv.line(img, (i, 0), (i, self.img_height), self.GRID_LINE_COLOR, 1)
        # Draw outer border
        cv.rectangle(
            img,
            (0, 0),
            (self.img_width - 1, self.img_height - 1),
            self.GRID_LINE_COLOR,
            1,
        )

    def _render_walls(self, img: np.ndarray, walls: List[Wall]) -> None:
        # Render wall objects on the image
        for wall in walls:
            start, end = self.get_square_edges(wall.pos)
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
        start, end = self.get_square_edges(pos)
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
            start, end = self.get_square_edges(marker.pos)
            cv.rectangle(img, start, end, fill_color, -1)

    def _render_keys(self, img: np.ndarray, keys: List[Key]) -> None:
        for key in keys:
            # Convert from grid coordinates (row, col) to screen coordinates (x, y)
            center_x = key.pos[1] * self.block_size + self.block_size // 2
            center_y = key.pos[0] * self.block_size + self.block_size // 2

            # Key dimensions
            handle_outer_radius = self.block_size // 4
            handle_inner_radius = self.block_size // 6
            shaft_width = self.block_size // 8
            shaft_length = self.block_size // 2
            teeth_width = self.block_size // 4
            teeth_height = self.block_size // 6

            # Draw handle (donut shape using two circles)
            cv.circle(
                img,
                (center_x - shaft_length // 2, center_y),
                handle_outer_radius,
                self.KEY_FILL,
                -1,
            )
            cv.circle(
                img,
                (center_x - shaft_length // 2, center_y),
                handle_inner_radius,
                self.BACKGROUND_COLOR,
                -1,
            )
            cv.circle(
                img,
                (center_x - shaft_length // 2, center_y),
                handle_outer_radius,
                self.KEY_BORDER,
                1,
            )
            cv.circle(
                img,
                (center_x - shaft_length // 2, center_y),
                handle_inner_radius,
                self.KEY_BORDER,
                1,
            )

            # Draw shaft (rectangle)
            shaft_start = (center_x - shaft_length // 2, center_y - shaft_width // 2)
            shaft_end = (center_x + shaft_length // 2, center_y + shaft_width // 2)
            cv.rectangle(img, shaft_start, shaft_end, self.KEY_FILL, -1)
            cv.rectangle(img, shaft_start, shaft_end, self.KEY_BORDER, 1)

            # Draw teeth (small rectangles)
            teeth_x = center_x + shaft_length // 4
            teeth_start = (teeth_x, center_y - teeth_height)
            teeth_end = (teeth_x + teeth_width, center_y + teeth_height)
            cv.rectangle(img, teeth_start, teeth_end, self.KEY_FILL, -1)
            cv.rectangle(img, teeth_start, teeth_end, self.KEY_BORDER, 1)

    def _render_doors(self, img: np.ndarray, doors: List[Door]) -> None:
        for door in doors:
            start, end = self.get_square_edges(door.pos)
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
            # Convert from grid coordinates (row, col) to screen coordinates (x, y)
            center = (
                warp.pos[1] * self.block_size
                + self.block_size // 2,  # x = col * block_size
                warp.pos[0] * self.block_size
                + self.block_size // 2,  # y = row * block_size
            )
            radius = self.block_size // 4
            cv.circle(img, center, radius, self.WARP_FILL, -1)
            cv.circle(img, center, radius, self.WARP_BORDER, self.block_border - 1)

    def _render_other(self, img: np.ndarray, others: List[Other]) -> None:
        for other in others:
            # Convert from grid coordinates (row, col) to screen coordinates (x, y)
            center = (
                other.pos[1] * self.block_size
                + self.block_size // 2,  # x = col * block_size
                other.pos[0] * self.block_size
                + self.block_size // 2,  # y = row * block_size
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

    def _render_trees(self, img: np.ndarray, trees: List[Any]) -> None:
        """Render tree objects with a triangular crown and rectangular trunk."""
        for tree in trees:
            # Convert from grid coordinates (row, col) to screen coordinates (x, y)
            center_x = (
                tree.pos[1] * self.block_size + self.block_size // 2
            )  # x = col * block_size
            center_y = (
                tree.pos[0] * self.block_size + self.block_size // 2
            )  # y = row * block_size
            crown_size = self.block_size // 3
            trunk_width = self.block_size // 6
            trunk_height = self.block_size // 3

            trunk_start = (center_x - trunk_width // 2, center_y + crown_size // 2)
            trunk_end = (
                center_x + trunk_width // 2,
                center_y + crown_size // 2 + trunk_height,
            )
            cv.rectangle(img, trunk_start, trunk_end, self.TREE_BORDER, -1)

            crown_pts = np.array(
                [
                    [center_x, center_y - crown_size],
                    [center_x - crown_size, center_y + crown_size // 2],
                    [center_x + crown_size, center_y + crown_size // 2],
                ],
                np.int32,
            )
            cv.fillPoly(img, [crown_pts], self.TREE_FILL)
            cv.polylines(
                img, [crown_pts], True, self.TREE_BORDER, self.block_border - 1
            )

    def _render_fruits(self, img: np.ndarray, fruits: List[Any]) -> None:
        """Render fruit objects as small circles."""
        for fruit in fruits:
            # Convert from grid coordinates (row, col) to screen coordinates (x, y)
            center = (
                fruit.pos[1] * self.block_size
                + self.block_size // 2,  # x = col * block_size
                fruit.pos[0] * self.block_size
                + self.block_size // 2,  # y = row * block_size
            )
            radius = self.block_size // 4
            cv.circle(img, center, radius, self.FRUIT_FILL, -1)
            cv.circle(img, center, radius, self.FRUIT_BORDER, self.block_border - 1)

    def _render_signs(self, img: np.ndarray, signs: List[Any]) -> None:
        """Render sign objects as small rectangles with a post."""
        for sign in signs:
            # Convert from grid coordinates (row, col) to screen coordinates (x, y)
            center_x = sign.pos[1] * self.block_size + self.block_size // 2
            center_y = sign.pos[0] * self.block_size + self.block_size // 2

            # Draw the sign post
            post_width = self.block_size // 6
            post_height = self.block_size // 2
            post_start = (center_x - post_width // 2, center_y)
            post_end = (center_x + post_width // 2, center_y + post_height)
            cv.rectangle(img, post_start, post_end, self.SIGN_BORDER, -1)

            # Draw the sign board
            board_width = self.block_size // 2
            board_height = self.block_size // 3
            board_start = (center_x - board_width // 2, center_y - board_height)
            board_end = (center_x + board_width // 2, center_y)
            cv.rectangle(img, board_start, board_end, self.SIGN_FILL, -1)
            cv.rectangle(
                img, board_start, board_end, self.SIGN_BORDER, self.block_border - 1
            )

    def _create_new_frame(self, env: Any) -> np.ndarray:
        img = self._create_base_image()
        self._render_gridlines(img)

        for key, renderer in self.object_renderers.items():
            if key in env.objects:
                renderer(img, env.objects[key])
        return img

    def render_agent(
        self,
        img: np.ndarray,
        agent_pos: Tuple[int, int],
        agent_dir: int,
        agent_color: Tuple[int, int, int] = None,
    ) -> None:
        if agent_color is None:
            agent_color = self.AGENT_COLOR
        agent_dir = int(agent_dir)
        agent_size = self.block_size // 2
        agent_offset = self.block_size // 4
        # Convert from grid coordinates (row, col) to screen coordinates (x, y)
        x_offset = agent_pos[1] * self.block_size + agent_offset  # x = col * block_size
        y_offset = agent_pos[0] * self.block_size + agent_offset  # y = row * block_size

        triangle_pts = {
            0: [(0, 1), (1, 1), (0.5, 0)],  # North
            1: [(0, 0), (0, 1), (1, 0.5)],  # East
            2: [(0, 0), (1, 0), (0.5, 1)],  # South
            3: [(1, 0), (1, 1), (0, 0.5)],  # West
        }

        pts = np.array(
            [
                (x_offset + pt[0] * agent_size, y_offset + pt[1] * agent_size)
                for pt in triangle_pts[agent_dir]
            ],
            dtype=np.int32,
        )

        cv.fillConvexPoly(img, pts, agent_color)

    def _should_update_cache(self, env: Any) -> bool:
        objects_changed = self.cached_objects != env.objects
        return objects_changed or self.cached_image is None

    def _update_cache(self, env: Any) -> None:
        self.cached_objects = {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in env.objects.items()
        }

    # Helper methods to compute crop coordinates to remove duplicated logic
    def _compute_vertical_crop(
        self, start_block: int, end_block: int, window_size: int, direction: int = None
    ) -> Tuple[int, int]:
        if direction == 0:  # Facing up: use bottom edge of block
            v_end = end_block
            v_start = v_end - window_size
        elif direction == 2:  # Facing down: use top edge of block
            v_start = start_block
            v_end = v_start + window_size
        else:
            center = start_block + self.block_size // 2
            half = window_size // 2
            v_start = center - half
            v_end = center + half
        return v_start, v_end

    def _compute_horizontal_crop(
        self, start_block: int, end_block: int, window_size: int, direction: int = None
    ) -> Tuple[int, int]:
        if direction == 1:  # Facing right: use left edge
            h_start = start_block
            h_end = h_start + window_size
        elif direction == 3:  # Facing left: use right edge
            h_end = end_block
            h_start = h_end - window_size
        else:
            center = start_block + self.block_size // 2
            half = window_size // 2
            h_start = center - half
            h_end = center + half
        return h_start, h_end

    def create_visibility_mask(
        self, agent_pos, vision_range, egocentric=False, looking=None
    ):
        mask = np.zeros((self.img_height, self.img_width, 4), dtype=np.uint8)
        mask[:, :, 3] = self.FOG_COLOR[3]  # Set alpha channel for fog of war
        window_size = (2 * vision_range + 1) * self.block_size
        row_start_block = agent_pos[0] * self.block_size
        row_end_block = (agent_pos[0] + 1) * self.block_size
        col_start_block = agent_pos[1] * self.block_size
        col_end_block = (agent_pos[1] + 1) * self.block_size

        if not egocentric:
            v_start, v_end = self._compute_vertical_crop(
                row_start_block, row_end_block, window_size
            )
            h_start, h_end = self._compute_horizontal_crop(
                col_start_block, col_end_block, window_size
            )
        else:
            if looking in [0, 2]:
                v_start, v_end = self._compute_vertical_crop(
                    row_start_block, row_end_block, window_size, direction=looking
                )
                h_start, h_end = self._compute_horizontal_crop(
                    col_start_block, col_end_block, window_size
                )
            elif looking in [1, 3]:
                v_start, v_end = self._compute_vertical_crop(
                    row_start_block, row_end_block, window_size
                )
                h_start, h_end = self._compute_horizontal_crop(
                    col_start_block, col_end_block, window_size, direction=looking
                )
            else:
                v_start, v_end = self._compute_vertical_crop(
                    row_start_block, row_end_block, window_size
                )
                h_start, h_end = self._compute_horizontal_crop(
                    col_start_block, col_end_block, window_size
                )

        v_start = max(0, v_start)
        v_end = min(self.img_height, v_end)
        h_start = max(0, h_start)
        h_end = min(self.img_width, h_end)
        mask[v_start:v_end, h_start:h_end, 3] = 0
        return mask

    def _apply_fog(self, img: np.ndarray, env: Any, agent_idx: int) -> np.ndarray:
        rgba_img = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
        rgba_img[:, :, :3] = img
        rgba_img[:, :, 3] = 255
        combined_mask = np.full(
            (self.img_height, self.img_width), self.FOG_COLOR[3], dtype=np.uint8
        )
        for agent in env.agents:
            if agent.field_of_view is not None:
                egocentric = env.control_type == ControlType.egocentric
                mask = self.create_visibility_mask(
                    agent.pos,
                    agent.field_of_view,
                    egocentric,
                    agent.looking if egocentric else None,
                )
                combined_mask = np.minimum(combined_mask, mask[:, :, 3])
        fog = np.zeros_like(rgba_img)
        fog[:, :] = self.FOG_COLOR
        fog[:, :, 3] = combined_mask
        alpha = fog[:, :, 3:4].astype(float) / 255.0
        rgba_img[:, :, :3] = (
            rgba_img[:, :, :3].astype(float) * (1 - alpha)
            + fog[:, :, :3].astype(float) * alpha
        ).astype(np.uint8)
        rgba_img[:, :, 3] = 255 - combined_mask
        return rgba_img

    def _get_base_image_cached(self, env: Any) -> np.ndarray:
        if self._should_update_cache(env):
            self._update_cache(env)
            base_img = self._create_new_frame(env)
            self.cached_image = base_img.copy()
        else:
            base_img = self.cached_image.copy()
        return base_img

    def _render_agents(
        self, img: np.ndarray, env: Any, agent_idx: int, is_state_view: bool
    ) -> None:
        if is_state_view:
            for agent in env.agents:
                self.render_agent(img, agent.pos, agent.looking, self.OTHER_AGENT_COLOR)
        else:
            for i, agent in enumerate(env.agents):
                color = self.AGENT_COLOR if i == agent_idx else self.OTHER_AGENT_COLOR
                self.render_agent(img, agent.pos, agent.looking, color)

    def render_frame(
        self, env: Any, agent_idx: int = 0, is_state_view: bool = False
    ) -> np.ndarray:
        img = self._get_base_image_cached(env)
        self._render_agents(img, env, agent_idx, is_state_view)
        if is_state_view and env.agents[agent_idx].field_of_view is not None:
            return self._apply_fog(img, env, agent_idx)
        return resize_obs(img, self.resolution, self.torch_obs)

    def _rotate_image(self, image: np.ndarray, agent_dir: int) -> np.ndarray:
        """Rotate image so agent faces upward (rotate 90° clockwise * number of turns needed)."""
        if agent_dir > 0:
            return np.rot90(image, k=agent_dir, axes=(0, 1))
        return image

    def render_window(
        self, env: Any, w_size: int = 2, agent_idx: int = 0, is_state_view: bool = False
    ) -> np.ndarray:
        img = self._get_base_image_cached(env)
        self._render_agents(img, env, agent_idx, is_state_view)

        # Create padded template with correct dimensions
        padded_width = self.img_width + (2 * self.block_size)
        padded_height = self.img_height + (2 * self.block_size)
        template = np.ones((padded_height, padded_width, 3), dtype=np.uint8) * np.array(
            self.TEMPLATE_COLOR, dtype=np.uint8
        )

        # Place the image in the center of the padded template
        template[
            self.block_size : self.block_size + self.img_height,
            self.block_size : self.block_size + self.img_width,
        ] = img

        x, y = env.agents[agent_idx].pos
        window_size = (2 * w_size + 1) * self.block_size
        x_start_block = (x * self.block_size) + self.block_size
        x_end_block = ((x + 1) * self.block_size) + self.block_size
        y_start_block = (y * self.block_size) + self.block_size
        y_end_block = ((y + 1) * self.block_size) + self.block_size

        if env.control_type == ControlType.egocentric:
            agent_dir = env.agents[agent_idx].looking
            if agent_dir in [0, 2]:
                x_start, x_end = self._compute_vertical_crop(
                    x_start_block, x_end_block, window_size, direction=agent_dir
                )
                y_start, y_end = self._compute_horizontal_crop(
                    y_start_block, y_end_block, window_size
                )
            elif agent_dir in [1, 3]:
                x_start, x_end = self._compute_vertical_crop(
                    x_start_block, x_end_block, window_size
                )
                y_start, y_end = self._compute_horizontal_crop(
                    y_start_block, y_end_block, window_size, direction=agent_dir
                )
            else:
                x_start, x_end = self._compute_vertical_crop(
                    x_start_block, x_end_block, window_size
                )
                y_start, y_end = self._compute_horizontal_crop(
                    y_start_block, y_end_block, window_size
                )
        else:
            x_start, x_end = self._compute_vertical_crop(
                x_start_block, x_end_block, window_size
            )
            y_start, y_end = self._compute_horizontal_crop(
                y_start_block, y_end_block, window_size
            )

        x_start = max(0, x_start)
        x_end = min(padded_height, x_end)
        y_start = max(0, y_start)
        y_end = min(padded_width, y_end)
        window = template[x_start:x_end, y_start:y_end]
        window = resize_obs(window, self.resolution, self.torch_obs)

        # Rotate the window in egocentric mode so agent faces up
        if env.control_type == ControlType.egocentric:
            window = self._rotate_image(window, env.agents[agent_idx].looking)

        return window

    def render(
        self, env: Any, agent_idx: int = 0, is_state_view: bool = False
    ) -> np.ndarray:
        if self.window_size is not None:
            img = self.render_window(env, self.window_size, agent_idx, is_state_view)
        else:
            img = self.render_frame(env, agent_idx, is_state_view)
        return img
