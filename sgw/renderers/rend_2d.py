import numpy as np
import cv2 as cv
from typing import Tuple, List, Dict, Any, Callable
from gym import spaces
import copy

from sgw.renderers.rend_interface import RendererInterface
from sgw.utils.base_utils import resize_obs
from sgw.object import (
    Wall,
    Reward,
    Key,
    Door,
    Warp,
    Other,
    LinkedDoor,
    PressurePlate,
    Lever,
    Tree,
    Fruit,
    Sign,
    Box,
    PushableBox,
    ResetButton,
    Marker,
)
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
    BOX_FILL = (160, 82, 45)  # Sienna brown
    BOX_BORDER = (139, 69, 19)  # Saddle brown
    PUSHABLE_BOX_FILL = (205, 133, 63)  # Peru (lighter brown)
    PUSHABLE_BOX_BORDER = (160, 82, 45)  # Sienna brown
    LINKED_DOOR_CLOSED_FILL = (100, 70, 50)  # Darker brown for closed
    LINKED_DOOR_CLOSED_BORDER = (70, 50, 40)
    LINKED_DOOR_OPEN_FILL = (140, 100, 80)  # Lighter brown for open
    LINKED_DOOR_OPEN_BORDER = (100, 70, 50)
    PRESSURE_PLATE_INACTIVE_FILL = (192, 192, 192)  # Silver
    PRESSURE_PLATE_INACTIVE_BORDER = (160, 160, 160)
    PRESSURE_PLATE_ACTIVE_FILL = (218, 165, 32)  # Goldenrod (to indicate activation)
    PRESSURE_PLATE_ACTIVE_BORDER = (184, 134, 11)
    LEVER_BASE_FILL = (105, 105, 105)  # Dim gray
    LEVER_BASE_BORDER = (85, 85, 85)
    LEVER_HANDLE_INACTIVE_FILL = (255, 0, 0)  # Red (inactive)
    LEVER_HANDLE_ACTIVE_FILL = (0, 255, 0)  # Green (active)
    LEVER_HANDLE_BORDER = (50, 50, 50)
    RESET_BUTTON_FILL = (200, 150, 255)  # Light purple fill
    RESET_BUTTON_BORDER = (150, 100, 200)  # Darker light purple border

    # Dictionary mapping color names to RGB values for agents
    AGENT_COLOR_DICT = {
        "Magenta": (255, 0, 255),
        "Orange": (255, 165, 0),
        "Cyan": (0, 255, 255),
        "Purple": (128, 0, 128),
        "Brown": (165, 42, 42),
        "Dark Green": (0, 128, 0),
        "Steel Blue": (70, 130, 180),
        "Rosy Brown": (188, 143, 143),
        "Sea Green": (46, 139, 87),
        "Gold": (255, 215, 0),
        "Slate Blue": (106, 90, 205),
        "Tomato": (255, 99, 71),
        "Medium Sea Green": (60, 179, 113),
        "Violet": (238, 130, 238),
        "Dark Orange": (255, 140, 0),
        "Light Sea Green": (32, 178, 170),
    }

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
        self.register_renderer("linked_doors", self._render_linked_doors)
        self.register_renderer("pressure_plates", self._render_pressure_plates)
        self.register_renderer("levers", self._render_levers)
        self.register_renderer("warps", self._render_warps)
        self.register_renderer("other", self._render_other)
        self.register_renderer("trees", self._render_trees)
        self.register_renderer("fruits", self._render_fruits)
        self.register_renderer("signs", self._render_signs)
        self.register_renderer("boxes", self._render_boxes)
        self.register_renderer("pushable_boxes", self._render_pushable_boxes)
        self.register_renderer("reset_buttons", self._render_reset_buttons)

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
        start_coord, end_coord = self.get_square_edges(
            pos
        )  # Use these for centering calculation

        # Calculate center and dimensions for the gem
        center_x = (start_coord[0] + end_coord[0]) // 2
        center_y = (start_coord[1] + end_coord[1]) // 2
        # Ensure width and height are integers for coordinate calculations
        width = int((end_coord[0] - start_coord[0]) * 0.8)  # Cast to int
        height = int((end_coord[1] - start_coord[1]) * 0.8)  # Cast to int
        half_width = width // 2  # Now guaranteed to be int
        half_height = height // 2  # Now guaranteed to be int

        # Define the vertices of the gem polygon (octagon shape)
        # Order: Top, Top-Right, Mid-Right, Bottom-Right, Bottom, Bottom-Left, Mid-Left, Top-Left
        pts = np.array(
            [
                [center_x, center_y - half_height],  # Top point
                [
                    center_x + half_width // 2,
                    center_y - half_height // 2,
                ],  # Top-Right facet point
                [center_x + half_width, center_y],  # Mid-Right point (widest)
                [
                    center_x + half_width // 2,
                    center_y + half_height // 2,
                ],  # Bottom-Right facet point
                [center_x, center_y + half_height],  # Bottom point
                [
                    center_x - half_width // 2,
                    center_y + half_height // 2,
                ],  # Bottom-Left facet point
                [center_x - half_width, center_y],  # Mid-Left point (widest)
                [
                    center_x - half_width // 2,
                    center_y - half_height // 2,
                ],  # Top-Left facet point
            ],
            np.int32,
        )
        pts = pts.reshape((-1, 1, 2))  # Reshape for OpenCV polygon functions

        # Draw the filled gem polygon
        cv.fillPoly(img, [pts], fill_color)

        # Draw the gem border
        cv.polylines(
            img,
            [pts],
            isClosed=True,
            color=border_color,
            thickness=self.block_border - 1,
        )

        # Optional: Add a simple shine/facet line for visual flair
        # Ensure coordinates are integers before creating tuples
        shine_start = (center_x - half_width // 2, center_y - half_height // 2)
        shine_end = (center_x + half_width // 4, center_y - half_height // 4)
        shine_color = tuple(
            min(c + 50, 255) for c in fill_color
        )  # Slightly lighter fill color
        cv.line(img, shine_start, shine_end, shine_color, 1)

    def _process_reward(self, reward: Any) -> Tuple[bool, float, float]:
        if isinstance(reward, list):
            draw = reward[1]
            # Factor might not be directly applicable to polygon size in the same way
            # We'll pass it but _draw_reward currently calculates size based on block_size
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

            # Calculate key dimensions based on block size
            key_head_radius_outer = self.block_size // 5
            key_head_radius_inner = self.block_size // 10
            key_stem_length = self.block_size // 3
            key_stem_width = self.block_size // 8
            key_tooth_width = self.block_size // 10
            key_tooth_height = self.block_size // 10

            # Draw key head (donut shape - outer circle)
            head_center = (center_x - key_stem_length // 3, center_y)
            cv.circle(img, head_center, key_head_radius_outer, self.KEY_FILL, -1)
            cv.circle(
                img,
                head_center,
                key_head_radius_outer,
                self.KEY_BORDER,
                self.block_border - 2,
            )

            # Draw inner circle (hole) to create donut effect
            cv.circle(
                img, head_center, key_head_radius_inner, self.BACKGROUND_COLOR, -1
            )
            cv.circle(img, head_center, key_head_radius_inner, self.KEY_BORDER, 1)

            # Draw key stem (rectangle) - starting at the right edge of the donut
            stem_start_x = head_center[0] + key_head_radius_outer
            stem_end_x = center_x + key_stem_length * 2 // 2
            stem_start_y = center_y - key_stem_width // 2
            stem_end_y = center_y + key_stem_width // 2

            cv.rectangle(
                img,
                (stem_start_x, stem_start_y),
                (stem_end_x, stem_end_y),
                self.KEY_FILL,
                -1,
            )
            cv.rectangle(
                img,
                (stem_start_x, stem_start_y),
                (stem_end_x, stem_end_y),
                self.KEY_BORDER,
                1,
            )

            # Draw single downward tooth
            tooth_start_x = stem_end_x - key_tooth_width
            tooth_end_x = stem_end_x
            tooth_start_y = stem_end_y
            tooth_end_y = stem_end_y + key_tooth_height

            cv.rectangle(
                img,
                (tooth_start_x, tooth_start_y),
                (tooth_end_x, tooth_end_y),
                self.KEY_FILL,
                -1,
            )
            cv.rectangle(
                img,
                (tooth_start_x, tooth_start_y),
                (tooth_end_x, tooth_end_y),
                self.KEY_BORDER,
                1,
            )

    def _render_doors(self, img: np.ndarray, doors: List[Door]) -> None:
        # Standard doors (key operated)
        for door in doors:
            start, end = self.get_square_edges(door.pos)
            center_x = (start[0] + end[0]) // 2
            center_y = (start[1] + end[1]) // 2

            if door.obstacle:  # Door is locked/closed
                fill_color = self.DOOR_FILL
                border_color = self.DOOR_BORDER
                # Draw the closed door square
                cv.rectangle(img, start, end, fill_color, -1)
                cv.rectangle(img, start, end, border_color, self.block_border - 1)

                # Draw keyhole symbol
                keyhole_radius = self.block_size // 12
                keyhole_stem_height = self.block_size // 6
                keyhole_stem_width = self.block_size // 16

                # Keyhole top circle
                cv.circle(
                    img,
                    (center_x, center_y - keyhole_stem_height // 4),
                    keyhole_radius,
                    self.DOOR_BORDER,
                    -1,
                )
                # Keyhole bottom stem (rectangle)
                stem_start_x = center_x - keyhole_stem_width // 2
                stem_end_x = center_x + keyhole_stem_width // 2
                stem_start_y = center_y - keyhole_stem_height // 4 + keyhole_radius // 2
                stem_end_y = stem_start_y + keyhole_stem_height
                cv.rectangle(
                    img,
                    (stem_start_x, stem_start_y),
                    (stem_end_x, stem_end_y),
                    self.DOOR_BORDER,
                    -1,
                )

            else:  # Door is unlocked/open
                fill_color = self.BACKGROUND_COLOR
                border_color = self.GRID_LINE_COLOR
                # Draw as open passage (background color)
                cv.rectangle(img, start, end, fill_color, -1)
                # Optional: Draw a faint frame to show where the door was
                frame_thickness = 1
                cv.rectangle(img, start, end, self.DOOR_BORDER, frame_thickness)

    def _render_linked_doors(self, img: np.ndarray, doors: List[LinkedDoor]) -> None:
        # Linked doors (plate/lever operated)
        for door in doors:
            start, end = self.get_square_edges(door.pos)
            center_x = (start[0] + end[0]) // 2
            center_y = (start[1] + end[1]) // 2

            if door.is_open:
                fill_color = self.LINKED_DOOR_OPEN_FILL
                border_color = self.LINKED_DOOR_OPEN_BORDER
                # Draw as open passage (background color)
                cv.rectangle(img, start, end, self.BACKGROUND_COLOR, -1)
                # Optionally draw frame or indication it *was* a door
                # Draw thin frame:
                frame_thickness = 1
                cv.rectangle(img, start, end, border_color, frame_thickness)

            else:  # Door is closed
                fill_color = self.LINKED_DOOR_CLOSED_FILL
                border_color = self.LINKED_DOOR_CLOSED_BORDER
                # Draw closed door as full block
                cv.rectangle(img, start, end, fill_color, -1)
                cv.rectangle(img, start, end, border_color, self.block_border - 1)

                # Add the '!' symbol
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6  # Adjust size as needed
                thickness = 2  # Adjust thickness as needed
                text = "!"
                text_color = (255, 255, 255)  # White text for contrast
                (text_width, text_height), _ = cv.getTextSize(
                    text, font, font_scale, thickness
                )
                # Center the text within the door square
                text_pos = (center_x - text_width // 2, center_y + text_height // 2)
                cv.putText(img, text, text_pos, font, font_scale, text_color, thickness)

    def _render_pressure_plates(
        self, img: np.ndarray, plates: List[PressurePlate]
    ) -> None:
        """Render pressure plates as flat squares."""
        # Need access to env state to know if plate is active (agent on it)
        # This renderer currently doesn't get env state easily for non-agent renders.
        # For now, render all as inactive. Activation state might require renderer refactor.
        # TODO: Find a way to check if agent is on the plate to change color.
        for plate in plates:
            start, end = self.get_square_edges(plate.pos)
            # Shrink slightly to make it look flat on the floor
            inset = self.block_border
            plate_start = (start[0] + inset, start[1] + inset)
            plate_end = (end[0] - inset, end[1] - inset)

            # Assume inactive for now
            fill_color = self.PRESSURE_PLATE_INACTIVE_FILL
            border_color = self.PRESSURE_PLATE_INACTIVE_BORDER

            cv.rectangle(img, plate_start, plate_end, fill_color, -1)
            cv.rectangle(img, plate_start, plate_end, border_color, 1)  # Thin border

    def _render_levers(self, img: np.ndarray, levers: List[Lever]) -> None:
        """Render levers with a base and a handle indicating state."""
        for lever in levers:
            start, end = self.get_square_edges(lever.pos)
            center_x = (start[0] + end[0]) // 2
            center_y = (start[1] + end[1]) // 2

            # Draw base (small square)
            base_size = self.block_size // 4
            base_start = (center_x - base_size, center_y - base_size)
            base_end = (center_x + base_size, center_y + base_size)
            cv.rectangle(img, base_start, base_end, self.LEVER_BASE_FILL, -1)
            cv.rectangle(img, base_start, base_end, self.LEVER_BASE_BORDER, 1)

            # Draw handle (line indicating state)
            handle_length = self.block_size // 3
            handle_thickness = 3
            if lever.activated:
                # Point right when active
                handle_end_x = center_x + handle_length
                handle_end_y = center_y
                handle_color = self.LEVER_HANDLE_ACTIVE_FILL
            else:
                # Point left when inactive
                handle_end_x = center_x - handle_length
                handle_end_y = center_y
                handle_color = self.LEVER_HANDLE_INACTIVE_FILL

            cv.line(
                img,
                (center_x, center_y),
                (handle_end_x, handle_end_y),
                self.LEVER_HANDLE_BORDER,
                handle_thickness + 2,
            )  # Border
            cv.line(
                img,
                (center_x, center_y),
                (handle_end_x, handle_end_y),
                handle_color,
                handle_thickness,
            )  # Fill

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
            post_height = self.block_size // 3
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

    def _render_boxes(self, img: np.ndarray, boxes: List[Any]) -> None:
        """Render box objects as chests with a lid line."""
        for box in boxes:

            start, end = self.get_square_edges(box.pos)

            # Draw main box (chest)
            cv.rectangle(img, start, end, self.BOX_FILL, -1)
            cv.rectangle(img, start, end, self.BOX_BORDER, self.block_border - 1)

            # Draw lid line to indicate it's a chest that can be opened
            lid_y = start[1] + (end[1] - start[1]) // 3
            cv.line(img, (start[0], lid_y), (end[0], lid_y), self.BOX_BORDER, 2)

            # Draw a small keyhole
            keyhole_x = (start[0] + end[0]) // 2
            keyhole_y = (start[1] + lid_y) // 2
            keyhole_radius = (end[0] - start[0]) // 10
            cv.circle(img, (keyhole_x, keyhole_y), keyhole_radius, self.BOX_BORDER, 1)

    def _render_pushable_boxes(
        self, img: np.ndarray, pushable_boxes: List[Any]
    ) -> None:
        """Render pushable box objects as simple boxes with directional arrows."""
        for box in pushable_boxes:
            start, end = self.get_square_edges(box.pos)

            # Draw main box (simpler than a chest)
            cv.rectangle(img, start, end, self.PUSHABLE_BOX_FILL, -1)
            cv.rectangle(
                img, start, end, self.PUSHABLE_BOX_BORDER, self.block_border - 1
            )

            # Calculate center of the box
            center_x = (start[0] + end[0]) // 2
            center_y = (start[1] + end[1]) // 2

            # Draw larger arrows in four directions
            arrow_size = (end[0] - start[0]) // 4  # Larger arrows (was // 6)
            arrow_width = 2  # Thicker lines

            # Top arrow
            cv.line(
                img,
                (center_x, center_y - arrow_size),
                (center_x, center_y),
                self.PUSHABLE_BOX_BORDER,
                arrow_width,
            )
            cv.line(
                img,
                (center_x - arrow_size // 2, center_y - arrow_size // 2),
                (center_x, center_y - arrow_size),
                self.PUSHABLE_BOX_BORDER,
                arrow_width,
            )
            cv.line(
                img,
                (center_x + arrow_size // 2, center_y - arrow_size // 2),
                (center_x, center_y - arrow_size),
                self.PUSHABLE_BOX_BORDER,
                arrow_width,
            )

            # Bottom arrow
            cv.line(
                img,
                (center_x, center_y + arrow_size),
                (center_x, center_y),
                self.PUSHABLE_BOX_BORDER,
                arrow_width,
            )
            cv.line(
                img,
                (center_x - arrow_size // 2, center_y + arrow_size // 2),
                (center_x, center_y + arrow_size),
                self.PUSHABLE_BOX_BORDER,
                arrow_width,
            )
            cv.line(
                img,
                (center_x + arrow_size // 2, center_y + arrow_size // 2),
                (center_x, center_y + arrow_size),
                self.PUSHABLE_BOX_BORDER,
                arrow_width,
            )

            # Left arrow
            cv.line(
                img,
                (center_x - arrow_size, center_y),
                (center_x, center_y),
                self.PUSHABLE_BOX_BORDER,
                arrow_width,
            )
            cv.line(
                img,
                (center_x - arrow_size // 2, center_y - arrow_size // 2),
                (center_x - arrow_size, center_y),
                self.PUSHABLE_BOX_BORDER,
                arrow_width,
            )
            cv.line(
                img,
                (center_x - arrow_size // 2, center_y + arrow_size // 2),
                (center_x - arrow_size, center_y),
                self.PUSHABLE_BOX_BORDER,
                arrow_width,
            )

            # Right arrow
            cv.line(
                img,
                (center_x + arrow_size, center_y),
                (center_x, center_y),
                self.PUSHABLE_BOX_BORDER,
                arrow_width,
            )
            cv.line(
                img,
                (center_x + arrow_size // 2, center_y - arrow_size // 2),
                (center_x + arrow_size, center_y),
                self.PUSHABLE_BOX_BORDER,
                arrow_width,
            )
            cv.line(
                img,
                (center_x + arrow_size // 2, center_y + arrow_size // 2),
                (center_x + arrow_size, center_y),
                self.PUSHABLE_BOX_BORDER,
                arrow_width,
            )

    def _render_reset_buttons(
        self, img: np.ndarray, buttons: List[ResetButton]
    ) -> None:
        """Render reset buttons as red circles with an 'R'."""
        for button in buttons:
            start, end = self.get_square_edges(button.pos)
            center_x = (start[0] + end[0]) // 2
            center_y = (start[1] + end[1]) // 2
            radius = (
                end[0] - start[0]
            ) // 2  # Make it slightly smaller than the square

            # Draw the button circle
            cv.circle(img, (center_x, center_y), radius, self.RESET_BUTTON_FILL, -1)
            cv.circle(
                img,
                (center_x, center_y),
                radius,
                self.RESET_BUTTON_BORDER,
                self.block_border - 1,
            )

            # Add the 'R' symbol
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5  # Adjust size as needed
            thickness = 1  # Adjust thickness as needed
            text = "R"
            text_color = (255, 255, 255)  # White text
            (text_width, text_height), _ = cv.getTextSize(
                text, font, font_scale, thickness
            )
            # Center the text within the circle
            text_pos = (center_x - text_width // 2, center_y + text_height // 2)
            cv.putText(img, text, text_pos, font, font_scale, text_color, thickness)

    def _create_new_frame(self, env: Any) -> np.ndarray:
        img = self._create_base_image()
        self._render_gridlines(img)

        # Iterate through registered renderers ensuring new keys are handled
        for key, renderer in self.object_renderers.items():
            # Ensure the key exists and corresponds to a list in env.objects
            if (
                key in env.objects
                and isinstance(env.objects[key], list)
                and env.objects[key]
            ):
                renderer(img, env.objects[key])
            # Handle potential cases where env.objects might not have the key yet
            # elif key not in env.objects:
            #     print(f"Warning: Renderer registered for '{key}', but key not found in env.objects.")

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
        """Determine if the renderer cache needs to be updated based on changes in the environment."""
        if self.cached_objects is None or self.cached_image is None:
            return True

        # Check if object types (keys) have changed
        if set(self.cached_objects.keys()) != set(env.objects.keys()):
            return True

        # Check if the number of objects of each type has changed
        for key in self.cached_objects:
            if key not in env.objects or len(self.cached_objects[key]) != len(
                env.objects[key]
            ):
                return True

        # Check if any object's position or relevant state has changed
        for key in self.cached_objects:
            # Create sets/dicts for quick comparison based on object type
            if key == "linked_doors":
                cached_states = {
                    (tuple(obj.pos), obj.is_open) for obj in self.cached_objects[key]
                }
                current_states = {
                    (tuple(obj.pos), obj.is_open) for obj in env.objects[key]
                }
                if cached_states != current_states:
                    return True
            elif key == "levers":
                cached_states = {
                    (tuple(obj.pos), obj.activated) for obj in self.cached_objects[key]
                }
                current_states = {
                    (tuple(obj.pos), obj.activated) for obj in env.objects[key]
                }
                if cached_states != current_states:
                    return True
            # Add checks for pressure plates if their visual state depends on agent pos
            # elif key == 'pressure_plates': ...
            else:  # Default position check for other types
                cached_positions = {tuple(obj.pos) for obj in self.cached_objects[key]}
                current_positions = {tuple(obj.pos) for obj in env.objects[key]}
                if cached_positions != current_positions:
                    return True

        return False

    def _update_cache(self, env: Any) -> None:
        # Deep copy objects to cache their state
        self.cached_objects = {}
        for key, value in env.objects.items():
            self.cached_objects[key] = [
                obj.copy() if hasattr(obj, "copy") else copy.deepcopy(obj)
                for obj in value
            ]
        # Re-render the base image with new states
        self.cached_image = self._create_new_frame(env)

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
            # Get list of color names for indexing
            color_names = list(self.AGENT_COLOR_DICT.keys())

            # Print legend connecting agent names to colors
            print("\n=== Agent Color Legend ===")
            for i, agent in enumerate(env.agents):
                # Get agent color (use modulo to handle more agents than colors)
                color_index = i % len(color_names)
                color_name = color_names[color_index]
                agent_color = self.AGENT_COLOR_DICT[color_name]

                # Get agent name or index if name not available
                agent_name = getattr(agent, "name", f"Agent {i}")

                # Print color legend with color name
                print(f"{agent_name}: {color_name}")
            print("========================\n")

            # Render each agent with a different color
            for i, agent in enumerate(env.agents):
                # Use modulo to handle more agents than colors
                color_index = i % len(color_names)
                color_name = color_names[color_index]
                agent_color = self.AGENT_COLOR_DICT[color_name]
                self.render_agent(img, agent.pos, agent.looking, agent_color)
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

        # Calculate padding based on field of view
        # We need at least w_size blocks of padding on each side
        padding_blocks = max(w_size, 1)  # Ensure at least 1 block of padding
        padded_width = self.img_width + (2 * padding_blocks * self.block_size)
        padded_height = self.img_height + (2 * padding_blocks * self.block_size)
        template = np.ones((padded_height, padded_width, 3), dtype=np.uint8) * np.array(
            self.TEMPLATE_COLOR, dtype=np.uint8
        )

        # Place the image in the center of the padded template
        padding_pixels = padding_blocks * self.block_size
        template[
            padding_pixels : padding_pixels + self.img_height,
            padding_pixels : padding_pixels + self.img_width,
        ] = img

        x, y = env.agents[agent_idx].pos
        window_size = (2 * w_size + 1) * self.block_size
        # Adjust block positions to account for the new padding
        x_start_block = (x * self.block_size) + padding_pixels
        x_end_block = ((x + 1) * self.block_size) + padding_pixels
        y_start_block = (y * self.block_size) + padding_pixels
        y_end_block = ((y + 1) * self.block_size) + padding_pixels

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
