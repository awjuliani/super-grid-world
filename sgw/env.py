from typing import Dict
from gym import Env, spaces
import numpy as np
import random
import enum
from sgw.templates import (
    generate_layout,
    LayoutTemplate,
    GridSize,
    add_outer,
)
from sgw.renderers.rend_2d import Grid2DRenderer
from sgw.renderers.rend_lang import GridLangRenderer
from sgw.renderers.rend_symbolic import GridSymbolicRenderer
from sgw.renderers.rend_ascii import GridASCIIRenderer
from sgw.renderers.rend_3d import Grid3DRenderer
import matplotlib.pyplot as plt
import copy


class ObservationType(enum.Enum):
    visual = "visual"
    visual_window = "visual_window"
    visual_window_tight = "visual_window_tight"
    symbolic = "symbolic"
    symbolic_window = "symbolic_window"
    symbolic_window_tight = "symbolic_window_tight"
    rendered_3d = "rendered_3d"
    ascii = "ascii"
    language = "language"


class ControlType(enum.Enum):
    allocentric = "allocentric"
    egocentric = "egocentric"


# New enum to represent all possible actions semantically.
class Action(enum.Enum):
    # Actions for allocentric orientation (direct movement)
    MOVE_UP = enum.auto()
    MOVE_RIGHT = enum.auto()
    MOVE_DOWN = enum.auto()
    MOVE_LEFT = enum.auto()
    # Actions for egocentric orientation (rotate then move)
    ROTATE_LEFT = enum.auto()
    ROTATE_RIGHT = enum.auto()
    MOVE_FORWARD = enum.auto()
    # Optional actions for both types
    NOOP = enum.auto()
    COLLECT = enum.auto()


class SuperGridWorld(Env):
    """
    Super Grid World. A 2D maze-like OpenAI gym compatible RL environment.

    Parameters
    ----------
    template : GridTemplate
        The layout template to use for the environment.
    size : GridSize
        The size of the grid (micro, small, large).
    obs_type : ObservationType
        The type of observation to use.
    control_type : ControlType
        The type of control to use.
    seed : int
        The seed to use for the environment.
    use_noop : bool
        Whether to include a no-op action in the action space.
    torch_obs : bool
        Whether to use torch observations.
        This converts the observation to a torch tensor.
        If the observation is an image, it will be in the shape (3, 64, 64).
    manual_collect : bool
        Whether to use the collect reward action (default == False).
    """

    def __init__(
        self,
        template: LayoutTemplate = LayoutTemplate.empty,
        size: GridSize = GridSize.small,
        obs_type: ObservationType = ObservationType.visual,
        control_type: ControlType = ControlType.allocentric,
        seed: int = None,
        use_noop: bool = False,
        torch_obs: bool = False,
        manual_collect: bool = False,
        resolution: int = 256,
        add_outer_walls: bool = True,
        vision_range: float = None,
    ):
        # Initialize basic attributes
        self._init_basic_attrs(
            seed, use_noop, manual_collect, add_outer_walls, vision_range
        )

        # Setup grid and walls
        self._init_grid(template, size)

        # Setup action and observation spaces
        self.set_action_space(control_type)
        self.set_obs_space(obs_type, torch_obs, resolution)

    def _init_basic_attrs(
        self, seed, use_noop, manual_collect, add_outer_walls, vision_range
    ):
        self.rng = np.random.RandomState(seed)
        self.use_noop = use_noop
        self.manual_collect = manual_collect
        self.add_outer_walls = add_outer_walls
        self.vision_range = vision_range
        self.agent_pos = [0, 0]
        self.direction_map = np.array(
            [[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0], [0, 0]]
        )

    def _init_grid(self, template, size):
        walls, self.agent_start_pos, self.template_objects = generate_layout(
            template, size
        )
        if self.add_outer_walls:
            walls = add_outer(walls, size.value)
        self.grid_size = size.value
        self.base_objects = {
            "rewards": {},
            "markers": {},
            "keys": [],
            "doors": {},
            "warps": {},
            "other": {},
            "walls": walls,
        }

    def _init_renderers(self, resolution, torch_obs):
        """Initialize renderers based on observation type."""
        if self.obs_type == ObservationType.visual:
            self.renderer = Grid2DRenderer(
                self.grid_size, resolution=resolution, torch_obs=torch_obs
            )
        elif self.obs_type == ObservationType.visual_window:
            self.renderer = Grid2DRenderer(
                self.grid_size,
                window_size=2,
                resolution=resolution,
                torch_obs=torch_obs,
            )
        elif self.obs_type == ObservationType.visual_window_tight:
            self.renderer = Grid2DRenderer(
                self.grid_size,
                window_size=1,
                resolution=resolution,
                torch_obs=torch_obs,
            )
        elif self.obs_type == ObservationType.symbolic:
            self.renderer = GridSymbolicRenderer(self.grid_size)
        elif self.obs_type == ObservationType.symbolic_window:
            self.renderer = GridSymbolicRenderer(self.grid_size, window_size=5)
        elif self.obs_type == ObservationType.symbolic_window_tight:
            self.renderer = GridSymbolicRenderer(self.grid_size, window_size=3)
        elif self.obs_type == ObservationType.rendered_3d:
            self.renderer = Grid3DRenderer(resolution)
        elif self.obs_type == ObservationType.ascii:
            self.renderer = GridASCIIRenderer(self.grid_size)
        elif self.obs_type == ObservationType.language:
            self.renderer = GridLangRenderer(self.grid_size)
        else:
            raise ValueError("No valid ObservationType provided.")

    def set_action_space(self, control_type):
        self.control_type = control_type
        if self.control_type == ControlType.egocentric:
            # For egocentric orientation, we use rotation/move semantics
            actions = [
                Action.ROTATE_LEFT,
                Action.ROTATE_RIGHT,
                Action.MOVE_FORWARD,
            ]
        elif self.control_type == ControlType.allocentric:
            # For allocentric orientation, the actions represent absolute directions
            actions = [
                Action.MOVE_UP,
                Action.MOVE_RIGHT,
                Action.MOVE_DOWN,
                Action.MOVE_LEFT,
            ]
        else:
            raise Exception("No valid ControlType provided.")
        if self.use_noop:
            actions.append(Action.NOOP)
        if self.manual_collect:
            actions.append(Action.COLLECT)
        self.action_list = actions
        self.action_space = spaces.Discrete(len(actions))

    def set_obs_space(self, obs_type, torch_obs, resolution):
        if isinstance(obs_type, str):
            obs_type = ObservationType(obs_type)
        self.obs_type = obs_type

        # Initialize the appropriate renderer
        self._init_renderers(resolution, torch_obs)

        # Get the observation space from the renderer
        self.obs_space = self.renderer.observation_space

    @property
    def observation(self):
        """Get the current observation from the environment."""
        return self.renderer.render(self)

    def reset(
        self,
        objects: Dict = None,
        agent_pos: list = None,
        episode_length: int = 100,
        random_start: bool = False,
        terminate_on_reward: bool = True,
        time_penalty: float = 0.0,
        stochasticity: float = 0.0,
        visible_walls: bool = True,
    ):
        """
        Resets the environment to its initial configuration.
        """
        # Reset basic state variables
        self._reset_state(
            episode_length,
            terminate_on_reward,
            time_penalty,
            stochasticity,
            visible_walls,
        )

        # Handle objects setup
        self.objects = self._setup_objects(objects)

        self.free_spots = self.make_free_spots(self.base_objects["walls"])

        # Set agent position
        self.agent_pos = self._setup_agent(agent_pos, random_start)

        return self.observation

    def _reset_state(
        self,
        episode_length,
        terminate_on_reward,
        time_penalty,
        stochasticity,
        visible_walls,
    ):
        """Helper method to reset all state variables."""
        self.done = False
        self.episode_time = 0
        self.orientation = 0
        self.looking = 0
        self.keys = 0
        self.time_penalty = time_penalty
        self.max_episode_time = episode_length
        self.terminate_on_reward = terminate_on_reward
        self.stochasticity = stochasticity
        self.visible_walls = visible_walls
        self.cached_objects = None

    def _setup_objects(self, objects: Dict = None) -> Dict:
        """Helper method to set up environment objects."""
        base_object = copy.deepcopy(self.base_objects)
        use_objects = copy.deepcopy(
            objects if objects is not None else self.template_objects
        )

        for key in use_objects:
            if key in base_object:
                base_object[key] = use_objects[key]
        if self.add_outer_walls:
            base_object["walls"] = add_outer(base_object["walls"], self.grid_size)
        return base_object

    def _setup_agent(self, agent_pos: list = None, random_start: bool = False) -> list:
        """Helper method to determine the agent's starting position."""
        if random_start:
            return self.get_free_spot()
        return agent_pos if agent_pos is not None else self.agent_start_pos

    def get_free_spot(self):
        return random.choice(self.free_spots)

    def make_free_spots(self, walls: list):
        return [
            [i, j]
            for i in range(self.grid_size)
            for j in range(self.grid_size)
            if [i, j] not in walls
        ]

    def render(self, provide=False, mode="human"):
        image = self.renderer.render(self)
        if mode == "human":
            plt.imshow(image)
            plt.axis("off")
            plt.show()
        if provide:
            return image

    def move_agent(self, direction: np.array):
        """
        Moves the agent in the given direction.
        """
        new_pos = np.array(self.agent_pos) + direction
        if self.check_target(new_pos):
            self.agent_pos = list(new_pos)

    def check_target(self, target: list):
        """
        Checks if the target is a valid (movable) position.
        Returns True if the target is valid, False otherwise.
        """
        target_tuple = tuple(target)
        target_list = list(target_tuple)
        x_check = -1 < target[0] < self.grid_size
        y_check = -1 < target[1] < self.grid_size

        if not (x_check and y_check):
            return False

        if target_list in self.objects["walls"]:
            return False

        if target_tuple in self.objects["doors"]:
            if self.keys > 0:
                self.objects["doors"].pop(target_tuple)
                self.keys -= 1
            else:
                return False

        return True

    def rotate(self, direction: int):
        """
        Rotates the agent orientation in the given direction.
        """
        self.orientation = (self.orientation + direction) % 4

    def step(self, action: int):
        """Steps the environment forward given an action."""
        if self.done:
            print("Episode finished. Please reset the environment.")
            return None, None, None, None

        # Handle stochastic actions
        if self.stochasticity > self.rng.rand():
            action = self.rng.randint(0, self.action_space.n)

        # Process action and determine if collection is allowed

        self._process_action(action)

        # Update state and calculate reward
        self.episode_time += 1
        reward = self._calculate_reward(action)

        # Process special tiles
        self._process_special_tiles()

        return self.observation, reward, self.done, {}

    def _determine_can_collect(self, action: int) -> bool:
        """Determines if collection is allowed based on action and manual_collect setting."""
        if not self.manual_collect:
            return True
        chosen_action = self.action_list[action]
        return chosen_action == Action.COLLECT

    def _process_action(self, action: int):
        """Unified action processing for both orientation types using semantic labels."""
        chosen_action = self.action_list[action]
        if self.control_type == ControlType.egocentric:
            if chosen_action == Action.ROTATE_LEFT:
                self.rotate(-1)
            elif chosen_action == Action.ROTATE_RIGHT:
                self.rotate(1)
            elif chosen_action == Action.MOVE_FORWARD:
                self.move_agent(self.direction_map[self.orientation])
            # No additional operation is needed for NOOP or COLLECT.
            if chosen_action in (
                Action.ROTATE_LEFT,
                Action.ROTATE_RIGHT,
                Action.MOVE_FORWARD,
            ):
                self.looking = self.orientation
        else:  # allocentric orientation
            if chosen_action in (
                Action.MOVE_UP,
                Action.MOVE_RIGHT,
                Action.MOVE_DOWN,
                Action.MOVE_LEFT,
            ):
                allocentric_mapping = {
                    Action.MOVE_UP: 0,
                    Action.MOVE_RIGHT: 1,
                    Action.MOVE_DOWN: 2,
                    Action.MOVE_LEFT: 3,
                }
                direction_idx = allocentric_mapping[chosen_action]
                self.looking = direction_idx
                self.move_agent(self.direction_map[direction_idx])
        # For NOOP or COLLECT the environment does not change the agent's position.

    def _calculate_reward(self, action: int) -> float:
        """Calculate reward based on action and current state."""
        reward = self.time_penalty if action != 4 else 0
        eval_pos = tuple(self.agent_pos)

        if eval_pos in self.objects["rewards"] and self._determine_can_collect(action):
            reward_info = self.objects["rewards"][eval_pos]
            reward += self._process_reward(reward_info)

        return reward

    def _process_reward(self, reward_info) -> float:
        if isinstance(reward_info, list):
            self.done = reward_info[2]
            return reward_info[0]
        self.done = self.terminate_on_reward
        return reward_info

    def _process_special_tiles(self):
        eval_pos = tuple(self.agent_pos)
        if eval_pos in self.objects["other"]:
            self.objects["other"].pop(eval_pos)

        if eval_pos in self.objects["keys"]:
            self.keys += 1
            self.objects["keys"].remove(eval_pos)

        if eval_pos in self.objects["warps"]:
            self.agent_pos = self.objects["warps"][eval_pos]

    def close(self) -> None:
        if self.obs_type == ObservationType.rendered_3d:
            self.renderer.close()
        return super().close()
