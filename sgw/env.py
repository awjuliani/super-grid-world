from typing import Dict
from gym import Env, spaces
import numpy as np
import random
import enum
from sgw.templates import generate_layout
from sgw.renderers.rend_2d import Grid2DRenderer
from sgw.renderers.rend_lang import GridLangRenderer
from sgw.renderers.rend_symbolic import GridSymbolicRenderer
from sgw.renderers.rend_ascii import GridASCIIRenderer
from sgw.renderers.rend_3d import Grid3DRenderer
from sgw.agent import Agent
import matplotlib.pyplot as plt
import copy


class ObsType(enum.Enum):
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
    """

    def __init__(
        self,
        template_name: str = "empty",
        grid_size: int = 11,
        obs_type: ObsType = ObsType.visual,
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
        self._init_grid(template_name, grid_size)

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
        self.agent = None  # Will be initialized in reset

    def _init_grid(self, template, size):
        self.agent_start_pos, self.template_objects = generate_layout(
            template, size, self.add_outer_walls
        )
        self.grid_size = size

    def _init_renderers(self, resolution, torch_obs):
        """Initialize renderers based on observation type."""
        renderer_map = {
            ObsType.visual: lambda: Grid2DRenderer(
                self.grid_size, resolution=resolution, torch_obs=torch_obs
            ),
            ObsType.visual_window: lambda: Grid2DRenderer(
                self.grid_size,
                window_size=2,
                resolution=resolution,
                torch_obs=torch_obs,
            ),
            ObsType.visual_window_tight: lambda: Grid2DRenderer(
                self.grid_size,
                window_size=1,
                resolution=resolution,
                torch_obs=torch_obs,
            ),
            ObsType.symbolic: lambda: GridSymbolicRenderer(self.grid_size),
            ObsType.symbolic_window: lambda: GridSymbolicRenderer(
                self.grid_size, window_size=5
            ),
            ObsType.symbolic_window_tight: lambda: GridSymbolicRenderer(
                self.grid_size, window_size=3
            ),
            ObsType.rendered_3d: lambda: Grid3DRenderer(
                resolution=resolution, torch_obs=torch_obs
            ),
            ObsType.ascii: lambda: GridASCIIRenderer(self.grid_size),
            ObsType.language: lambda: GridLangRenderer(self.grid_size),
        }

        if self.obs_type not in renderer_map:
            raise ValueError("No valid ObservationType provided.")

        self.renderer = renderer_map[self.obs_type]()

    def set_action_space(self, control_type):
        self.control_type = control_type
        if self.control_type == ControlType.egocentric:
            # For egocentric orientation, we use rotation/move semantics
            self.valid_actions = [
                Action.ROTATE_LEFT,
                Action.ROTATE_RIGHT,
                Action.MOVE_FORWARD,
            ]
        elif self.control_type == ControlType.allocentric:
            # For allocentric orientation, the actions represent absolute directions
            self.valid_actions = [
                Action.MOVE_UP,
                Action.MOVE_RIGHT,
                Action.MOVE_DOWN,
                Action.MOVE_LEFT,
            ]
        else:
            raise Exception("No valid ControlType provided.")
        if self.use_noop:
            self.valid_actions.append(Action.NOOP)
        if self.manual_collect:
            self.valid_actions.append(Action.COLLECT)
        self.action_space = spaces.Discrete(len(self.valid_actions))

    def set_obs_space(self, obs_type, torch_obs, resolution):
        if isinstance(obs_type, str):
            obs_type = ObsType(obs_type)
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
            time_penalty,
            stochasticity,
            visible_walls,
        )

        # Handle objects setup
        self._setup_objects(objects)

        # Set agent position
        self._setup_agent(agent_pos, random_start)

        return self.observation

    def _reset_state(
        self,
        episode_length,
        time_penalty,
        stochasticity,
        visible_walls,
    ):
        """Helper method to reset all state variables."""
        self.episode_time = 0
        self.time_penalty = time_penalty
        self.max_episode_time = episode_length
        self.stochasticity = stochasticity
        self.visible_walls = visible_walls
        self.cached_objects = None

    def _setup_objects(self, objects: Dict = None) -> Dict:
        """Helper method to set up environment objects."""
        self.objects = copy.deepcopy(
            objects if objects is not None else self.template_objects
        )

    def _setup_agent(self, agent_pos: list = None, random_start: bool = False) -> list:
        """Helper method to determine the agent's starting position."""
        pos = (
            random.choice(self.free_spots)
            if random_start
            else (agent_pos if agent_pos is not None else self.agent_start_pos)
        )
        self.agent = Agent(pos)
        return pos

    @property
    def free_spots(self):
        # Create set of all grid positions
        all_positions = {
            (i, j) for i in range(self.grid_size) for j in range(self.grid_size)
        }
        # Create set of occupied positions
        occupied_positions = {
            tuple(obj.pos) for obj_type in self.objects.values() for obj in obj_type
        }
        # Return difference as list of lists
        return [list(pos) for pos in all_positions - occupied_positions]

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
        Moves the agent in the given direction if valid.
        """
        new_pos = np.array(self.agent.pos) + direction
        if self.check_target(new_pos):
            self.agent.move(direction)

    def check_target(self, target: list):
        """
        Checks if the target is a valid (movable) position.
        Returns True if the target is valid, False otherwise.
        """
        x_check = -1 < target[0] < self.grid_size
        y_check = -1 < target[1] < self.grid_size

        if not (x_check and y_check):
            return False

        # Convert numpy array to list for comparison
        target = list(map(int, target))

        # Check all objects for whether they are an obstacle
        for obj_type in self.objects.values():
            obj = next((o for o in obj_type if o == target), None)
            if obj and obj.obstacle:
                return False

        return True

    def step(self, action: int):
        """Steps the environment forward given an action."""
        if self.agent.done:
            print("Episode finished. Please reset the environment.")
            return None, None, None, None

        # Handle stochastic actions
        if self.stochasticity > self.rng.rand():
            action = self.rng.randint(0, self.action_space.n)

        # Process action and determine if collection is allowed
        self._move_agent(action)

        # Process object interactions
        self._object_interactions(action)

        # Update time
        self.episode_time += 1

        return self.observation, self.agent.reward, self.agent.done, {}

    def _determine_can_collect(self, action: int) -> bool:
        """Determines if collection is allowed based on action and manual_collect setting."""
        if not self.manual_collect:
            return True
        chosen_action = self.valid_actions[action]
        return chosen_action == Action.COLLECT

    def _move_agent(self, action: int):
        """Process the action and move the agent if applicable."""
        # Process action
        chosen_action = self.valid_actions[action]
        direction = self.agent.process_action(chosen_action, self.control_type)
        if direction is not None and self.check_target(
            np.array(self.agent.pos) + direction
        ):
            self.agent.move(direction)

    def _object_interactions(self, action: int):
        """Process interactions with objects at the agent's position."""
        agent_pos = self.agent.get_position()
        self.agent.reward = self.time_penalty

        # Check all object types for interactions
        for obj_type in self.objects.values():
            obj = next((o for o in obj_type if o == agent_pos), None)
            if obj and self._determine_can_collect(action):
                # Interact with the object
                obj.interact(self.agent)
                # Remove object if specified
                if obj.consumable:
                    obj_type.remove(obj)

    def close(self) -> None:
        self.renderer.close()
        return super().close()
