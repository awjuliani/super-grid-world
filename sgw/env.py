from typing import Dict
from gym import Env, spaces
import numpy as np
import random
from sgw.templates import generate_layout
from sgw.renderers.rend_2d import Grid2DRenderer
from sgw.renderers.rend_lang import GridLangRenderer
from sgw.renderers.rend_symbolic import GridSymbolicRenderer
from sgw.renderers.rend_ascii import GridASCIIRenderer
from sgw.renderers.rend_3d import Grid3DRenderer
from sgw.agent import Agent
import matplotlib.pyplot as plt
import copy
from sgw.enums import ObsType, ControlType, Action


class SuperGridWorld(Env):
    """
    Super Grid World. A 2D maze-like OpenAI gym compatible RL environment.
    """

    def __init__(
        self,
        grid_shape: tuple = (9, 9),
        obs_type: ObsType = ObsType.visual_2d,
        control_type: ControlType = ControlType.allocentric,
        seed: int = None,
        template_name: str = "empty",
        use_noop: bool = False,
        manual_interact: bool = False,
        torch_obs: bool = False,
        resolution: int = 256,
        add_outer_walls: bool = True,
        field_of_view: int = None,
        num_agents: int = 1,
    ):
        # Initialize basic attributes
        self._init_basic_attrs(
            seed, use_noop, manual_interact, add_outer_walls, field_of_view, num_agents
        )

        # Setup grid and walls
        self._init_grid(template_name, grid_shape)

        # Setup action and observation spaces
        self.set_action_space(control_type)
        self.set_obs_space(obs_type, torch_obs, resolution)

    def _init_basic_attrs(
        self,
        seed,
        use_noop,
        manual_interact,
        add_outer_walls,
        field_of_view,
        num_agents,
    ):
        self.rng = np.random.RandomState(seed)
        self.use_noop = use_noop
        self.manual_interact = manual_interact
        self.add_outer_walls = add_outer_walls
        self.field_of_view = field_of_view
        self.num_agents = num_agents
        self.agents = [None] * num_agents  # List to store multiple agents

    def _init_grid(self, template, grid_shape):
        # grid_shape is (height, width)
        self.agent_start_pos, self.template_objects = generate_layout(
            template, grid_shape[0], grid_shape[1], self.add_outer_walls
        )
        self.grid_shape = grid_shape

    def _init_renderers(self, resolution, torch_obs):
        """Initialize renderers based on observation type."""
        renderer_map = {
            ObsType.visual_2d: lambda: Grid2DRenderer(
                grid_shape=self.grid_shape,
                window_size=self.field_of_view,
                resolution=resolution,
                torch_obs=torch_obs,
            ),
            ObsType.symbolic: lambda: GridSymbolicRenderer(
                grid_shape=self.grid_shape,
                window_size=self.field_of_view,
            ),
            ObsType.visual_3d: lambda: Grid3DRenderer(
                resolution=resolution,
                torch_obs=torch_obs,
                field_of_view=self.field_of_view,
            ),
            ObsType.ascii: lambda: GridASCIIRenderer(
                grid_shape=self.grid_shape,
            ),
            ObsType.language: lambda: GridLangRenderer(
                grid_shape=self.grid_shape,
            ),
        }

        if self.obs_type not in renderer_map:
            raise ValueError("No valid ObservationType provided.")

        self.renderer = renderer_map[self.obs_type]()
        self.state_renderer = renderer_map[ObsType.visual_2d]()
        self.state_renderer.window_size = None

    def set_action_space(self, control_type):
        """Set up the action space based on control type and optional actions."""
        self.control_type = control_type
        # Get valid actions using the Action enum helper
        self.valid_actions = Action.get_actions_for_control(
            control_type, self.use_noop, self.manual_interact
        )
        # Create action space for each agent
        self.action_space = spaces.Tuple(
            [spaces.Discrete(len(self.valid_actions))] * self.num_agents
        )

    def set_obs_space(self, obs_type, torch_obs, resolution):
        if isinstance(obs_type, str):
            obs_type = ObsType(obs_type)
        self.obs_type = obs_type

        # Initialize the appropriate renderer
        self._init_renderers(resolution, torch_obs)

        # Create observation space for each agent
        single_obs_space = self.renderer.observation_space
        self.observation_space = spaces.Tuple([single_obs_space] * self.num_agents)

    @property
    def observation(self):
        """Get the current observation from the environment for all agents."""
        return [self.renderer.render(self, agent_idx=i) for i in range(self.num_agents)]

    def reset(
        self,
        objects: Dict = None,
        agent_positions: list = None,
        episode_length: int = 100,
        random_start: bool = False,
        time_penalty: float = 0.0,
        stochasticity: float = 0.0,
    ):
        """
        Resets the environment to its initial configuration.
        """
        # Reset basic state variables
        self._reset_state(
            episode_length,
            time_penalty,
            stochasticity,
        )

        # Handle objects setup
        self._setup_objects(objects)

        # Set agent positions
        self._setup_agents(agent_positions, random_start)

        return self.observation

    def _reset_state(
        self,
        episode_length,
        time_penalty,
        stochasticity,
    ):
        """Helper method to reset all state variables."""
        self.episode_time = 0
        self.time_penalty = time_penalty
        self.max_episode_time = episode_length
        self.stochasticity = stochasticity
        self.cached_objects = None
        self.events = []  # Initialize empty events list

    def _setup_objects(self, objects: Dict = None) -> Dict:
        """Helper method to set up environment objects."""
        self.objects = copy.deepcopy(
            objects if objects is not None else self.template_objects
        )

    def _setup_agents(self, agent_positions: list = None, random_start: bool = False):
        """Helper method to determine the agents' starting positions."""
        if agent_positions is None:
            agent_positions = [None] * self.num_agents
        elif len(agent_positions) != self.num_agents:
            raise ValueError(
                f"Expected {self.num_agents} agent positions, got {len(agent_positions)}"
            )

        free_spots = self.free_spots
        for i in range(self.num_agents):
            if random_start:
                if not free_spots:
                    raise ValueError("Not enough free spots for all agents")
                pos = random.choice(free_spots)
                free_spots.remove(pos)
            else:
                pos = (
                    agent_positions[i]
                    if agent_positions[i] is not None
                    else self.agent_start_pos
                )
            self.agents[i] = Agent(pos, field_of_view=self.field_of_view)

    @property
    def free_spots(self):
        # Create set of all grid positions
        all_positions = {
            (i, j) for i in range(self.grid_shape[0]) for j in range(self.grid_shape[1])
        }
        # Create set of occupied positions
        occupied_positions = {
            tuple(obj.pos) for obj_type in self.objects.values() for obj in obj_type
        }
        # Return difference as list of lists
        return [list(pos) for pos in all_positions - occupied_positions]

    def render(self, provide=False, mode="human"):
        image = self.state_renderer.render(self, is_state_view=True)
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

    def check_target(self, target: list, obstacles_only: bool = True):
        """
        Checks if the target is a valid (movable) position.
        Returns True if the target is valid, False otherwise.

        Args:
            target: The target position to check
            obstacles_only: If True, only obstacle objects block movement.
                           If False, any object blocks movement.
        """
        x_check = -1 < target[0] < self.grid_shape[0]
        y_check = -1 < target[1] < self.grid_shape[1]

        if not (x_check and y_check):
            return False

        # Convert numpy array to list for comparison
        target = list(map(int, target))

        # Check all objects
        for obj_type in self.objects.values():
            obj = next((o for o in obj_type if o == target), None)
            if obj:
                # If obstacles_only is True, only obstacle objects block movement
                # If obstacles_only is False, any object blocks movement
                if obstacles_only:
                    if obj.obstacle:
                        return False
                else:
                    return False

        # Check if any other agent is at the target position
        for agent in self.agents:
            if agent is not None and agent.get_position() == target:
                return False

        return True

    def step(self, actions):
        """Steps the environment forward given actions for all agents."""
        if any(agent.done for agent in self.agents):
            print("Episode finished. Please reset the environment.")
            return None, None, None, None

        if not isinstance(actions, (list, tuple)) or len(actions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {actions}")

        rewards = []
        dones = []
        self.events = []  # Reset events list at start of step

        # Process each agent's action
        for i, action in enumerate(actions):
            # Handle stochastic actions
            if self.stochasticity > self.rng.rand():
                action = self.rng.randint(0, len(self.valid_actions))

            # Process action and determine if interaction is allowed
            self._move_agent(action, i)

            # Process object interactions
            self._object_interactions(action, i)

            rewards.append(self.agents[i].reward)
            dones.append(self.agents[i].done)

        # Update all objects
        for obj_type in self.objects.values():
            for obj in obj_type:
                event = obj.step(self)
                if event:
                    self.events.append(event)

        # Update time
        self.episode_time += 1

        return self.observation, rewards, dones, {"events": self.events}

    def _determine_can_interact(self, action: int) -> bool:
        """Determines if interaction is allowed based on action and manual_interact setting."""
        if not self.manual_interact:
            return True
        chosen_action = self.valid_actions[action]
        return chosen_action == Action.INTERACT

    def _find_object_at(self, pos):
        """Find the first object at the given position."""
        for obj_type in self.objects.values():
            obj = next((o for o in obj_type if o == pos), None)
            if obj:
                return obj
        return None

    def _move_agent(self, action: int, agent_idx: int):
        """Process the action and move the specified agent if applicable."""
        chosen_action = self.valid_actions[action]
        agent = self.agents[agent_idx]
        direction = agent.process_action(chosen_action, self.control_type)
        if direction is None:
            return

        target_pos = list((np.array(agent.pos) + direction).tolist())

        # Check grid bounds
        if not (
            -1 < target_pos[0] < self.grid_shape[0]
            and -1 < target_pos[1] < self.grid_shape[1]
        ):
            return

        # Handle any pre-step interactions
        obj = self._find_object_at(target_pos)
        if obj and hasattr(obj, "pre_step_interaction"):
            allowed, message = obj.pre_step_interaction(agent, direction, self)
            if message:
                self.events.append(message)
            if not allowed:
                return

        # Move agent if target is valid
        if self.check_target(target_pos):
            agent.move(direction)

    def _object_interactions(self, action: int, agent_idx: int):
        """Process interactions with objects at the specified agent's position."""
        agent = self.agents[agent_idx]
        agent_pos = agent.get_position()
        agent.reward = self.time_penalty

        # Check all object types for interactions
        for obj_type in self.objects.values():
            obj = next((o for o in obj_type if o == agent_pos), None)
            if obj and self._determine_can_interact(action):
                # Interact with the object and handle any events
                event = obj.interact(agent)
                if event:
                    self.events.append(event)
                # Remove object if specified
                if obj.consumable:
                    obj_type.remove(obj)

    def close(self) -> None:
        self.renderer.close()
        return super().close()

    @property
    def grid_width(self):
        return self.grid_shape[1]  # Second dimension is width

    @property
    def grid_height(self):
        return self.grid_shape[0]  # First dimension is height
