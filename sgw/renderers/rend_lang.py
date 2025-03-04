import numpy as np
from sgw.renderers.rend_interface import RendererInterface
from typing import Any, Tuple, List, Dict
from gym import spaces
from sgw.enums import ControlType


class GridLangRenderer(RendererInterface):
    def __init__(self, grid_shape: Tuple[int, int]):
        self.grid_shape = grid_shape

    @property
    def observation_space(self) -> spaces.Space:
        """Return the observation space for language observations."""
        return spaces.Discrete(1)

    def _get_region(self, pos):
        """Helper method to determine region in the maze."""
        third_width = self.grid_shape[0] / 3
        third_height = self.grid_shape[1] / 3
        x, y = pos[1], pos[0]

        # Determine vertical region (north/center/south)
        if y < third_height:
            v_region = "north"
        elif y > 2 * third_height:
            v_region = "south"
        else:
            v_region = "center"

        # Determine horizontal region (west/center/east)
        if x < third_width:
            h_region = "west"
        elif x > 2 * third_width:
            h_region = "east"
        else:
            h_region = "center"

        # Combine regions based on position
        if v_region == "center" and h_region == "center":
            return "center"
        elif v_region == "center":
            return f"center {h_region}"
        elif h_region == "center":
            return f"center {v_region}"
        else:
            return f"{v_region}-{h_region}"

    def _describe_position_relative_to_walls(self, pos) -> str:
        """Describe a position relative to the cardinal walls of the maze."""
        row, col = int(pos[0]), int(pos[1])

        # Calculate distances to each wall
        # grid_shape is (height, width) where height is rows and width is columns
        # In grid coordinates: row=0 is north, max row is south, col=0 is west, max col is east
        north_dist = row  # Distance from north wall (row 0)
        south_dist = self.grid_shape[0] - 1 - row  # Distance from south wall (max row)
        west_dist = col  # Distance from west wall (col 0)
        east_dist = self.grid_shape[1] - 1 - col  # Distance from east wall (max col)

        # Group walls by axis to avoid redundancy
        ns_axis = [("north", north_dist), ("south", south_dist)]
        ew_axis = [("west", west_dist), ("east", east_dist)]

        # Sort each axis by distance
        ns_axis.sort(key=lambda x: x[1])
        ew_axis.sort(key=lambda x: x[1])

        # Get the closest wall from each axis
        closest_ns_wall, closest_ns_dist = ns_axis[0]
        closest_ew_wall, closest_ew_dist = ew_axis[0]

        # Format the description - always include distances from both axes
        if closest_ns_dist == 0 and closest_ew_dist == 0:
            # Position is at a corner
            return f"at the {closest_ns_wall}-{closest_ew_wall} corner of the maze"
        elif closest_ns_dist == 0:
            # Position is at a north or south wall
            return f"at the {closest_ns_wall} wall, {closest_ew_dist} {'meter' if closest_ew_dist == 1 else 'meters'} from the {closest_ew_wall} wall"
        elif closest_ew_dist == 0:
            # Position is at an east or west wall
            return f"at the {closest_ew_wall} wall, {closest_ns_dist} {'meter' if closest_ns_dist == 1 else 'meters'} from the {closest_ns_wall} wall"
        else:
            # Not at any wall, describe distances to closest wall from each axis
            return f"{closest_ns_dist} {'meter' if closest_ns_dist == 1 else 'meters'} from the {closest_ns_wall} wall and {closest_ew_dist} {'meter' if closest_ew_dist == 1 else 'meters'} from the {closest_ew_wall} wall"

    def _describe_position_relative_to_north_west(self, pos) -> str:
        """Describe a position relative to only the north and west walls of the maze."""
        row, col = int(pos[0]), int(pos[1])

        # Calculate distances to north and west walls
        # In grid coordinates: row=0 is north, col=0 is west
        north_dist = row  # Distance from north wall (row 0)
        west_dist = col  # Distance from west wall (col 0)

        # Format the description - always include distances from both north and west walls
        if north_dist == 0 and west_dist == 0:
            # Position is at the north-west corner
            return f"at the north-west corner of the maze"
        elif north_dist == 0:
            # Position is at the north wall
            return f"at the north wall, {west_dist} {'meter' if west_dist == 1 else 'meters'} from the west wall"
        elif west_dist == 0:
            # Position is at the west wall
            return f"at the west wall, {north_dist} {'meter' if north_dist == 1 else 'meters'} from the north wall"
        else:
            # Not at any wall, describe distances to north and west walls
            return f"{north_dist} {'meter' if north_dist == 1 else 'meters'} from the north wall and {west_dist} {'meter' if west_dist == 1 else 'meters'} from the west wall"

    def _cardinal_to_egocentric(self, cardinal_dir: str, agent_looking: int) -> str:
        """Convert cardinal direction to egocentric direction based on agent's orientation.
        agent_looking: 0=North, 1=East, 2=South, 3=West
        """
        # Map of cardinal directions to their angle in degrees (clockwise from north)
        cardinal_angles = {
            "north": 0,
            "north-east": 45,
            "east": 90,
            "south-east": 135,
            "south": 180,
            "south-west": 225,
            "west": 270,
            "north-west": 315,
            "same position": -1,
        }

        if cardinal_dir == "same position":
            return "at the same position"

        # Convert agent's looking direction to degrees
        agent_angle = agent_looking * 90

        # Get angle of cardinal direction
        dir_angle = cardinal_angles[cardinal_dir]

        # Calculate relative angle
        relative_angle = (dir_angle - agent_angle) % 360

        # Convert relative angle to egocentric direction
        if relative_angle == 0:
            return "in front of"
        elif relative_angle == 180:
            return "behind"
        elif 0 < relative_angle < 180:
            return "to the right of"
        else:
            return "to the left of"

    def _get_direction_and_distance(
        self, obj_pos, agent_pos, control_type=None, agent_looking=None
    ):
        """Helper method to get direction and distance.
        In allocentric mode: returns cardinal directions
        In egocentric mode: returns relative directions (left/right/front/behind)
        """
        diff = np.array(obj_pos) - agent_pos

        # Calculate Euclidean distance
        distance = np.floor(2 * np.sqrt(diff[0] ** 2 + diff[1] ** 2)) / 2

        if distance == 0:
            return "same position", 0

        # Get cardinal direction first
        y_dir = "north" if diff[0] < 0 else "south" if diff[0] > 0 else ""
        x_dir = "east" if diff[1] > 0 else "west" if diff[1] < 0 else ""

        cardinal_dir = (
            f"{y_dir}-{x_dir}"
            if (y_dir and x_dir)
            else (y_dir or x_dir or "same position")
        )

        # If in egocentric mode, convert to relative direction
        if control_type == ControlType.egocentric and agent_looking is not None:
            direction = self._cardinal_to_egocentric(cardinal_dir, agent_looking)
        else:
            direction = cardinal_dir

        return direction, int(distance)

    def _is_in_visible_window(self, obj_pos, agent_pos, fov, looking):
        """Determine if an object at obj_pos is within the visible window when the agent is in egocentric mode.
        The window is defined as a rectangle in front of the agent:
        - North (0): rows [agent_row - fov, agent_row] and columns [agent_col - fov, agent_col + fov]
        - East  (1): rows [agent_row - fov, agent_row + fov] and columns [agent_col, agent_col + fov]
        - South (2): rows [agent_row, agent_row + fov] and columns [agent_col - fov, agent_col + fov]
        - West  (3): rows [agent_row - fov, agent_row + fov] and columns [agent_col - fov, agent_col]
        """
        agent_row, agent_col = int(agent_pos[0]), int(agent_pos[1])
        obj_row, obj_col = int(obj_pos[0]), int(obj_pos[1])
        if looking == 0:  # North
            return (agent_row - fov <= obj_row <= agent_row) and (
                agent_col - fov <= obj_col <= agent_col + fov
            )
        elif looking == 1:  # East
            return (agent_col <= obj_col <= agent_col + fov) and (
                agent_row - fov <= obj_row <= agent_row + fov
            )
        elif looking == 2:  # South
            return (agent_row <= obj_row <= agent_row + fov) and (
                agent_col - fov <= obj_col <= agent_col + fov
            )
        elif looking == 3:  # West
            return (agent_col - fov <= obj_col <= agent_col) and (
                agent_row - fov <= obj_row <= agent_row + fov
            )
        return True

    # First-person specific methods
    def _get_first_person_object_descriptions(
        self, objects, agent_pos, field_of_view, agent_looking=None, control_type=None
    ) -> List[str]:
        """Generate first-person descriptions for objects."""
        descriptions = []
        for obj in objects:
            direction, distance = self._get_direction_and_distance(
                obj.pos,
                agent_pos,
                control_type=control_type,
                agent_looking=agent_looking,
            )

            # Skip objects beyond field of view
            if distance > field_of_view:
                continue

            # In egocentric mode, also check if in visible window
            if agent_looking is not None and control_type == ControlType.egocentric:
                if not self._is_in_visible_window(
                    obj.pos, agent_pos, field_of_view, agent_looking
                ):
                    continue

            # Format description
            if direction == "same position":
                descriptions.append(f"You see a {obj.name} at your position.")
            else:
                descriptions.append(
                    f"You see a {obj.name} {direction} of you, {distance} "
                    f"{'meter' if distance == 1 else 'meters'} away."
                )

        return descriptions

    def _get_first_person_agent_descriptions(
        self,
        env,
        current_agent_idx,
        agent_pos,
        field_of_view,
        agent_looking=None,
        control_type=None,
    ) -> List[str]:
        """Generate first-person descriptions for other agents in the environment."""
        descriptions = []
        direction_map = {0: "north", 1: "east", 2: "south", 3: "west"}

        for i, other_agent in enumerate(env.agents):
            # Skip the current agent
            if i == current_agent_idx:
                continue

            direction, distance = self._get_direction_and_distance(
                other_agent.pos,
                agent_pos,
                control_type=control_type,
                agent_looking=agent_looking,
            )

            # Skip agents beyond field of view
            if distance > field_of_view:
                continue

            # In egocentric mode, also check if in visible window
            if agent_looking is not None and control_type == ControlType.egocentric:
                if not self._is_in_visible_window(
                    other_agent.pos, agent_pos, field_of_view, agent_looking
                ):
                    continue

            # Format description
            name = other_agent.name

            if direction == "same position":
                descriptions.append(f"You see an agent named {name} at your position.")
            else:
                descriptions.append(
                    f"You see an agent named {name} {direction} of you, {distance} "
                    f"{'meter' if distance == 1 else 'meters'} away."
                )

            # Add information about the agent's orientation if in egocentric mode
            if (
                hasattr(other_agent, "looking")
                and env.control_type == ControlType.egocentric
            ):
                facing = direction_map.get(other_agent.looking, "unknown direction")
                descriptions.append(f"{name} is facing {facing}.")

        return descriptions

    def _get_first_person_base_description(self, agent, agent_pos) -> str:
        """Generate the base description for first-person perspective."""
        # Get agent looking direction if applicable
        orientation_desc = ""
        if (
            hasattr(agent, "looking")
            and hasattr(agent, "control_type")
            and agent.control_type == ControlType.egocentric
        ):
            direction_map = {0: "north", 1: "east", 2: "south", 3: "west"}
            orientation_desc = f"You are facing {direction_map[agent.looking]}. You cannot see objects behind you."
        # Add position information for allocentric mode or when control_type is not specified
        else:
            row, col = int(agent_pos[0]), int(agent_pos[1])
            north_dist = row  # Distance from north wall (row 0)
            west_dist = col  # Distance from west wall (col 0)
            orientation_desc = f"You are {north_dist} {'meter' if north_dist == 1 else 'meters'} from the north wall and {west_dist} {'meter' if west_dist == 1 else 'meters'} from the west wall."

        # Create inventory description
        if agent.inventory:
            inventory_items = {}
            for item in agent.inventory:
                inventory_items[item.name] = inventory_items.get(item.name, 0) + 1
            inventory_desc = ", ".join(
                f"{count} {name}" + ("s" if count > 1 else "")
                for name, count in inventory_items.items()
            )
            inventory_text = f"You are carrying {inventory_desc}."
        else:
            inventory_text = "You are not carrying anything."

        # Base description
        return (
            f"You are in a {self.grid_shape[0]}x{self.grid_shape[1]} meter maze. "
            f"\n{orientation_desc}"
            f"\n{inventory_text} "
            f"\nYou can see all objects up to {agent.field_of_view} {'meter' if agent.field_of_view == 1 else 'meters'} away. "
        )

    # Third-person specific methods
    def _get_third_person_object_descriptions(self, objects) -> List[str]:
        """Generate descriptions for objects using cardinal directions."""
        descriptions = []
        # Group objects by type for more concise descriptions
        objects_by_type = {}

        for obj in objects:
            obj_type = obj.name
            if obj_type not in objects_by_type:
                objects_by_type[obj_type] = []
            objects_by_type[obj_type].append(obj)

        # Generate descriptions for each object type
        for obj_type, obj_list in objects_by_type.items():
            if len(obj_list) == 1:
                obj = obj_list[0]
                position_desc = self._describe_position_relative_to_north_west(obj.pos)
                descriptions.append(f"There is a {obj_type} {position_desc}.")
            else:
                # For multiple objects of the same type, list all positions
                positions = [
                    self._describe_position_relative_to_north_west(obj.pos)
                    for obj in obj_list
                ]
                if len(positions) == 2:
                    pos_str = f"{positions[0]} and {positions[1]}"
                else:
                    pos_str = ", ".join(positions[:-1]) + f", and {positions[-1]}"
                descriptions.append(
                    f"There are {len(obj_list)} {obj_type}s located {pos_str}."
                )

        return descriptions

    def _get_third_person_boundary_descriptions(self, agent_pos) -> List[str]:
        """Describe boundaries from third-person perspective with cardinal directions."""
        descriptions = []

        # Check each boundary
        if agent_pos[0] == 0:  # North wall
            descriptions.append(f"The agent is at the north wall of the maze.")
        elif agent_pos[0] == self.grid_shape[1] - 1:  # South wall
            descriptions.append(f"The agent is at the south wall of the maze.")

        if agent_pos[1] == 0:  # West wall
            descriptions.append(f"The agent is at the west wall of the maze.")
        elif agent_pos[1] == self.grid_shape[0] - 1:  # East wall
            descriptions.append(f"The agent is at the east wall of the maze.")

        return descriptions

    def _get_third_person_agent_descriptions(self, env) -> List[str]:
        """Generate descriptions for all agents using cardinal directions."""
        descriptions = []
        direction_map = {0: "north", 1: "east", 2: "south", 3: "west"}

        for i, agent in enumerate(env.agents):
            name = agent.name
            position_desc = self._describe_position_relative_to_north_west(agent.pos)
            descriptions.append(f"\nThere is an agent named {name}.")

            # Include direction the agent is facing if in egocentric mode
            if (
                hasattr(agent, "looking")
                and hasattr(env, "control_type")
                and env.control_type == ControlType.egocentric
            ):
                facing = direction_map.get(agent.looking, "unknown direction")
                descriptions.append(f"{name} is {position_desc} facing {facing}.")
            else:
                descriptions.append(f"{name} is {position_desc}.")

            # Add explicit north/west coordinates for all modes
            row, col = int(agent.pos[0]), int(agent.pos[1])
            descriptions.append(
                f"{name} is {row} {'meter' if row == 1 else 'meters'} from the north wall and {col} {'meter' if col == 1 else 'meters'} from the west wall."
            )

            descriptions.append(
                f"{name} can see all objects up to {agent.field_of_view} {'meter' if agent.field_of_view == 1 else 'meters'} away."
            )
            if (
                hasattr(agent, "control_type")
                and agent.control_type == ControlType.egocentric
            ):
                descriptions.append(f"{name} cannot see objects behind them.")

            # Add inventory information if available
            if hasattr(agent, "inventory") and agent.inventory:
                inventory_items = {}
                for item in agent.inventory:
                    inventory_items[item.name] = inventory_items.get(item.name, 0) + 1
                inventory_desc = ", ".join(
                    f"{count} {name}" + ("s" if count > 1 else "")
                    for name, count in inventory_items.items()
                )
                descriptions.append(f"{name} is carrying {inventory_desc}.")

        return descriptions

    def _generate_grid_overview(self, env) -> List[str]:
        """Generate a high-level overview of the grid for third-person observations."""
        # Create a summary of object counts and their distribution
        object_counts = {}
        region_distribution = {
            "north-west": [],
            "north": [],
            "north-east": [],
            "center west": [],
            "center": [],
            "center east": [],
            "south-west": [],
            "south": [],
            "south-east": [],
        }

        # Count objects by type
        for obj_type, obj_list in env.objects.items():
            if obj_list:
                object_counts[obj_type] = len(obj_list)

                # Track distribution across regions
                for obj in obj_list:
                    region = self._get_region(obj.pos)
                    if region in region_distribution:
                        if obj_type not in region_distribution[region]:
                            region_distribution[region].append(obj_type)

        # Generate overview text
        overview = ["Grid overview:"]

        # Add object counts
        if object_counts:
            count_strs = [
                f"{count} {obj_type}" + ("s" if count > 1 else "")
                for obj_type, count in object_counts.items()
            ]
            overview.append("The environment contains " + ", ".join(count_strs) + ".")
        else:
            overview.append("The environment contains no objects.")

        # Add region distribution information
        populated_regions = {
            region: objs for region, objs in region_distribution.items() if objs
        }
        if populated_regions:
            overview.append("Object distribution by region:")
            for region, obj_types in populated_regions.items():
                overview.append(f"- {region} region: {', '.join(obj_types)}")

        # Add all agent positions
        agent_positions = []
        for i, agent in enumerate(env.agents):
            pos_desc = self._describe_position_relative_to_north_west(agent.pos)
            row, col = int(agent.pos[0]), int(agent.pos[1])
            north_west_desc = f"{row} {'meter' if row == 1 else 'meters'} from north, {col} {'meter' if col == 1 else 'meters'} from west"
            agent_positions.append(f"Agent {i}: {pos_desc} ({north_west_desc})")

        overview.append("Agent positions: " + ", ".join(agent_positions))

        return overview

    def _get_third_person_base_description(self, env, add_overview=False) -> str:
        """Generate the base description for third-person perspective."""
        # Base description of the environment
        base_desc = f"There is a {self.grid_shape[0]}x{self.grid_shape[1]} meter maze environment, in which:"

        if add_overview:
            # Add grid overview
            grid_overview = self._generate_grid_overview(env)
            base_desc += "\n\n" + "\n".join(grid_overview)

        return base_desc

    def _make_first_person_obs(self, env: Any, agent_idx: int = 0) -> str:
        """Generate observations from first-person perspective."""
        agent = env.agents[agent_idx]
        agent_pos = np.array(agent.pos)

        # Get base description
        base_description = self._get_first_person_base_description(agent, agent_pos)

        # Process all objects
        all_objects = []
        for obj_list in env.objects.values():
            all_objects.extend(obj_list)

        # Get object descriptions
        object_descriptions = []
        if all_objects:
            agent_looking = (
                agent.looking if env.control_type == ControlType.egocentric else None
            )
            object_descriptions = self._get_first_person_object_descriptions(
                all_objects,
                agent_pos,
                agent.field_of_view,
                agent_looking=agent_looking,
                control_type=env.control_type,
            )

        # Get descriptions of other agents
        agent_descriptions = []
        if len(env.agents) > 1:  # Only if there are other agents
            agent_looking = (
                agent.looking if env.control_type == ControlType.egocentric else None
            )
            agent_descriptions = self._get_first_person_agent_descriptions(
                env,
                agent_idx,
                agent_pos,
                agent.field_of_view,
                agent_looking=agent_looking,
                control_type=env.control_type,
            )

        # Combine all descriptions
        all_descriptions = object_descriptions + agent_descriptions

        if not all_descriptions:
            return f"{base_description}\nThere are no visible objects, agents, or walls near you."
        else:
            return f"{base_description}\n\n" + "\n".join(all_descriptions)

    def _make_third_person_obs(self, env: Any) -> str:
        """Generate observations from third-person perspective.

        This provides an objective view of the entire environment without
        focusing on any particular agent.
        """
        # Get base description
        base_description = self._get_third_person_base_description(env)

        # Process all objects
        all_objects = []
        for obj_list in env.objects.values():
            all_objects.extend(obj_list)

        # Get object and agent descriptions
        all_descriptions = []

        if all_objects:
            all_descriptions.extend(
                self._get_third_person_object_descriptions(all_objects)
            )

        # Add descriptions of all agents
        all_descriptions.extend(self._get_third_person_agent_descriptions(env))

        if not all_descriptions:
            return (
                f"{base_description}\nThere are no objects or walls in the environment."
            )
        else:
            return f"{base_description}\n\n" + "\n".join(all_descriptions)

    def make_language_obs(
        self, env: Any, first_person: bool = True, agent_idx: int = 0
    ) -> str:
        """Create a language observation of the environment.

        Args:
            env: The environment to observe
            first_person: Whether to create a first-person (True) or third-person (False) observation
            agent_idx: The index of the agent to observe from (only used in first-person mode)

        Returns:
            A string description of the environment from the appropriate perspective
        """
        if first_person:
            return self._make_first_person_obs(env, agent_idx)
        else:
            return self._make_third_person_obs(env)

    def render(self, env: Any, **kwargs) -> str:
        """Render method for language observation."""
        first_person = kwargs.get("first_person", True)
        agent_idx = kwargs.get("agent_idx", 0)
        return self.make_language_obs(
            env,
            first_person=first_person,
            agent_idx=agent_idx if first_person else None,
        )
