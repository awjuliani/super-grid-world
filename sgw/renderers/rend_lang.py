import numpy as np
from sgw.renderers.rend_interface import RendererInterface
from typing import Any, Tuple
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

    def _get_object_descriptions(
        self,
        objects,
        agent_pos,
        field_of_view,
        pronouns=None,
        agent_looking=None,
        control_type=None,
    ):
        """Unified helper method to generate descriptions for any type of object."""
        descriptions = []
        pos_type_pairs = [(obj.pos, obj.name) for obj in objects]

        first_person = pronouns["subject"] == "you"
        for pos, item_type in pos_type_pairs:
            direction, distance = self._get_direction_and_distance(
                pos, agent_pos, control_type=control_type, agent_looking=agent_looking
            )
            # Skip objects beyond field of view only in first person mode
            if first_person and distance > field_of_view:
                continue
            # If in egocentric mode, further restrict to objects in front based on viewing window
            if first_person and agent_looking is not None:
                if not self._is_in_visible_window(
                    pos, agent_pos, field_of_view, agent_looking
                ):
                    continue

            # Choose description format based on perspective
            if first_person:
                descriptions.append(
                    f"You see a {item_type} at your position."
                    if direction == "same position"
                    else f"You see a {item_type} {direction} you, {distance} {'meter' if distance == 1 else 'meters'} away."
                )
            else:
                descriptions.append(
                    f"There is a {item_type} at {pronouns['possessive']} position."
                    if direction == "same position"
                    else f"There is a {item_type} {direction} {pronouns['subject']}, {distance} {'meter' if distance == 1 else 'meters'} away."
                )

        return descriptions

    def _get_boundary_descriptions(self, agent_pos, pronouns):
        """Helper method to describe adjacent outer walls."""
        descriptions = []

        # Check each boundary
        if agent_pos[0] == 0:  # North wall
            descriptions.append(
                f"{pronouns['subject'].capitalize()} {pronouns['be']} against the north wall of the maze."
            )
        elif agent_pos[0] == self.grid_shape[1] - 1:  # South wall
            descriptions.append(
                f"{pronouns['subject'].capitalize()} {pronouns['be']} against the south wall of the maze."
            )

        if agent_pos[1] == 0:  # West wall
            descriptions.append(
                f"{pronouns['subject'].capitalize()} {pronouns['be']} against the west wall of the maze."
            )
        elif agent_pos[1] == self.grid_shape[0] - 1:  # East wall
            descriptions.append(
                f"{pronouns['subject'].capitalize()} {pronouns['be']} against the east wall of the maze."
            )

        return descriptions

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

    def make_language_obs(
        self, env: Any, first_person: bool = True, agent_idx: int = 0
    ) -> str:
        # Create pronouns dictionary locally based on first_person parameter
        pronouns = {
            "subject": "you" if first_person else "the agent",
            "possessive": "your" if first_person else "the agent's",
            "be": "are" if first_person else "is",
            "object": "yourself" if first_person else "itself",
        }

        agent = env.agents[agent_idx]
        agent_pos = np.array(agent.pos)

        # Get orientation description based on control type
        orientation_desc = ""
        if env.control_type == ControlType.egocentric:
            direction_map = {0: "north", 1: "east", 2: "south", 3: "west"}
            orientation_desc = f"{pronouns['subject'].capitalize()} {pronouns['be']} facing {direction_map[agent.looking]}. {pronouns['subject'].capitalize()} cannot see objects behind {pronouns['object']}."

        # Create inventory description
        if agent.inventory:
            inventory_items = {}
            for item in agent.inventory:
                inventory_items[item.name] = inventory_items.get(item.name, 0) + 1
            inventory_desc = ", ".join(
                f"{count} {name}" + ("s" if count > 1 else "")
                for name, count in inventory_items.items()
            )
            inventory_text = f"{pronouns['subject'].capitalize()} {pronouns['be']} carrying {inventory_desc}."
        else:
            inventory_text = f"{pronouns['subject'].capitalize()} {pronouns['be']} not carrying anything."

        base_description = (
            f"{pronouns['subject'].capitalize()} {pronouns['be']} in the {self._get_region(agent_pos)} region of a {self.grid_shape[0]}x{self.grid_shape[1]} meter maze. "
            f"\n{orientation_desc}"
            f"\n{inventory_text} "
            f"\n{pronouns['subject'].capitalize()} can only see objects up to {agent.field_of_view} {'meter' if agent.field_of_view == 1 else 'meters'} away. Objects further away than that are not visible."
        )

        # Get boundary descriptions
        all_descriptions = self._get_boundary_descriptions(agent_pos, pronouns)

        # Process all objects in a single pass
        all_objects = []
        for obj_list in env.objects.values():
            all_objects.extend(obj_list)

        if all_objects:
            agent_looking = (
                agent.looking if env.control_type == ControlType.egocentric else None
            )
            all_descriptions.extend(
                self._get_object_descriptions(
                    all_objects,
                    agent_pos,
                    agent.field_of_view,
                    pronouns=pronouns,
                    agent_looking=agent_looking,
                    control_type=env.control_type,
                )
            )

        return (
            f"{base_description}\nThere are no visible objects or walls near you."
            if not all_descriptions
            else f"{base_description}\n\n" + "\n".join(all_descriptions)
        )

    def render(self, env: Any, **kwargs) -> str:
        # Render method for language observation.
        first_person = kwargs.get("first_person", True)
        agent_idx = kwargs.get("agent_idx", 0)
        return self.make_language_obs(
            env, first_person=first_person, agent_idx=agent_idx
        )
