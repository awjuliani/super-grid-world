import numpy as np
from sgw.renderers.rend_interface import RendererInterface
from typing import Any
from gym import spaces
from sgw.enums import ControlType


class GridLangRenderer(RendererInterface):
    def __init__(self, grid_size: int):
        self.grid_size = grid_size

    @property
    def observation_space(self) -> spaces.Space:
        """Return the observation space for language observations."""
        return spaces.Discrete(1)

    def _get_region(self, pos):
        """Helper method to determine region in the maze."""
        third = self.grid_size / 3
        x, y = pos[1], pos[0]

        # Determine vertical region (north/center/south)
        if y < third:
            v_region = "north"
        elif y > 2 * third:
            v_region = "south"
        else:
            v_region = "center"

        # Determine horizontal region (west/center/east)
        if x < third:
            h_region = "west"
        elif x > 2 * third:
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

    def _get_object_descriptions(
        self,
        objects,
        agent_pos,
        field_of_view,
        pronouns=None,
        agent_looking=None,
    ):
        """Unified helper method to generate descriptions for any type of object."""
        descriptions = []
        pos_type_pairs = [(obj.pos, obj.name) for obj in objects]

        first_person = pronouns["subject"] == "you"
        for pos, item_type in pos_type_pairs:
            direction, distance = self._get_direction_and_distance(pos, agent_pos)
            # Skip objects beyond field of view only in first person mode
            if first_person and distance > field_of_view:
                continue
            # If in egocentric mode, further restrict to objects in front based on viewing window
            if first_person and agent_looking is not None:
                if not self._is_in_visible_window(
                    pos, agent_pos, field_of_view, agent_looking
                ):
                    continue
            descriptions.append(
                f"There is a {item_type} at {pronouns['possessive']} position."
                if direction == "same position"
                else f"There is a {item_type} {direction} of {pronouns['subject']}, {distance} {'meter' if distance == 1 else 'meters'} away."
            )

        return descriptions

    def _get_direction_and_distance(self, obj_pos, agent_pos):
        """Helper method to get cardinal direction and distance."""
        diff = np.array(obj_pos) - agent_pos
        y_dir = "north" if diff[0] < 0 else "south" if diff[0] > 0 else ""
        x_dir = "east" if diff[1] > 0 else "west" if diff[1] < 0 else ""

        direction = (
            f"{y_dir}-{x_dir}"
            if (y_dir and x_dir)
            else (y_dir or x_dir or "same position")
        )
        # Use Euclidean distance with 0.5 unit increments
        distance = (
            np.floor(2 * np.sqrt(diff[0] ** 2 + diff[1] ** 2)) / 2
            if direction != "same position"
            else 0
        )

        return direction, int(distance)

    def _get_boundary_descriptions(self, agent_pos, pronouns):
        """Helper method to describe adjacent outer walls."""
        descriptions = []

        # Check each boundary
        if agent_pos[0] == 0:  # North wall
            descriptions.append(
                f"{pronouns['subject'].capitalize()} {pronouns['be']} against the north wall of the maze."
            )
        elif agent_pos[0] == self.grid_size - 1:  # South wall
            descriptions.append(
                f"{pronouns['subject'].capitalize()} {pronouns['be']} against the south wall of the maze."
            )

        if agent_pos[1] == 0:  # West wall
            descriptions.append(
                f"{pronouns['subject'].capitalize()} {pronouns['be']} against the west wall of the maze."
            )
        elif agent_pos[1] == self.grid_size - 1:  # East wall
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
        }

        agent = env.agents[agent_idx]
        agent_pos = np.array(agent.pos)

        # Get orientation description based on control type
        orientation_desc = ""
        if env.control_type == ControlType.egocentric:
            direction_map = {0: "north", 1: "east", 2: "south", 3: "west"}
            orientation_desc = f"{pronouns['subject'].capitalize()} {pronouns['be']} facing {direction_map[agent.looking]}. "

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
            f"{pronouns['subject'].capitalize()} {pronouns['be']} in the {self._get_region(agent_pos)} region of a {self.grid_size}x{self.grid_size} meter maze. "
            f"{orientation_desc}"
            f"\n{inventory_text} "
            f"\n{pronouns['subject'].capitalize()} can only see objects up to {agent.field_of_view} {'meter' if agent.field_of_view == 1 else 'meters'} away."
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
