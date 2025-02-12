import numpy as np
from sgw.object import Key
from sgw.enums import Action, ControlType


class Agent:
    def __init__(self, pos, direction=0, field_of_view=2):
        self.pos = list(map(int, pos))
        self.orientation = direction  # Current rotation (0-3)
        self.looking = direction  # Direction agent is looking
        self.inventory = []
        self.reward = 0
        self.field_of_view = field_of_view
        self.done = False
        # Create direction map from Action enum
        direction_vectors = Action.get_direction_map()
        self.direction_map = np.array(
            [
                direction_vectors[Action.MOVE_NORTH],
                direction_vectors[Action.MOVE_EAST],
                direction_vectors[Action.MOVE_SOUTH],
                direction_vectors[Action.MOVE_WEST],
                [0, 0],  # For NOOP
                [0, 0],  # Extra padding
            ]
        )

    def process_action(self, action, control_type):
        """
        Process an action based on the control type.
        Returns the movement direction if any, None otherwise.
        """
        # Handle special actions (noop, collect)
        if action.is_special:
            return None

        # Handle egocentric actions
        if control_type == ControlType.egocentric:
            if action == Action.MOVE_FORWARD:
                return self.direction_map[self.orientation]
            elif action in [Action.ROTATE_LEFT, Action.ROTATE_RIGHT]:
                rotation = -1 if action == Action.ROTATE_LEFT else 1
                self.rotate(rotation)
                return None
        # Handle allocentric actions
        else:  # allocentric orientation
            if action.is_allocentric:
                direction_idx = Action.get_direction_index(action)
                self.looking = direction_idx
                return self.direction_map[direction_idx]
        return None

    def move(self, direction):
        """
        Moves the agent in the given direction.
        """
        if direction is not None:
            new_pos = np.array(self.pos) + direction
            self.pos = list(map(int, new_pos))
            return True
        return False

    def rotate(self, direction):
        """
        Rotates the agent orientation in the given direction.
        direction: -1 for left, 1 for right
        """
        self.orientation = (self.orientation + direction) % 4
        self.looking = self.orientation

    def move_forward(self):
        """Move in the direction the agent is facing."""
        self.move(self.direction_map[self.orientation])
        self.looking = self.orientation

    def move_allocentric(self, direction_idx):
        """Move in absolute direction regardless of orientation."""
        self.looking = direction_idx
        self.move(self.direction_map[direction_idx])

    def collect_object(self, object):
        """Collect an object."""
        self.inventory.append(object)

    def use_key(self):
        """Use a key. Returns True if key was available."""
        # check if a key object is in the inventory
        for item in self.inventory:
            if isinstance(item, Key):
                self.inventory.remove(item)
                return True
        return False

    def get_position(self):
        """Returns the current position as a list of ints."""
        return list(map(int, self.pos))

    def teleport(self, new_pos):
        """Teleport agent to new position."""
        self.pos = list(map(int, new_pos))

    def collect_reward(self, reward):
        """Collect a reward."""
        self.reward += reward
