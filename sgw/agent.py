import numpy as np


class Agent:
    def __init__(self, pos, direction=0):
        self.pos = list(map(int, pos))
        self.orientation = direction  # Current rotation (0-3)
        self.looking = direction  # Direction agent is looking
        self.keys = 0
        self.direction_map = np.array(
            [[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0], [0, 0]]
        )

    def process_action(self, action, control_type):
        """
        Process an action based on the control type.
        Returns the movement direction if any, None otherwise.
        """
        from sgw.env import Action, ControlType

        if control_type == ControlType.egocentric:
            if action == Action.ROTATE_LEFT:
                self.rotate(-1)
                return None
            elif action == Action.ROTATE_RIGHT:
                self.rotate(1)
                return None
            elif action == Action.MOVE_FORWARD:
                return self.direction_map[self.orientation]
        else:  # allocentric orientation
            allocentric_mapping = {
                Action.MOVE_UP: 0,
                Action.MOVE_RIGHT: 1,
                Action.MOVE_DOWN: 2,
                Action.MOVE_LEFT: 3,
            }
            if action in allocentric_mapping:
                direction_idx = allocentric_mapping[action]
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

    def collect_key(self):
        """Collect a key."""
        self.keys += 1

    def use_key(self):
        """Use a key. Returns True if key was available."""
        if self.keys > 0:
            self.keys -= 1
            return True
        return False

    def get_position(self):
        """Returns the current position as a list of ints."""
        return list(map(int, self.pos))

    def teleport(self, new_pos):
        """Teleport agent to new position."""
        self.pos = list(map(int, new_pos))
