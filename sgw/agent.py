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

    def move(self, direction):
        """
        Moves the agent in the given direction if the move is valid.
        Returns True if move was successful, False otherwise.
        """
        new_pos = np.array(self.pos) + direction
        new_pos = list(map(int, new_pos))
        self.pos = new_pos
        return True

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
