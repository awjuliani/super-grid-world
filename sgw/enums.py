import enum


class ObsType(enum.Enum):
    visual_2d = "visual_2d"
    visual_3d = "visual_3d"
    symbolic = "symbolic"
    ascii = "ascii"
    language = "language"


class ControlType(enum.Enum):
    allocentric = "allocentric"
    egocentric = "egocentric"


# New enum to represent all possible actions semantically.
class Action(enum.Enum):
    # Actions for allocentric orientation (direct movement)
    MOVE_NORTH = (enum.auto(), "move_north")
    MOVE_EAST = (enum.auto(), "move_east")
    MOVE_SOUTH = (enum.auto(), "move_south")
    MOVE_WEST = (enum.auto(), "move_west")
    # Actions for egocentric orientation (rotate then move)
    ROTATE_LEFT = (enum.auto(), "rotate_left")
    ROTATE_RIGHT = (enum.auto(), "rotate_right")
    MOVE_FORWARD = (enum.auto(), "move_forward")
    # Optional actions for both types
    NOOP = (enum.auto(), "noop")
    INTERACT = (enum.auto(), "interact")

    def __init__(self, id, action_name):
        self._value_ = (id, action_name)
        self.id = id
        self.action_name = action_name

    @property
    def is_allocentric(self):
        """Check if action is an allocentric movement."""
        return self in [
            Action.MOVE_NORTH,
            Action.MOVE_EAST,
            Action.MOVE_SOUTH,
            Action.MOVE_WEST,
        ]

    @property
    def is_egocentric(self):
        """Check if action is an egocentric movement/rotation."""
        return self in [Action.ROTATE_LEFT, Action.ROTATE_RIGHT, Action.MOVE_FORWARD]

    @property
    def is_special(self):
        """Check if action is a special action (noop/interact)."""
        return self in [Action.NOOP, Action.INTERACT]

    @classmethod
    def get_actions_for_control(
        cls,
        control_type: ControlType,
        use_noop: bool = False,
        manual_interact: bool = False,
    ):
        """Get valid actions for a given control type."""
        if control_type == ControlType.allocentric:
            actions = [a for a in cls if a.is_allocentric]
        else:  # egocentric
            actions = [a for a in cls if a.is_egocentric]

        if use_noop:
            actions.append(cls.NOOP)
        if manual_interact:
            actions.append(cls.INTERACT)
        return actions

    @classmethod
    def get_direction_index(cls, action):
        """Get the direction index for an allocentric action."""
        if not action.is_allocentric:
            raise ValueError(f"Action {action} is not an allocentric movement action")
        direction_order = [cls.MOVE_NORTH, cls.MOVE_EAST, cls.MOVE_SOUTH, cls.MOVE_WEST]
        return direction_order.index(action)

    @staticmethod
    def get_direction_map():
        """Get the mapping of actions to direction vectors."""
        return {
            Action.MOVE_NORTH: [-1, 0],
            Action.MOVE_EAST: [0, 1],
            Action.MOVE_SOUTH: [1, 0],
            Action.MOVE_WEST: [0, -1],
        }
