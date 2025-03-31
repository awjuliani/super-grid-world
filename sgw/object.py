from typing import List, Tuple, Optional, Any


class Object:
    """Base class for all objects in the environment."""

    def __init__(
        self,
        pos: List[int],
        obstacle: bool = True,
        consumable: bool = False,
        terminal: bool = False,
    ):
        """
        Initialize an object.

        Args:
            pos: [x, y] position of the object
            obstacle: Whether the object blocks movement
            consumable: Whether the object can be consumed/collected
            terminal: Whether interacting with this object ends the episode
        """
        self.pos = pos
        self.obstacle = obstacle
        self.consumable = consumable
        self.terminal = terminal
        self.name = "object"

    def __eq__(self, other: Any) -> bool:
        """Check if this object is at the same position as another object or coordinate."""
        if isinstance(other, (list, tuple)):
            return self.pos[0] == other[0] and self.pos[1] == other[1]
        elif isinstance(other, Object):
            return self.pos[0] == other.pos[0] and self.pos[1] == other.pos[1]
        return False

    def copy(self) -> "Object":
        """Create a deep copy of this object."""
        return type(self)(
            list(self.pos),
            self.obstacle,
            self.consumable,
            self.terminal,
        )

    def interact(self, agent: Any) -> Optional[str]:
        """
        Called when agent interacts with object.

        Args:
            agent: The agent interacting with this object

        Returns:
            Optional event message string
        """
        if self.terminal:
            agent.done = True
        return None

    def step(self, env: Any) -> None:
        """
        Called each step of the environment.

        Args:
            env: The environment instance
        """
        pass


class Wall(Object):
    """A wall that blocks movement."""

    def __init__(self, pos: List[int]):
        super().__init__(pos, obstacle=True, consumable=False)
        self.name = "obstacle"

    def copy(self) -> "Wall":
        return type(self)(list(self.pos))


class Reward(Object):
    """A reward object that gives the agent a specified value when collected."""

    def __init__(self, pos: List[int], value: float):
        super().__init__(pos, obstacle=False, consumable=True)
        self.value = value
        self.name = "reward"

    def copy(self) -> "Reward":
        return type(self)(list(self.pos), self.value)

    def interact(self, agent: Any) -> str:
        super().interact(agent)
        agent.collect_reward(self.value)
        return f"{agent.name} collected {self.name} of {self.value}"


class Key(Object):
    """A key that can be collected and used to unlock doors."""

    def __init__(self, pos: List[int]):
        super().__init__(pos, obstacle=False, consumable=True)
        self.name = "door key"

    def copy(self) -> "Key":
        return type(self)(list(self.pos))

    def interact(self, agent: Any) -> str:
        super().interact(agent)
        agent.collect_object(self)
        return f"{agent.name} collected a {self.name}"


class Door(Object):
    """A door that can be unlocked with a key."""

    def __init__(self, pos: List[int], orientation: Any):
        super().__init__(pos, obstacle=True, consumable=True)
        self.orientation = orientation
        self.name = "locked door"

    def copy(self) -> "Door":
        return type(self)(list(self.pos), self.orientation)

    def try_unlock(self, agent: Any) -> Tuple[bool, Optional[str]]:
        """
        Attempt to unlock the door with a key from the agent's inventory.

        Args:
            agent: The agent trying to unlock the door

        Returns:
            Tuple of (success, message)
        """
        if not self.obstacle:  # Already unlocked
            return True, None
        if agent.use_key():
            self.obstacle = False  # Unlock the door
            return True, f"{agent.name} unlocked and went through a {self.name}"
        return False, f"{agent.name} tried to open a {self.name} but had no key"

    def interact(self, agent: Any) -> Optional[str]:
        super().interact(agent)
        success, message = self.try_unlock(agent)
        return message

    def pre_step_interaction(
        self, agent: Any, direction: Any, env: Any = None
    ) -> Tuple[bool, Optional[str]]:
        return self.try_unlock(agent)


class Warp(Object):
    """A teleportation pad that moves the agent to a target location."""

    def __init__(self, pos: List[int], target: List[int]):
        super().__init__(pos, obstacle=False, consumable=False)
        self.target = target
        self.name = "warp pad"

    def copy(self) -> "Warp":
        return type(self)(list(self.pos), list(self.target))

    def interact(self, agent: Any) -> str:
        super().interact(agent)
        agent.teleport(self.target)
        return f"{agent.name} used a {self.name} to teleport"


class Marker(Object):
    """A colored marker on the ground."""

    def __init__(self, pos: List[int], color: Tuple[int, int, int]):
        super().__init__(pos, obstacle=False, consumable=False)
        self.color = color
        self.name = "marker"

    def copy(self) -> "Marker":
        return type(self)(list(self.pos), tuple(self.color))


class Other(Object):
    """A generic collectible object with a custom name."""

    def __init__(self, pos: List[int], name: str):
        super().__init__(pos, obstacle=False, consumable=True, terminal=False)
        self.name = name

    def copy(self) -> "Other":
        return type(self)(list(self.pos), self.name)

    def interact(self, agent: Any) -> str:
        super().interact(agent)
        agent.collect_object(self)
        return f"{agent.name} collected {self.name}"


class Tree(Object):
    """A tree that can spawn fruits in nearby locations."""

    def __init__(self, pos: List[int], spawn_rate: float = 0.5, spawn_radius: int = 2):
        super().__init__(pos, obstacle=True, consumable=False, terminal=False)
        self.name = "tree"
        self.spawn_rate = spawn_rate
        self.spawn_radius = spawn_radius

    def copy(self) -> "Tree":
        return type(self)(list(self.pos), self.spawn_rate, self.spawn_radius)

    def step(self, env: Any) -> Optional[str]:
        """
        Potentially spawn a fruit in a nearby location.

        Args:
            env: The environment instance

        Returns:
            Optional event message string
        """
        super().step(env)

        # Random chance to spawn fruit
        if env.rng.random() < self.spawn_rate:
            # Get possible spawn positions in radius
            possible_positions = []
            for dx in range(-self.spawn_radius, self.spawn_radius + 1):
                for dy in range(-self.spawn_radius, self.spawn_radius + 1):
                    new_pos = [self.pos[0] + dx, self.pos[1] + dy]
                    # Check if position is valid and empty
                    if env.check_target(new_pos):
                        possible_positions.append(new_pos)

            # Spawn fruit at random valid position if any exist
            if possible_positions:
                spawn_idx = env.rng.choice(len(possible_positions))
                spawn_pos = possible_positions[spawn_idx]
                new_fruit = Fruit(spawn_pos)
                if "fruits" not in env.objects:
                    env.objects["fruits"] = []
                env.objects["fruits"].append(new_fruit)
                return "A fruit fell from a tree"
        return None


class Fruit(Object):
    """A fruit that can be collected by the agent."""

    def __init__(self, pos: List[int]):
        super().__init__(pos, obstacle=False, consumable=True, terminal=False)
        self.name = "fruit"

    def copy(self) -> "Fruit":
        return type(self)(list(self.pos))

    def interact(self, agent: Any) -> str:
        super().interact(agent)
        agent.collect_object(self)
        return f"{agent.name} collected a {self.name}"


class Box(Object):
    """A box that can store and provide items."""

    def __init__(self, pos: List[int]):
        super().__init__(pos, obstacle=False, consumable=False, terminal=False)
        self.name = "box"
        self.contents = []

    def copy(self) -> "Box":
        new_box = type(self)(list(self.pos))
        new_box.contents = self.contents.copy()
        return new_box

    def interact(self, agent: Any) -> str:
        super().interact(agent)
        if agent.inventory:
            # Put item in box
            item = agent.inventory.pop(0)  # Remove and get first item from inventory
            self.contents.append(item)
            return f"{agent.name} put {item.name} in the box"
        elif self.contents:
            # Take item from box
            item = self.contents.pop()  # Remove and get last item from box
            agent.collect_object(item)
            return f"{agent.name} took {item.name} from the box"
        return f"{agent.name} found a {self.name} but it was empty"


class Sign(Object):
    """A sign with a readable message."""

    def __init__(self, pos: List[int], message: str):
        super().__init__(pos, obstacle=False, consumable=False, terminal=False)
        self.message = message
        self.name = "sign"

    def copy(self) -> "Sign":
        return type(self)(list(self.pos), self.message)

    def interact(self, agent: Any) -> str:
        super().interact(agent)
        return f"{agent.name} read the sign: '{self.message}'"


class PushableBox(Object):
    """A box that can be pushed by the agent."""

    def __init__(self, pos: List[int]):
        super().__init__(pos, obstacle=True, consumable=False, terminal=False)
        self.name = "pushable box"
        self.being_pushed = False

    def copy(self) -> "PushableBox":
        new_box = type(self)(list(self.pos))
        new_box.being_pushed = self.being_pushed
        return new_box

    def pre_step_interaction(
        self, agent: Any, direction: Any, env: Any = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Called before the agent moves onto the box's position.
        Calculates the new position for the box based on the direction the agent is looking.

        Args:
            agent: The agent trying to push the box
            direction: The direction the agent is moving
            env: The environment instance, needed to check if the new position is valid

        Returns:
            Tuple of (allowed, message)
        """
        # Get the direction the agent is looking
        looking_idx = agent.looking

        # Get the direction vector from the agent's direction map
        push_direction = agent.direction_map[looking_idx].tolist()

        # Calculate the new position for the box
        new_box_pos = [self.pos[0] + push_direction[0], self.pos[1] + push_direction[1]]

        # Check if the new position is valid (if env is provided)
        # Use obstacles_only=False to prevent pushing onto any object
        if env is None or not env.check_target(new_box_pos, obstacles_only=False):
            return False, f"{agent.name} tried to push a {self.name} but it's blocked"

        # Update the box's position
        self.pos = new_box_pos
        self.being_pushed = True

        # Return True to allow the agent to move to the box's original position
        return True, f"{agent.name} pushed a {self.name}"

    def interact(self, agent: Any) -> None:
        """Called when the agent interacts with the box directly."""
        super().interact(agent)
        self.being_pushed = False
        return None

    def step(self, env: Any) -> None:
        super().step(env)
