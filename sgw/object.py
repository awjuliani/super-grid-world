import numpy as np


class Object:
    def __init__(
        self,
        pos,
        obstacle=True,
        consumable=False,
        terminal=False,
    ):
        self.pos = pos
        self.obstacle = obstacle
        self.consumable = consumable
        self.terminal = terminal
        self.name = "object"

    def __eq__(self, other):
        if isinstance(other, (list, tuple)):
            return self.pos[0] == other[0] and self.pos[1] == other[1]
        elif isinstance(other, Object):
            return self.pos[0] == other.pos[0] and self.pos[1] == other.pos[1]
        return False

    def copy(self):
        return type(self)(
            list(self.pos),
            self.obstacle,
            self.consumable,
            self.terminal,
        )

    def interact(self, agent):
        """Called when agent interacts with object. Returns event string if any."""
        if self.terminal:
            agent.done = True
        return None

    def step(self, env):
        """Called each step of the environment. Can be used to update object state."""
        pass


class Wall(Object):
    def __init__(self, pos):
        super().__init__(pos, True, False)
        self.name = "wall"

    def copy(self):
        return type(self)(list(self.pos))

    def interact(self, agent):
        super().interact(agent)
        return None


class Reward(Object):
    def __init__(self, pos, value):
        super().__init__(pos, False, True)
        self.value = value
        self.name = "reward"

    def copy(self):
        return type(self)(list(self.pos), self.value)

    def interact(self, agent):
        super().interact(agent)
        agent.collect_reward(self.value)
        return f"Agent collected reward of {self.value}"


class Key(Object):
    def __init__(self, pos):
        super().__init__(pos, False, True)
        self.name = "door key"

    def copy(self):
        return type(self)(list(self.pos))

    def interact(self, agent):
        super().interact(agent)
        agent.collect_object(self)
        return "Agent collected a key"


class Door(Object):
    def __init__(self, pos, orientation):
        super().__init__(pos, obstacle=True, consumable=True)
        self.orientation = orientation
        self.name = "locked door"

    def copy(self):
        return type(self)(list(self.pos), self.orientation)

    def try_unlock(self, agent):
        """Attempt to unlock the door with a key from the agent's inventory."""
        if not self.obstacle:  # Already unlocked
            return True, None
        if agent.use_key():
            self.obstacle = False  # Unlock the door
            return True, "Agent unlocked and went through a door"
        return False, "Agent tried to open door but had no key"

    def interact(self, agent):
        super().interact(agent)
        success, message = self.try_unlock(agent)
        return message

    def pre_step_interaction(self, agent, direction):
        return self.try_unlock(agent)


class Warp(Object):
    def __init__(self, pos, target):
        super().__init__(pos, False, False)
        self.target = target
        self.name = "warp pad"

    def copy(self):
        return type(self)(list(self.pos), list(self.target))

    def interact(self, agent):
        super().interact(agent)
        agent.teleport(self.target)
        return "Agent used a warp pad to teleport"


class Marker(Object):
    def __init__(self, pos, color):
        super().__init__(pos, False, False)
        self.color = color
        self.name = "marker"

    def copy(self):
        return type(self)(list(self.pos), tuple(self.color))

    def interact(self, agent):
        super().interact(agent)
        return None


class Other(Object):
    def __init__(self, pos, name):
        super().__init__(pos, False, True, False)
        self.name = name

    def copy(self):
        return type(self)(list(self.pos), self.name)

    def interact(self, agent):
        super().interact(agent)
        agent.collect_object(self)
        return f"Agent collected {self.name}"


class Tree(Object):
    def __init__(self, pos, spawn_rate=0.5, spawn_radius=2):
        super().__init__(pos, True, False, False)
        self.name = "tree"
        self.spawn_rate = spawn_rate
        self.spawn_radius = spawn_radius

    def copy(self):
        return type(self)(list(self.pos), self.spawn_rate, self.spawn_radius)

    def interact(self, agent):
        super().interact(agent)
        return None

    def step(self, env):
        """Potentially spawn a fruit in a nearby location."""
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
    def __init__(self, pos):
        super().__init__(pos, False, True, False)
        self.name = "fruit"

    def copy(self):
        return type(self)(list(self.pos))

    def interact(self, agent):
        super().interact(agent)
        agent.collect_object(self)
        return "Agent collected a fruit"


class Box(Object):
    def __init__(self, pos):
        super().__init__(pos, False, False, False)
        self.name = "box"
        self.contents = []

    def copy(self):
        new_box = type(self)(list(self.pos))
        new_box.contents = self.contents.copy()
        return new_box

    def interact(self, agent):
        super().interact(agent)
        if agent.inventory:
            # Put item in box
            item = agent.inventory.pop(0)  # Remove and get first item from inventory
            self.contents.append(item)
            return f"Agent put {item.name} in the box"
        elif self.contents:
            # Take item from box
            item = self.contents.pop()  # Remove and get last item from box
            agent.collect_object(item)
            return f"Agent took {item.name} from the box"
        return "Box is empty and agent has no items"


class Sign(Object):
    def __init__(self, pos, message):
        super().__init__(pos, False, False, False)
        self.message = message
        self.name = "sign"

    def copy(self):
        return type(self)(list(self.pos), self.message)

    def interact(self, agent):
        super().interact(agent)
        return f"The sign reads: '{self.message}'"


class PushableBox(Object):
    def __init__(self, pos):
        super().__init__(pos, obstacle=True, consumable=False, terminal=False)
        self.name = "pushable box"
        self.being_pushed = False

    def copy(self):
        new_box = type(self)(list(self.pos))
        new_box.being_pushed = self.being_pushed
        return new_box

    def interact(self, agent):
        super().interact(agent)
        return "Agent tried to interact with a pushable box"

    def pre_step_interaction(self, agent, direction):
        """
        Handle pushing the box when an agent tries to move into its position.
        Returns (success, message) tuple.
        """
        # Calculate the position the box would be pushed to
        push_target = list((np.array(self.pos) + direction).tolist())

        # Store the direction for the step method
        self.push_direction = direction
        self.being_pushed = True

        # Return True to allow the agent to move into our position
        # We'll temporarily set ourselves as non-obstacle
        self.obstacle = False

        return True, "Agent is pushing a box"

    def step(self, env):
        """Handle the actual pushing of the box if needed."""
        super().step(env)

        if self.being_pushed and hasattr(self, "push_direction"):
            # Calculate the position the box would be pushed to
            push_target = list((np.array(self.pos) + self.push_direction).tolist())

            # Check if the target position is valid
            if env.check_target(push_target):
                # Move the box to the new position
                self.pos = push_target
                # Reset the obstacle property
                self.obstacle = True
                # Reset the push flag
                self.being_pushed = False
                return "Box was pushed"
            else:
                # If we couldn't push, reset the obstacle property
                self.obstacle = True
                self.being_pushed = False
                return "Box couldn't be pushed"

        return None
