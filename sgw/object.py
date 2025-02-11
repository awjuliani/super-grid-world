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
        if self.terminal:
            agent.done = True

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


class Key(Object):
    def __init__(self, pos):
        super().__init__(pos, False, True)
        self.name = "door key"

    def copy(self):
        return type(self)(list(self.pos))

    def interact(self, agent):
        super().interact(agent)
        agent.collect_object(self)


class Door(Object):
    def __init__(self, pos, orientation):
        super().__init__(pos, False, True)
        self.orientation = orientation
        self.name = "locked door"

    def copy(self):
        return type(self)(list(self.pos), self.orientation)

    def interact(self, agent):
        super().interact(agent)
        if agent.use_key():
            agent.teleport(self.pos)


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


class Marker(Object):
    def __init__(self, pos, color):
        super().__init__(pos, False, False)
        self.color = color
        self.name = "marker"

    def copy(self):
        return type(self)(list(self.pos), tuple(self.color))

    def interact(self, agent):
        super().interact(agent)


class Other(Object):
    def __init__(self, pos, name):
        super().__init__(pos, False, True, False)
        self.name = name

    def copy(self):
        return type(self)(list(self.pos), self.name)

    def interact(self, agent):
        super().interact(agent)
        agent.collect_object(self)


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
                env.events.append("A fruit fell from a tree")


class Fruit(Object):
    def __init__(self, pos):
        super().__init__(pos, False, True, False)
        self.name = "fruit"

    def copy(self):
        return type(self)(list(self.pos))

    def interact(self, agent):
        super().interact(agent)
        agent.collect_object(self)
