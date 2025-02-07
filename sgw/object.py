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
