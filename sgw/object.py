class Object:
    def __init__(
        self,
        pos,
        block_movement=True,
        remove_on_interact=False,
        terminate_on_interact=False,
    ):
        self.pos = pos
        self.block_movement = block_movement
        self.remove_on_interact = remove_on_interact
        self.terminate_on_interact = terminate_on_interact

    def __eq__(self, other):
        if isinstance(other, (list, tuple)):
            return self.pos[0] == other[0] and self.pos[1] == other[1]
        elif isinstance(other, Object):
            return self.pos[0] == other.pos[0] and self.pos[1] == other.pos[1]
        return False

    def copy(self):
        return type(self)(
            list(self.pos),
            self.block_movement,
            self.remove_on_interact,
            self.terminate_on_interact,
        )

    def interact(self, agent):
        if self.terminate_on_interact:
            agent.done = True


class Wall(Object):
    def __init__(self, pos):
        super().__init__(pos, True, False)

    def copy(self):
        return type(self)(list(self.pos))

    def interact(self, agent):
        super().interact(agent)


class Reward(Object):
    def __init__(self, pos, value):
        super().__init__(pos, False, True)
        self.value = value

    def copy(self):
        return type(self)(list(self.pos), self.value)

    def interact(self, agent):
        super().interact(agent)
        agent.collect_reward(self.value)


class Key(Object):
    def __init__(self, pos):
        super().__init__(pos, False, True)

    def copy(self):
        return type(self)(list(self.pos))

    def interact(self, agent):
        super().interact(agent)
        agent.collect_key()


class Door(Object):
    def __init__(self, pos, orientation):
        super().__init__(pos, False, True)
        self.orientation = orientation

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

    def copy(self):
        return type(self)(list(self.pos), list(self.target))

    def interact(self, agent):
        super().interact(agent)
        agent.teleport(self.target)


class Marker(Object):
    def __init__(self, pos, color):
        super().__init__(pos, False, False)
        self.color = color

    def copy(self):
        return type(self)(list(self.pos), tuple(self.color))

    def interact(self, agent):
        super().interact(agent)


class Other(Object):
    def __init__(self, pos, name):
        super().__init__(pos, False, False)
        self.name = name

    def copy(self):
        return type(self)(list(self.pos), self.name)

    def interact(self, agent):
        super().interact(agent)
