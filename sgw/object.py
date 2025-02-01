class Object:
    def __init__(self, pos):
        self.pos = pos

    def __eq__(self, other):
        if isinstance(other, (list, tuple)):
            return self.pos[0] == other[0] and self.pos[1] == other[1]
        elif isinstance(other, Object):
            return self.pos[0] == other.pos[0] and self.pos[1] == other.pos[1]
        return False

    def copy(self):
        return type(self)(list(self.pos))


class Wall(Object):
    def __init__(self, pos):
        super().__init__(pos)


class Reward(Object):
    def __init__(self, pos, value):
        super().__init__(pos)
        self.value = value

    def copy(self):
        return type(self)(list(self.pos), self.value)


class Key(Object):
    def __init__(self, pos):
        super().__init__(pos)


class Door(Object):
    def __init__(self, pos, orientation):
        super().__init__(pos)
        self.orientation = orientation

    def copy(self):
        return type(self)(list(self.pos), self.orientation)


class Warp(Object):
    def __init__(self, pos, target):
        super().__init__(pos)
        self.target = target

    def copy(self):
        return type(self)(list(self.pos), list(self.target))


class Marker(Object):
    def __init__(self, pos, color):
        super().__init__(pos)
        self.color = color

    def copy(self):
        return type(self)(list(self.pos), tuple(self.color))


class Other(Object):
    def __init__(self, pos, name):
        super().__init__(pos)
        self.name = name

    def copy(self):
        return type(self)(list(self.pos), self.name)
