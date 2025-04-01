import pytest
import numpy as np
from sgw.object import (
    Object,
    Wall,
    Reward,
    Key,
    Door,
    Warp,
    Marker,
    Other,
    Tree,
    Fruit,
    Box,
    Sign,
    PushableBox,
    LinkedDoor,
    PressurePlate,
    Lever,
    ResetButton,
)
from sgw.enums import Action, ControlType  # Import necessary enums
import copy


# --- Mocks for testing interactions ---
class MockAgent:
    def __init__(self, pos=[0, 0], name="test_agent", inventory=None):
        self.pos = pos
        self.name = name
        self.inventory = inventory if inventory is not None else []
        self.reward = 0
        self.done = False
        self.looking = 1  # Default looking East
        self.direction_map = np.array(
            [
                [-1, 0],  # North
                [0, 1],  # East
                [1, 0],  # South
                [0, -1],  # West
            ]
        )

    def collect_reward(self, value):
        self.reward += value

    def collect_object(self, obj):
        self.inventory.append(obj)

    def use_key(self):
        key_instance = next(
            (item for item in self.inventory if isinstance(item, Key)), None
        )
        if key_instance:
            self.inventory.remove(key_instance)
            return True
        return False

    def teleport(self, target_pos):
        self.pos = target_pos

    def get_position(self):
        return self.pos


class MockEnv:
    def __init__(self, grid_shape=(10, 10), objects=None, agents=None, rng_seed=123):
        self.grid_shape = grid_shape
        self.objects = objects if objects is not None else {}
        self.agents = agents if agents is not None else []
        self.rng = np.random.RandomState(rng_seed)
        self._check_target_return = True  # Default behavior for check_target
        self.initial_objects = None

    def set_check_target_return(self, value):
        self._check_target_return = value

    def check_target(self, target_pos, obstacles_only=True):
        # Basic bounds check
        if not (
            0 <= target_pos[0] < self.grid_shape[0]
            and 0 <= target_pos[1] < self.grid_shape[1]
        ):
            return False
        # Mock object collision check (simplified)
        for obj_list in self.objects.values():
            for obj in obj_list:
                if obj.pos == list(target_pos):
                    if obstacles_only:
                        if obj.obstacle:
                            return False
                    else:  # Any object blocks if obstacles_only is False
                        # Allow stepping on non-obstacle objects like plates
                        if not obj.obstacle:
                            continue
                        return False
        # Check agent collision
        for agent in self.agents:
            if agent.get_position() == list(target_pos):
                return False  # Cannot move into a square occupied by another agent

        # Allow controlling the return value for specific tests
        return self._check_target_return

    def _find_object_at(self, pos):
        """Helper to find any object at a given position."""
        target_pos = list(pos)
        for obj_list in self.objects.values():
            for obj in obj_list:
                if obj.pos == target_pos:
                    return obj
        return None


# --- Existing Tests ---
def test_object_initialization():
    """Test basic Object initialization."""
    pos = [1, 2]
    obj = Object(pos, obstacle=True, consumable=False, terminal=True)
    assert obj.pos == pos
    assert obj.obstacle is True
    assert obj.consumable is False
    assert obj.terminal is True
    assert obj.name == "object"


def test_object_equality():
    """Test Object equality check."""
    obj1 = Object([1, 2])
    obj2 = Object([1, 2])
    obj3 = Object([3, 4])
    assert obj1 == obj2
    assert obj1 == [1, 2]
    assert obj1 == (1, 2)
    assert obj1 != obj3
    assert obj1 != [3, 4]
    assert obj1 != (3, 4)
    assert obj1 != "not_an_object"


def test_object_copy():
    """Test the copy method of the base Object."""
    obj = Object([1, 2], obstacle=True, consumable=False, terminal=True)
    obj_copy = obj.copy()
    assert obj is not obj_copy  # Ensure it's a new instance
    assert obj.pos == obj_copy.pos
    assert obj.obstacle == obj_copy.obstacle
    assert obj.consumable == obj_copy.consumable
    assert obj.terminal == obj_copy.terminal
    assert type(obj) == type(obj_copy)
    # Modify original and check copy is unaffected
    obj.pos = [3, 3]
    assert obj_copy.pos == [1, 2]


def test_wall_initialization():
    """Test Wall initialization."""
    pos = [5, 5]
    wall = Wall(pos)
    assert wall.pos == pos
    assert wall.obstacle is True
    assert wall.consumable is False
    assert wall.terminal is False
    assert wall.name == "obstacle"


def test_wall_copy():
    """Test Wall copy method."""
    wall = Wall([5, 5])
    wall_copy = wall.copy()
    assert wall is not wall_copy
    assert wall.pos == wall_copy.pos
    assert wall.obstacle == wall_copy.obstacle
    assert type(wall) == type(wall_copy)
    wall.pos = [6, 6]
    assert wall_copy.pos == [5, 5]


def test_reward_initialization():
    """Test Reward initialization."""
    pos = [3, 4]
    value = 10.0
    reward = Reward(pos, value)
    assert reward.pos == pos
    assert reward.value == value
    assert reward.obstacle is False
    assert reward.consumable is True
    assert reward.terminal is False
    assert reward.name == "reward"


def test_reward_copy():
    """Test Reward copy method."""
    reward = Reward([3, 4], 10.0)
    reward_copy = reward.copy()
    assert reward is not reward_copy
    assert reward.pos == reward_copy.pos
    assert reward.value == reward_copy.value
    assert reward.consumable == reward_copy.consumable
    assert type(reward) == type(reward_copy)
    reward.pos = [7, 7]
    reward.value = 5.0
    assert reward_copy.pos == [3, 4]
    assert reward_copy.value == 10.0


def test_reward_interact():
    """Test Reward interaction."""
    agent = MockAgent()
    reward = Reward([0, 0], 5.0)
    message = reward.interact(agent)
    assert agent.reward == 5.0
    assert "collected reward of 5.0" in message


# --- Key Tests ---
def test_key_initialization():
    """Test Key initialization."""
    pos = [2, 2]
    key = Key(pos)
    assert key.pos == pos
    assert key.obstacle is False
    assert key.consumable is True
    assert key.terminal is False
    assert key.name == "door key"


def test_key_copy():
    """Test Key copy method."""
    key = Key([2, 2])
    key_copy = key.copy()
    assert key is not key_copy
    assert key.pos == key_copy.pos
    assert key.consumable == key_copy.consumable
    assert type(key) == type(key_copy)
    key.pos = [1, 1]
    assert key_copy.pos == [2, 2]


def test_key_interact():
    """Test Key interaction."""
    agent = MockAgent()
    key = Key([0, 0])
    message = key.interact(agent)
    assert key in agent.inventory
    assert f"{agent.name} collected a {key.name}" in message


# --- Door Tests ---
def test_door_initialization():
    """Test Door initialization."""
    pos = [4, 4]
    orientation = 1  # Example orientation
    door = Door(pos, orientation)
    assert door.pos == pos
    assert door.orientation == orientation
    assert door.obstacle is True  # Starts locked
    assert door.consumable is True  # Consumed on interaction (unlocking)
    assert door.terminal is False
    assert door.name == "locked door"


def test_door_copy():
    """Test Door copy method."""
    door = Door([4, 4], 1)
    door_copy = door.copy()
    assert door is not door_copy
    assert door.pos == door_copy.pos
    assert door.orientation == door_copy.orientation
    assert door.obstacle == door_copy.obstacle
    assert type(door) == type(door_copy)
    door.pos = [5, 5]
    door.obstacle = False
    assert door_copy.pos == [4, 4]
    assert door_copy.obstacle is True


def test_door_interact_locked_no_key():
    """Test interacting with a locked door without a key."""
    agent = MockAgent(pos=[4, 4])
    door = Door([4, 4], 1)
    message = door.interact(agent)
    assert door.obstacle is True
    assert "had no key" in message


def test_door_interact_locked_with_key():
    """Test interacting with a locked door with a key."""
    key = Key([9, 9])  # Key position doesn't matter for inventory
    agent = MockAgent(pos=[4, 4], inventory=[key])
    door = Door([4, 4], 1)
    message = door.interact(agent)
    assert door.obstacle is False  # Door should be unlocked
    assert key not in agent.inventory  # Key should be used
    assert "unlocked and went through" in message


def test_door_interact_unlocked():
    """Test interacting with an already unlocked door."""
    agent = MockAgent(pos=[4, 4])
    door = Door([4, 4], 1)
    door.obstacle = False  # Pre-unlock the door
    message = door.interact(agent)
    assert door.obstacle is False  # Stays unlocked
    assert message is None  # try_unlock returns (True, None) if already unlocked


def test_door_pre_step_interaction():
    """Test pre-step interaction (similar logic to interact)."""
    key = Key([9, 9])
    agent_with_key = MockAgent(pos=[3, 4], inventory=[key])  # Agent at adjacent pos
    agent_no_key = MockAgent(pos=[3, 4])
    door = Door([4, 4], 1)
    direction = [1, 0]  # Example direction towards door

    # Try unlock with key
    allowed, message = door.pre_step_interaction(agent_with_key, direction)
    assert allowed is True
    assert door.obstacle is False
    assert key not in agent_with_key.inventory
    assert "unlocked and went through" in message

    # Reset door and try without key
    door = Door([4, 4], 1)
    allowed, message = door.pre_step_interaction(agent_no_key, direction)
    assert allowed is False
    assert door.obstacle is True
    assert "had no key" in message

    # Try with already unlocked door
    door.obstacle = False
    allowed, message = door.pre_step_interaction(agent_no_key, direction)
    assert allowed is True
    assert message is None


# --- Warp Tests ---
def test_warp_initialization():
    """Test Warp initialization."""
    pos = [1, 1]
    target = [8, 8]
    warp = Warp(pos, target)
    assert warp.pos == pos
    assert warp.target == target
    assert warp.obstacle is False
    assert warp.consumable is False
    assert warp.terminal is False
    assert warp.name == "warp pad"


def test_warp_copy():
    """Test Warp copy method."""
    warp = Warp([1, 1], [8, 8])
    warp_copy = warp.copy()
    assert warp is not warp_copy
    assert warp.pos == warp_copy.pos
    assert warp.target == warp_copy.target
    assert type(warp) == type(warp_copy)
    warp.pos = [2, 2]
    warp.target = [7, 7]
    assert warp_copy.pos == [1, 1]
    assert warp_copy.target == [8, 8]


def test_warp_interact():
    """Test Warp interaction."""
    start_pos = [1, 1]
    target_pos = [8, 8]
    agent = MockAgent(pos=start_pos)
    warp = Warp(start_pos, target_pos)
    message = warp.interact(agent)
    assert agent.pos == target_pos
    assert f"{agent.name} used a {warp.name} to teleport" in message


# --- Marker Tests ---
def test_marker_initialization():
    """Test Marker initialization."""
    pos = [3, 3]
    color = (255, 0, 0)
    marker = Marker(pos, color)
    assert marker.pos == pos
    assert marker.color == color
    assert marker.obstacle is False
    assert marker.consumable is False
    assert marker.terminal is False
    assert marker.name == "marker"


def test_marker_copy():
    """Test Marker copy method."""
    marker = Marker([3, 3], (255, 0, 0))
    marker_copy = marker.copy()
    assert marker is not marker_copy
    assert marker.pos == marker_copy.pos
    assert marker.color == marker_copy.color
    assert type(marker) == type(marker_copy)
    marker.pos = [4, 4]
    marker.color = (0, 255, 0)
    assert marker_copy.pos == [3, 3]
    assert marker_copy.color == (255, 0, 0)


def test_marker_interact():
    """Test Marker interaction (should do nothing specific)."""
    agent = MockAgent(pos=[3, 3])
    marker = Marker([3, 3], (255, 0, 0))
    initial_agent_state = agent.__dict__.copy()
    message = marker.interact(agent)
    assert message is None
    # Check agent state hasn't changed (except potentially 'done' if terminal=True)
    assert agent.pos == initial_agent_state["pos"]
    assert agent.reward == initial_agent_state["reward"]
    assert agent.inventory == initial_agent_state["inventory"]


# --- Other Tests ---
def test_other_initialization():
    """Test Other initialization."""
    pos = [6, 7]
    name = "gadget"
    other = Other(pos, name)
    assert other.pos == pos
    assert other.name == name
    assert other.obstacle is False
    assert other.consumable is True
    assert other.terminal is False


def test_other_copy():
    """Test Other copy method."""
    other = Other([6, 7], "gadget")
    other_copy = other.copy()
    assert other is not other_copy
    assert other.pos == other_copy.pos
    assert other.name == other_copy.name
    assert type(other) == type(other_copy)
    other.pos = [5, 5]
    other.name = "widget"
    assert other_copy.pos == [6, 7]
    assert other_copy.name == "gadget"


def test_other_interact():
    """Test Other interaction."""
    agent = MockAgent(pos=[6, 7])
    other = Other([6, 7], "gadget")
    message = other.interact(agent)
    assert other in agent.inventory
    assert f"{agent.name} collected {other.name}" in message


# --- Tree Tests ---
def test_tree_initialization():
    """Test Tree initialization."""
    pos = [5, 5]
    spawn_rate = 0.8
    spawn_radius = 1
    tree = Tree(pos, spawn_rate, spawn_radius)
    assert tree.pos == pos
    assert tree.spawn_rate == spawn_rate
    assert tree.spawn_radius == spawn_radius
    assert tree.obstacle is True
    assert tree.consumable is False
    assert tree.terminal is False
    assert tree.name == "tree"


def test_tree_copy():
    """Test Tree copy method."""
    tree = Tree([5, 5], 0.8, 1)
    tree_copy = tree.copy()
    assert tree is not tree_copy
    assert tree.pos == tree_copy.pos
    assert tree.spawn_rate == tree_copy.spawn_rate
    assert tree.spawn_radius == tree_copy.spawn_radius
    assert type(tree) == type(tree_copy)
    tree.pos = [6, 6]
    tree.spawn_rate = 0.1
    assert tree_copy.pos == [5, 5]
    assert tree_copy.spawn_rate == 0.8


def test_tree_step_no_spawn():
    """Test Tree step method when random chance fails."""
    env = MockEnv()
    tree = Tree([5, 5], spawn_rate=0.1)
    env.rng = np.random.RandomState(1)  # Seed guarantees > 0.1 first
    message = tree.step(env)
    assert "fruits" not in env.objects
    assert message is None


def test_tree_step_spawn_success():
    """Test Tree step method when fruit should spawn."""
    tree_pos = [5, 5]
    env = MockEnv(grid_shape=(10, 10))
    tree = Tree(tree_pos, spawn_rate=0.9, spawn_radius=1)
    env.rng = np.random.RandomState(1)  # Seed guarantees < 0.9 first
    env.set_check_target_return(True)  # Assume nearby spots are free

    message = tree.step(env)
    assert "fruits" in env.objects
    assert len(env.objects["fruits"]) == 1
    fruit = env.objects["fruits"][0]
    assert isinstance(fruit, Fruit)
    # Check fruit spawned within radius
    dx = abs(fruit.pos[0] - tree_pos[0])
    dy = abs(fruit.pos[1] - tree_pos[1])
    assert dx <= tree.spawn_radius and dy <= tree.spawn_radius
    assert message == "A fruit fell from a tree"


def test_tree_step_spawn_fail_no_space():
    """Test Tree step method when spawn chance hits but no valid space."""
    env = MockEnv(grid_shape=(10, 10))
    tree = Tree([5, 5], spawn_rate=0.9, spawn_radius=1)
    env.rng = np.random.RandomState(1)  # Seed guarantees < 0.9 first
    env.set_check_target_return(False)  # Mock that all nearby spots are blocked

    message = tree.step(env)
    assert "fruits" not in env.objects
    assert message is None


# --- Fruit Tests ---
def test_fruit_initialization():
    """Test Fruit initialization."""
    pos = [6, 6]
    fruit = Fruit(pos)
    assert fruit.pos == pos
    assert fruit.obstacle is False
    assert fruit.consumable is True
    assert fruit.terminal is False
    assert fruit.name == "fruit"


def test_fruit_copy():
    """Test Fruit copy method."""
    fruit = Fruit([6, 6])
    fruit_copy = fruit.copy()
    assert fruit is not fruit_copy
    assert fruit.pos == fruit_copy.pos
    assert type(fruit) == type(fruit_copy)
    fruit.pos = [7, 7]
    assert fruit_copy.pos == [6, 6]


def test_fruit_interact():
    """Test Fruit interaction."""
    agent = MockAgent(pos=[6, 6])
    fruit = Fruit([6, 6])
    message = fruit.interact(agent)
    assert fruit in agent.inventory
    assert f"{agent.name} collected a {fruit.name}" in message


# --- Box Tests ---
def test_box_initialization():
    """Test Box initialization."""
    pos = [7, 7]
    box = Box(pos)
    assert box.pos == pos
    assert box.contents == []
    assert box.obstacle is False
    assert box.consumable is False
    assert box.terminal is False
    assert box.name == "box"


def test_box_copy():
    """Test Box copy method."""
    box = Box([7, 7])
    item = Key([0, 0])  # Example item
    box.contents.append(item)
    box_copy = box.copy()
    assert box is not box_copy
    assert box.pos == box_copy.pos
    assert box.contents == box_copy.contents
    assert box.contents is not box_copy.contents  # Ensure list is copied
    assert type(box) == type(box_copy)

    # Modify original
    box.pos = [8, 8]
    box.contents.append(Fruit([1, 1]))
    # Check copy is unaffected
    assert box_copy.pos == [7, 7]
    assert len(box_copy.contents) == 1
    assert isinstance(box_copy.contents[0], Key)


def test_box_interact_put_item():
    """Test putting an item into the box."""
    item = Key([9, 9])
    agent = MockAgent(pos=[7, 7], inventory=[item])
    box = Box([7, 7])
    message = box.interact(agent)
    assert not agent.inventory  # Agent inventory should be empty
    assert item in box.contents  # Item should be in the box
    assert f"{agent.name} put {item.name} in the box" in message


def test_box_interact_take_item():
    """Test taking an item from the box."""
    item = Fruit([9, 9])
    agent = MockAgent(pos=[7, 7], inventory=[])
    box = Box([7, 7])
    box.contents.append(item)
    message = box.interact(agent)
    assert not box.contents  # Box should be empty
    assert item in agent.inventory  # Agent should have the item
    assert f"{agent.name} took {item.name} from the box" in message


def test_box_interact_empty():
    """Test interacting with an empty box when agent has nothing."""
    agent = MockAgent(pos=[7, 7], inventory=[])
    box = Box([7, 7])
    message = box.interact(agent)
    assert not box.contents
    assert not agent.inventory
    assert "but it was empty" in message


# --- Sign Tests ---
def test_sign_initialization():
    """Test Sign initialization."""
    pos = [8, 8]
    message = "Beware of Grues"
    sign = Sign(pos, message)
    assert sign.pos == pos
    assert sign.message == message
    assert sign.obstacle is False
    assert sign.consumable is False
    assert sign.terminal is False
    assert sign.name == "sign"


def test_sign_copy():
    """Test Sign copy method."""
    sign = Sign([8, 8], "Go North")
    sign_copy = sign.copy()
    assert sign is not sign_copy
    assert sign.pos == sign_copy.pos
    assert sign.message == sign_copy.message
    assert type(sign) == type(sign_copy)
    sign.pos = [9, 9]
    sign.message = "Go South"
    assert sign_copy.pos == [8, 8]
    assert sign_copy.message == "Go North"


def test_sign_interact():
    """Test Sign interaction."""
    agent = MockAgent(pos=[8, 8])
    message_text = "Exit this way"
    sign = Sign([8, 8], message_text)
    message = sign.interact(agent)
    assert f"read the sign: '{message_text}'" in message


# --- PushableBox Tests ---
def test_pushable_box_initialization():
    """Test PushableBox initialization."""
    pos = [1, 5]
    pbox = PushableBox(pos)
    assert pbox.pos == pos
    assert pbox.obstacle is True
    assert pbox.consumable is False
    assert pbox.terminal is False
    assert pbox.name == "pushable box"
    assert pbox.being_pushed is False


def test_pushable_box_copy():
    """Test PushableBox copy method."""
    pbox = PushableBox([1, 5])
    pbox.being_pushed = True  # Set a non-default state
    pbox_copy = pbox.copy()
    assert pbox is not pbox_copy
    assert pbox.pos == pbox_copy.pos
    assert pbox.being_pushed == pbox_copy.being_pushed
    assert type(pbox) == type(pbox_copy)

    pbox.pos = [1, 6]
    pbox.being_pushed = False
    assert pbox_copy.pos == [1, 5]
    assert pbox_copy.being_pushed is True


def test_pushable_box_pre_step_success():
    """Test successful push pre-step interaction."""
    box_pos = [1, 5]
    agent_pos = [1, 4]
    agent_looking_idx = 1  # East
    push_direction = [0, 1]  # East
    target_box_pos = [1, 6]

    agent = MockAgent(pos=agent_pos)
    agent.looking = agent_looking_idx  # Ensure agent is looking the right way
    pbox = PushableBox(box_pos)
    env = MockEnv(grid_shape=(10, 10))
    env.set_check_target_return(True)  # Assume target pos is free

    allowed, message = pbox.pre_step_interaction(agent, push_direction, env)

    assert allowed is True
    assert pbox.pos == target_box_pos
    assert pbox.being_pushed is True
    assert f"{agent.name} pushed a {pbox.name}" in message


def test_pushable_box_pre_step_fail_blocked():
    """Test failed push pre-step interaction due to blockage."""
    box_pos = [1, 5]
    agent_pos = [1, 4]
    agent_looking_idx = 1  # East
    push_direction = [0, 1]  # East
    target_box_pos = [1, 6]  # Position where the box would land
    blocker_pos = target_box_pos  # Position of the blocking object

    agent = MockAgent(pos=agent_pos)
    agent.looking = agent_looking_idx
    pbox = PushableBox(box_pos)
    # Add a blocking object (e.g., a Wall) at the target push location
    blocker = Wall(blocker_pos)
    env = MockEnv(grid_shape=(10, 10), objects={"walls": [blocker]})
    # No longer need env.set_check_target_return(False)

    allowed, message = pbox.pre_step_interaction(agent, push_direction, env)

    assert allowed is False
    assert pbox.pos == box_pos  # Position shouldn't change
    assert pbox.being_pushed is False
    # Check for the more specific message from the implementation
    assert f"blocked by {blocker.name}" in message


def test_pushable_box_interact():
    """Test direct interaction with PushableBox (resets being_pushed)."""
    agent = MockAgent(pos=[1, 5])
    pbox = PushableBox([1, 5])
    pbox.being_pushed = True  # Set state
    message = pbox.interact(agent)
    assert pbox.being_pushed is False
    assert message is None  # interact returns None


def test_pushable_box_step():
    """Test step method for PushableBox (resets being_pushed)."""
    pbox = PushableBox([1, 5])
    pbox.being_pushed = True  # Set state
    env = MockEnv()
    pbox.step(env)
    # The PushableBox.step method explicitly sets being_pushed to False
    assert pbox.being_pushed is False


# --- LinkedDoor Tests ---
def test_linked_door_initialization():
    """Test LinkedDoor initialization."""
    pos = [2, 3]
    linked_id = "door1"
    orientation = 0
    door = LinkedDoor(pos, linked_id, orientation)
    assert door.pos == pos
    assert door.linked_id == linked_id
    assert door.orientation == orientation
    assert door.obstacle is True  # Starts closed
    assert door.is_open is False
    assert door.consumable is False
    assert door.terminal is False
    assert door.name == "linked door"


def test_linked_door_copy():
    """Test LinkedDoor copy method."""
    door = LinkedDoor([2, 3], "door1", 0)
    door.obstacle = False
    door.is_open = True
    door_copy = door.copy()
    assert door is not door_copy
    assert door.pos == door_copy.pos
    assert door.linked_id == door_copy.linked_id
    assert door.orientation == door_copy.orientation
    assert door.obstacle == door_copy.obstacle
    assert door.is_open == door_copy.is_open
    assert type(door) == type(door_copy)

    door.pos = [2, 4]
    door.obstacle = True
    door.is_open = False
    assert door_copy.pos == [2, 3]
    assert door_copy.obstacle is False
    assert door_copy.is_open is True


def test_linked_door_activate_deactivate():
    """Test activating and deactivating a LinkedDoor."""
    door = LinkedDoor([2, 3], "door1")
    activator = Object([0, 0])  # Mock activator
    activator.name = "TestActivator"
    deactivator = Object([0, 0])  # Mock deactivator
    deactivator.name = "TestDeactivator"

    # Activate
    message = door.activate(activator)
    assert door.obstacle is False
    assert door.is_open is True
    assert f"opened by {activator.name}" in message

    # Try activating again (should do nothing)
    message = door.activate(activator)
    assert message is None

    # Deactivate
    message = door.deactivate(deactivator)
    assert door.obstacle is True
    assert door.is_open is False
    assert f"closed" in message  # Basic check, specific reasons tested with plate/lever

    # Try deactivating again (should do nothing)
    message = door.deactivate(deactivator)
    assert message is None


def test_linked_door_interact_closed():
    """Test agent interaction with a closed LinkedDoor."""
    agent = MockAgent(pos=[2, 3])
    door = LinkedDoor([2, 3], "door1")
    message = door.interact(agent)
    assert door.obstacle is True
    assert "tried to open a locked linked door" in message


def test_linked_door_interact_open():
    """Test agent interaction with an open LinkedDoor."""
    agent = MockAgent(pos=[2, 3])
    door = LinkedDoor([2, 3], "door1")
    door.activate()  # Open the door
    message = door.interact(agent)
    assert door.obstacle is False
    assert message is None  # Agent can pass through


# --- PressurePlate Tests ---
def test_pressure_plate_initialization():
    """Test PressurePlate initialization."""
    pos = [4, 5]
    target_id = "door1"
    plate = PressurePlate(pos, target_id)
    assert plate.pos == pos
    assert plate.target_linked_id == target_id
    assert plate.obstacle is False
    assert plate.consumable is False
    assert plate.terminal is False
    assert plate.name == "pressure plate"
    assert plate.is_pressed is False


def test_pressure_plate_copy():
    """Test PressurePlate copy method."""
    plate = PressurePlate([4, 5], "door1")
    plate.is_pressed = True
    plate_copy = plate.copy()
    assert plate is not plate_copy
    assert plate.pos == plate_copy.pos
    assert plate.target_linked_id == plate_copy.target_linked_id
    assert plate.is_pressed == plate_copy.is_pressed
    assert type(plate) == type(plate_copy)

    plate.pos = [4, 6]
    plate.is_pressed = False
    assert plate_copy.pos == [4, 5]
    assert plate_copy.is_pressed is True


def test_pressure_plate_step_agent_press_release():
    """Test agent pressing and releasing a pressure plate."""
    plate_pos = [4, 5]
    door_pos = [2, 3]
    target_id = "door1"

    agent = MockAgent(pos=[0, 0])  # Start away from plate
    plate = PressurePlate(plate_pos, target_id)
    door = LinkedDoor(door_pos, target_id)
    env = MockEnv(
        objects={"pressure_plates": [plate], "linked_doors": [door]}, agents=[agent]
    )

    # Step 1: Agent not on plate
    message = plate.step(env)
    assert plate.is_pressed is False
    assert door.is_open is False
    assert message is None

    # Step 2: Agent moves onto plate
    agent.pos = plate_pos
    message = plate.step(env)
    assert plate.is_pressed is True
    assert door.is_open is True
    assert f"{agent.name} pressed {plate.name}" in message
    assert f"{door.name} (ID: {target_id}) was opened by {plate.name}" in message

    # Step 3: Agent still on plate
    message = plate.step(env)
    assert plate.is_pressed is True
    assert door.is_open is True
    assert message is None  # No state change

    # Step 4: Agent moves off plate
    agent.pos = [0, 0]
    message = plate.step(env)
    assert plate.is_pressed is False
    assert door.is_open is False
    assert (
        f"{door.name} (ID: {target_id}) was closed because pressure was released"
        in message
    )


def test_pressure_plate_step_box_press_release():
    """Test pushable box pressing and releasing a pressure plate."""
    plate_pos = [4, 5]
    door_pos = [2, 3]
    box_start_pos = [0, 0]
    target_id = "door1"

    plate = PressurePlate(plate_pos, target_id)
    door = LinkedDoor(door_pos, target_id)
    box = PushableBox(box_start_pos)
    # Need an agent to 'push' the box, but agent position isn't checked by plate directly
    agent = MockAgent(pos=[-1, -1])
    env = MockEnv(
        objects={
            "pressure_plates": [plate],
            "linked_doors": [door],
            "pushable_boxes": [box],
        },
        agents=[agent],
    )

    # Step 1: Box not on plate
    message = plate.step(env)
    assert plate.is_pressed is False
    assert door.is_open is False
    assert message is None

    # Step 2: Box moved onto plate
    box.pos = plate_pos
    message = plate.step(env)
    assert plate.is_pressed is True
    assert door.is_open is True
    assert f"{box.name} pressed {plate.name}" in message  # Box name should be detected
    assert f"{door.name} (ID: {target_id}) was opened by {plate.name}" in message

    # Step 3: Box still on plate
    message = plate.step(env)
    assert plate.is_pressed is True
    assert door.is_open is True
    assert message is None

    # Step 4: Box moved off plate
    box.pos = [0, 0]
    message = plate.step(env)
    assert plate.is_pressed is False
    assert door.is_open is False
    assert (
        f"{door.name} (ID: {target_id}) was closed because pressure was released"
        in message
    )


def test_pressure_plate_interact():
    """Test agent interact method on pressure plate (should do nothing directly)."""
    plate_pos = [4, 5]
    target_id = "door1"
    agent = MockAgent(pos=plate_pos)
    plate = PressurePlate(plate_pos, target_id)
    door = LinkedDoor([2, 3], target_id)
    env = MockEnv(
        objects={"pressure_plates": [plate], "linked_doors": [door]}, agents=[agent]
    )

    message = plate.interact(agent, env)
    assert message is None
    # State change happens in step(), not interact()
    assert plate.is_pressed is False
    assert door.is_open is False


# --- Lever Tests ---
def test_lever_initialization():
    """Test Lever initialization."""
    pos = [6, 7]
    target_id = "door2"
    lever = Lever(pos, target_id)
    assert lever.pos == pos
    assert lever.target_linked_id == target_id
    assert lever.obstacle is False
    assert lever.consumable is False
    assert lever.terminal is False
    assert lever.name == "lever"
    assert lever.activated is False


def test_lever_copy():
    """Test Lever copy method."""
    lever = Lever([6, 7], "door2")
    lever.activated = True
    lever_copy = lever.copy()
    assert lever is not lever_copy
    assert lever.pos == lever_copy.pos
    assert lever.target_linked_id == lever_copy.target_linked_id
    assert lever.activated == lever_copy.activated
    assert type(lever) == type(lever_copy)

    lever.pos = [6, 8]
    lever.activated = False
    assert lever_copy.pos == [6, 7]
    assert lever_copy.activated is True


def test_lever_interact_toggle():
    """Test agent interacting with a lever to toggle a linked door."""
    lever_pos = [6, 7]
    door_pos = [1, 1]
    target_id = "door2"

    agent = MockAgent(pos=lever_pos)  # Agent needs to be at lever pos for interact
    lever = Lever(lever_pos, target_id)
    door = LinkedDoor(door_pos, target_id)
    env = MockEnv(objects={"levers": [lever], "linked_doors": [door]}, agents=[agent])

    # Interaction 1: Activate lever, open door
    message = lever.interact(agent, env)
    assert lever.activated is True
    assert door.is_open is True
    assert door.obstacle is False
    assert f"{agent.name} activated the {lever.name}" in message
    assert f"{door.name} (ID: {target_id}) was opened by {lever.name}" in message

    # Interaction 2: Deactivate lever, close door
    message = lever.interact(agent, env)
    assert lever.activated is False
    assert door.is_open is False
    assert door.obstacle is True
    assert f"{agent.name} deactivated the {lever.name}" in message
    assert f"{door.name} (ID: {target_id}) was closed by {lever.name}" in message

    # Interaction 3: Activate lever again, open door again
    message = lever.interact(agent, env)
    assert lever.activated is True
    assert door.is_open is True
    assert door.obstacle is False
    assert f"{agent.name} activated the {lever.name}" in message
    assert f"{door.name} (ID: {target_id}) was opened by {lever.name}" in message


def test_lever_interact_no_target():
    """Test interacting with a lever that has no matching target."""
    lever_pos = [6, 7]
    target_id = "nonexistent_door"

    agent = MockAgent(pos=lever_pos)
    lever = Lever(lever_pos, target_id)
    env = MockEnv(objects={"levers": [lever]}, agents=[agent])  # No door in env

    # Interaction 1: Activate lever
    message = lever.interact(agent, env)
    assert lever.activated is True
    assert f"{agent.name} activated the {lever.name}" in message
    assert "opened" not in message  # No door message
    assert "closed" not in message

    # Interaction 2: Deactivate lever
    message = lever.interact(agent, env)
    assert lever.activated is False
    assert f"{agent.name} deactivated the {lever.name}" in message
    assert "opened" not in message
    assert "closed" not in message


# --- ResetButton Tests ---
def test_reset_button_initialization():
    """Test ResetButton initialization."""
    pos = [9, 9]
    button = ResetButton(pos)
    assert button.pos == pos
    assert button.obstacle is False
    assert button.consumable is False
    assert button.terminal is False
    assert button.name == "reset button"


def test_reset_button_copy():
    """Test ResetButton copy method."""
    button = ResetButton([9, 9])
    button_copy = button.copy()
    assert button is not button_copy
    assert button.pos == button_copy.pos
    assert type(button) == type(button_copy)
    button.pos = [8, 8]
    assert button_copy.pos == [9, 9]


def test_reset_button_interact():
    """Test ResetButton interaction resets environment objects."""
    button_pos = [9, 9]
    agent_pos = button_pos
    initial_obj_pos = [1, 1]
    modified_obj_pos = [5, 5]

    # Define initial and modified states
    initial_objects = {"walls": [Wall(initial_obj_pos)]}
    modified_objects = {"walls": [Wall(modified_obj_pos)]}

    agent = MockAgent(pos=agent_pos)
    button = ResetButton(button_pos)
    env = MockEnv(objects=modified_objects, agents=[agent])
    # Manually set the initial_objects attribute for the mock env
    env.initial_objects = copy.deepcopy(initial_objects)

    # Interact with the button
    message = button.interact(agent, env)

    # Check message
    assert f"{agent.name} pressed the {button.name}" in message
    assert "Objects reset to initial state" in message

    # Check if env.objects were reset
    assert "walls" in env.objects
    assert len(env.objects["walls"]) == 1
    assert env.objects["walls"][0].pos == initial_obj_pos


def test_reset_button_interact_no_initial_state():
    """Test ResetButton interaction when env has no initial_objects."""
    button_pos = [9, 9]
    agent_pos = button_pos
    current_obj_pos = [5, 5]

    current_objects = {"walls": [Wall(current_obj_pos)]}

    agent = MockAgent(pos=agent_pos)
    button = ResetButton(button_pos)
    env = MockEnv(objects=current_objects, agents=[agent])
    # Ensure initial_objects is None or not set
    env.initial_objects = None

    # Interact with the button
    message = button.interact(agent, env)

    # Check message indicates failure
    assert f"{agent.name} tried to press the {button.name}" in message
    assert "initial state was not found" in message

    # Check that objects were NOT reset
    assert "walls" in env.objects
    assert len(env.objects["walls"]) == 1
    assert env.objects["walls"][0].pos == current_obj_pos
