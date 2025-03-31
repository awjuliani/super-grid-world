import pytest
import numpy as np
from sgw.agent import Agent
from sgw.enums import Action, ControlType
from sgw.object import (
    Key,
    Reward,
)  # Assuming Reward might be used indirectly or in future tests


@pytest.fixture
def egocentric_agent():
    """Fixture for an agent with egocentric control."""
    return Agent(
        pos=[1, 1], direction=0, control_type=ControlType.egocentric, name="Ego"
    )


@pytest.fixture
def allocentric_agent():
    """Fixture for an agent with allocentric control."""
    return Agent(
        pos=[5, 5], direction=1, control_type=ControlType.allocentric, name="Allo"
    )


def test_agent_initialization(egocentric_agent, allocentric_agent):
    """Test agent initialization with different control types."""
    ego = egocentric_agent
    assert ego.pos == [1, 1]
    assert ego.orientation == 0  # North
    assert ego.looking == 0
    assert ego.inventory == []
    assert ego.reward == 0
    assert ego.done is False
    assert ego.name == "Ego"
    assert ego.control_type == ControlType.egocentric
    assert ego.direction_map.shape == (6, 2)

    allo = allocentric_agent
    assert allo.pos == [5, 5]
    assert allo.orientation == 1  # East
    assert allo.looking == 1
    assert allo.control_type == ControlType.allocentric
    assert allo.name == "Allo"


def test_agent_rotate(egocentric_agent):
    """Test agent rotation."""
    agent = egocentric_agent
    assert agent.orientation == 0  # North
    agent.rotate(1)  # Rotate right
    assert agent.orientation == 1  # East
    assert agent.looking == 1
    agent.rotate(1)
    assert agent.orientation == 2  # South
    assert agent.looking == 2
    agent.rotate(1)
    assert agent.orientation == 3  # West
    assert agent.looking == 3
    agent.rotate(1)
    assert agent.orientation == 0  # North
    assert agent.looking == 0
    agent.rotate(-1)  # Rotate left
    assert agent.orientation == 3  # West
    assert agent.looking == 3


def test_agent_move(egocentric_agent):
    """Test direct agent movement."""
    agent = egocentric_agent
    initial_pos = agent.pos.copy()
    move_success = agent.move(direction=[0, 1])  # Move East
    assert move_success is True
    assert agent.pos == [initial_pos[0], initial_pos[1] + 1]

    move_none = agent.move(direction=None)
    assert move_none is False
    assert agent.pos == [initial_pos[0], initial_pos[1] + 1]  # Position unchanged


def test_agent_process_action_egocentric(egocentric_agent):
    """Test processing actions for an egocentric agent."""
    agent = egocentric_agent
    agent.pos = [2, 2]
    agent.orientation = 0  # North

    # Test MOVE_FORWARD
    direction = agent.process_action(Action.MOVE_FORWARD)
    np.testing.assert_array_equal(direction, [-1, 0])  # North movement
    agent.move(direction)
    assert agent.pos == [1, 2]

    # Test ROTATE_RIGHT
    direction = agent.process_action(Action.ROTATE_RIGHT)
    assert direction is None
    assert agent.orientation == 1  # East
    assert agent.looking == 1

    # Test MOVE_FORWARD after rotation
    direction = agent.process_action(Action.MOVE_FORWARD)
    np.testing.assert_array_equal(direction, [0, 1])  # East movement
    agent.move(direction)
    assert agent.pos == [1, 3]

    # Test ROTATE_LEFT
    direction = agent.process_action(Action.ROTATE_LEFT)
    assert direction is None
    assert agent.orientation == 0  # North
    assert agent.looking == 0

    # Test NOOP
    direction = agent.process_action(Action.NOOP)
    assert direction is None
    assert agent.pos == [1, 3]  # Position unchanged
    assert agent.orientation == 0  # Orientation unchanged


def test_agent_process_action_allocentric(allocentric_agent):
    """Test processing actions for an allocentric agent."""
    agent = allocentric_agent
    agent.pos = [5, 5]
    agent.orientation = 1  # East (should not affect movement)

    # Test MOVE_NORTH
    direction = agent.process_action(Action.MOVE_NORTH)
    np.testing.assert_array_equal(direction, [-1, 0])
    assert agent.looking == 0  # North
    agent.move(direction)
    assert agent.pos == [4, 5]

    # Test MOVE_EAST
    direction = agent.process_action(Action.MOVE_EAST)
    np.testing.assert_array_equal(direction, [0, 1])
    assert agent.looking == 1  # East
    agent.move(direction)
    assert agent.pos == [4, 6]

    # Test MOVE_SOUTH
    direction = agent.process_action(Action.MOVE_SOUTH)
    np.testing.assert_array_equal(direction, [1, 0])
    assert agent.looking == 2  # South
    agent.move(direction)
    assert agent.pos == [5, 6]

    # Test MOVE_WEST
    direction = agent.process_action(Action.MOVE_WEST)
    np.testing.assert_array_equal(direction, [0, -1])
    assert agent.looking == 3  # West
    agent.move(direction)
    assert agent.pos == [5, 5]

    # Test NOOP
    direction = agent.process_action(Action.NOOP)
    assert direction is None
    assert agent.pos == [5, 5]  # Position unchanged


def test_agent_move_helpers(egocentric_agent, allocentric_agent):
    """Test move_forward and move_allocentric helpers."""
    # Egocentric
    agent_ego = egocentric_agent
    agent_ego.pos = [2, 2]
    agent_ego.orientation = 0  # North
    agent_ego.move_forward()
    assert agent_ego.pos == [1, 2]
    assert agent_ego.looking == 0

    # Allocentric
    agent_allo = allocentric_agent
    agent_allo.pos = [5, 5]
    agent_allo.orientation = 0  # North (should not affect allocentric move)
    agent_allo.move_allocentric(1)  # Move East (index 1)
    assert agent_allo.pos == [5, 6]
    assert agent_allo.looking == 1  # East


def test_agent_collect_object(egocentric_agent):
    """Test collecting objects."""
    agent = egocentric_agent
    key = Key([0, 0])
    assert len(agent.inventory) == 0
    agent.collect_object(key)
    assert len(agent.inventory) == 1
    assert agent.inventory[0] is key


def test_agent_use_key(egocentric_agent):
    """Test using a key from inventory."""
    agent = egocentric_agent
    key1 = Key([0, 0])
    key2 = Key([0, 1])  # Another key

    # Test using key when none available
    assert agent.use_key() is False
    assert len(agent.inventory) == 0

    # Collect keys
    agent.collect_object(key1)
    agent.collect_object(key2)
    assert len(agent.inventory) == 2

    # Test using one key
    assert agent.use_key() is True
    assert len(agent.inventory) == 1
    assert key2 in agent.inventory  # key1 should be removed
    assert key1 not in agent.inventory

    # Test using the second key
    assert agent.use_key() is True
    assert len(agent.inventory) == 0

    # Test using key when empty again
    assert agent.use_key() is False


def test_agent_collect_reward(egocentric_agent):
    """Test reward collection."""
    agent = egocentric_agent
    assert agent.reward == 0
    agent.collect_reward(10.0)
    assert agent.reward == 10.0
    agent.collect_reward(-5.0)
    assert agent.reward == 5.0


def test_agent_get_position(egocentric_agent):
    """Test getting agent position."""
    agent = egocentric_agent
    agent.pos = [3, 7]
    assert agent.get_position() == [3, 7]
    # Ensure it returns a copy
    pos_copy = agent.get_position()
    pos_copy[0] = 99
    assert agent.pos == [3, 7]


def test_agent_teleport(egocentric_agent):
    """Test agent teleportation."""
    agent = egocentric_agent
    agent.pos = [1, 1]
    agent.teleport([8, 8])
    assert agent.pos == [8, 8]
    agent.teleport((7, 7))  # Test with tuple
    assert agent.pos == [7, 7]
