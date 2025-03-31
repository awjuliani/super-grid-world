import pytest
import numpy as np
from sgw.env import SuperGridWorld
from sgw.enums import ObsType, ControlType, Action
from sgw.object import Wall, Reward


@pytest.fixture
def default_env():
    """Fixture for a default SuperGridWorld environment."""
    return SuperGridWorld(seed=42)


@pytest.fixture
def empty_env():
    """Fixture for an empty SuperGridWorld environment."""
    return SuperGridWorld(template_name="empty", seed=42)


def test_env_default_initialization(default_env):
    """Test default environment initialization."""
    env = default_env
    assert env.grid_shape == (9, 9)
    assert env.obs_type == ObsType.visual_2d
    assert env.control_type == ControlType.allocentric
    assert env.num_agents == 1
    assert len(env.agents) == 1
    assert env.agents[0] is None  # Agent is created during reset
    assert Action.MOVE_NORTH in env.valid_actions
    assert Action.MOVE_SOUTH in env.valid_actions
    assert Action.MOVE_WEST in env.valid_actions
    assert Action.MOVE_EAST in env.valid_actions
    assert Action.NOOP not in env.valid_actions
    assert Action.INTERACT not in env.valid_actions


def test_env_custom_initialization():
    """Test environment initialization with custom parameters."""
    env = SuperGridWorld(
        grid_shape=(5, 7),
        obs_type=ObsType.symbolic,
        control_type=ControlType.egocentric,
        seed=123,
        use_noop=True,
        manual_interact=True,
        num_agents=2,
        agent_names=["agent_a", "agent_b"],
    )
    assert env.grid_shape == (5, 7)
    assert env.obs_type == ObsType.symbolic
    assert env.control_type == ControlType.egocentric
    assert env.num_agents == 2
    assert len(env.agents) == 2
    assert env.agents[0] is None
    assert env.agents[1] is None
    assert env.agent_names == ["agent_a", "agent_b"]
    assert Action.MOVE_FORWARD in env.valid_actions
    # Note: No MOVE_BACKWARD action defined in enum
    assert Action.ROTATE_LEFT in env.valid_actions
    assert Action.ROTATE_RIGHT in env.valid_actions
    assert Action.NOOP in env.valid_actions
    assert Action.INTERACT in env.valid_actions
    assert (
        Action.MOVE_NORTH not in env.valid_actions
    )  # Check allocentric action isn't present


def test_env_reset_default(empty_env):
    """Test resetting the environment to default start."""
    env = empty_env
    obs = env.reset()
    assert env.episode_time == 0
    assert len(env.agents) == 1
    agent = env.agents[0]
    assert agent is not None
    assert agent.pos == env.agent_start_pos
    assert agent.done is False
    assert agent.reward == 0  # Corrected attribute name
    assert len(obs) == env.num_agents
    # Check observation space shape (assuming default 2D visual)
    expected_shape = env.observation_space.spaces[0].shape
    assert obs[0].shape == expected_shape


def test_env_reset_custom_position(empty_env):
    """Test resetting the environment with a custom agent position."""
    env = empty_env
    custom_pos = [3, 3]
    obs = env.reset(agent_positions=[custom_pos])
    assert env.agents[0].pos == custom_pos
    assert len(obs) == 1


def test_env_reset_random_start(empty_env):
    """Test resetting with random agent start position."""
    env = empty_env
    # Add a wall to limit free spots for predictability
    env.template_objects["walls"] = [Wall([1, 1])]
    obs = env.reset(random_start=True)
    # Agent should not start at the default start or the wall position
    assert env.agents[0].pos != env.agent_start_pos
    assert env.agents[0].pos != [1, 1]
    assert env.grid_shape[0] > env.agents[0].pos[0] >= 0
    assert env.grid_shape[1] > env.agents[0].pos[1] >= 0
    assert len(obs) == 1


def test_env_reset_objects(empty_env):
    """Test resetting with custom objects."""
    env = empty_env
    custom_objects = {
        "walls": [Wall([1, 1]), Wall([1, 2])],
        "rewards": [Reward([2, 2], 5)],
    }
    env.reset(objects=custom_objects)
    assert len(env.objects["walls"]) == 2
    assert isinstance(env.objects["walls"][0], Wall)
    assert env.objects["walls"][0].pos == [1, 1]
    assert len(env.objects["rewards"]) == 1
    assert isinstance(env.objects["rewards"][0], Reward)
    assert env.objects["rewards"][0].pos == [2, 2]
    assert env.objects["rewards"][0].value == 5


# --- Tests for Helper Methods ---


def test_env_check_target(empty_env):
    """Test the check_target method for boundary and obstacle checks."""
    env = empty_env
    env.reset(objects={"walls": [Wall([2, 2])]})
    agent = env.agents[0]

    # Check valid positions
    assert env.check_target([1, 1]) is True
    assert env.check_target([0, 0]) is True
    assert env.check_target([env.grid_height - 1, env.grid_width - 1]) is True

    # Check boundaries
    assert env.check_target([-1, 1]) is False  # Out of bounds (row)
    assert env.check_target([1, -1]) is False  # Out of bounds (col)
    assert env.check_target([env.grid_height, 1]) is False  # Out of bounds (row)
    assert env.check_target([1, env.grid_width]) is False  # Out of bounds (col)

    # Check obstacle
    assert env.check_target([2, 2]) is False  # Wall position

    # Check non-obstacle (e.g., reward) - should be valid by default
    env.objects["rewards"] = [Reward([3, 3], 1)]
    assert env.check_target([3, 3]) is True

    # Check non-obstacle when obstacles_only=False
    assert env.check_target([3, 3], obstacles_only=False) is False


def test_env_free_spots(empty_env):
    """Test the free_spots property."""
    env = empty_env
    env.reset(objects={"walls": [Wall([1, 1]), Wall([1, 2])]})
    free = env.free_spots
    total_spots = env.grid_height * env.grid_width
    # Should have total spots minus wall spots
    assert len(free) == total_spots - 2
    assert [1, 1] not in free
    assert [1, 2] not in free
    assert [0, 0] in free
    assert [2, 2] in free


def test_env_find_object_at(empty_env):
    """Test the _find_object_at method."""
    env = empty_env
    wall = Wall([2, 3])
    reward = Reward([4, 5], 10)
    env.reset(objects={"walls": [wall], "rewards": [reward]})

    internal_wall = env.objects["walls"][0]
    internal_reward = env.objects["rewards"][0]
    assert env._find_object_at([2, 3]) is internal_wall
    assert env._find_object_at([4, 5]) is internal_reward
    assert env._find_object_at([1, 1]) is None
    assert env._find_object_at([2, 4]) is None


# --- Tests for Step Method ---


@pytest.fixture
def step_env():
    """Fixture for an environment specifically for step tests."""
    env = SuperGridWorld(
        grid_shape=(5, 5),
        template_name="empty",
        control_type=ControlType.allocentric,  # Use allocentric for simpler action mapping
        add_outer_walls=False,  # Avoid outer walls complicating positions
        seed=42,
    )
    # Place a wall and a reward for interaction tests
    env.reset(objects={"walls": [Wall([1, 2])], "rewards": [Reward([2, 1], 10)]})
    # Action mapping for allocentric: 0:N, 1:E, 2:S, 3:W
    return env


def test_env_step_move(step_env):
    """Test basic agent movement via step."""
    env = step_env
    agent = env.agents[0]
    initial_pos = agent.pos.copy()  # e.g., [2, 2]

    # Move North (action index 0 for MOVE_NORTH)
    obs, rewards, dones, info = env.step([0])

    assert agent.pos == [initial_pos[0] - 1, initial_pos[1]]  # Should be [1, 2]
    assert rewards == [0.0]  # Default time penalty is 0
    assert dones == [False]
    assert isinstance(info, dict) and "events" in info
    assert env.episode_time == 1


def test_env_step_hit_wall(step_env):
    """Test agent hitting a wall during step."""
    env = step_env
    agent = env.agents[0]
    agent.pos = [1, 1]  # Position agent south of the wall at [1, 2]

    # Attempt to move East (action index 1 for MOVE_EAST) into the wall
    obs, rewards, dones, info = env.step([1])

    assert agent.pos == [1, 1]  # Position should not change
    assert rewards == [0.0]
    assert dones == [False]
    assert env.episode_time == 1


def test_env_step_collect_reward(step_env):
    """Test agent collecting a reward during step."""
    env = step_env
    agent = env.agents[0]
    reward_pos = [2, 1]
    agent.pos = [1, 1]  # Position agent North of the reward
    reward_obj = env.objects["rewards"][0]
    assert reward_obj.pos == reward_pos
    assert len(env.objects["rewards"]) == 1

    # Move South (action index 2 for MOVE_SOUTH) onto the reward
    obs, rewards, dones, info = env.step([2])

    assert agent.pos == reward_pos
    assert rewards == [10.0]  # Reward value collected
    assert dones == [False]
    assert env.agents[0].reward == 10.0  # Check agent's internal reward state
    assert len(env.objects.get("rewards", [])) == 0  # Reward should be consumed
    assert "collected reward" in info["events"][0].lower()
    assert env.episode_time == 1


def test_env_step_time_penalty(step_env):
    """Test time penalty application during step."""
    env = step_env
    env.reset(time_penalty=-0.1)  # Set a time penalty
    agent = env.agents[0]
    initial_pos = agent.pos.copy()

    # Move North (action index 0)
    obs, rewards, dones, info = env.step([0])

    assert agent.pos == [initial_pos[0] - 1, initial_pos[1]]
    assert rewards == [-0.1]  # Should receive the time penalty
    assert dones == [False]
    assert env.episode_time == 1


def test_env_step_max_time(step_env):
    """Test episode termination due to interaction with terminal object."""
    max_time = 3
    env = step_env
    # Rerun with slightly different logic - interact causes done
    terminal_reward = Reward([1, 1], 0)
    terminal_reward.terminal = True  # Make reward terminal
    env.reset(episode_length=max_time, objects={"terminals": [terminal_reward]})
    agent = env.agents[0]
    agent.pos = [2, 1]  # Start just below the terminal reward

    for t in range(max_time):
        obs, rewards, dones, info = env.step([0])  # Move North (Action.MOVE_NORTH)
        if agent.pos == [1, 1]:  # If agent reached terminal object
            assert dones[0] is True  # <-- Fix: Check the specific agent's done status
            break
    else:
        pytest.fail(
            "Agent did not reach terminal object or episode did not terminate correctly"
        )
