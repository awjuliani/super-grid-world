import sgw.utils.base_utils as base_utils
from sgw.object import Wall, Reward, Key, Door, Warp, Marker, Other, PushableBox


def default_agent_start(height, width, offset=2):
    """Returns the default agent start position."""
    return [height - offset, width - offset]


def grid_coords(height, width):
    """Generator for all grid coordinates as (i, j)."""
    return ((i, j) for i in range(height) for j in range(width))


def get_empty_objects(reward_pos=[1, 1], reward_value=1.0):
    """Returns a default empty objects dictionary with a single reward."""
    return {
        "rewards": [Reward(reward_pos, reward_value)],
        "markers": [],
        "keys": [],
        "doors": [],
        "warps": [],
        "other": [],
    }


def four_rooms(height: int, width: int):
    """Creates a four rooms layout."""
    agent_start = default_agent_start(height, width)
    mid_w = width // 2
    mid_h = height // 2
    # Adjust earl_mid and late_mid for grid sizes
    earl_mid_w = mid_w // 2 + 1
    late_mid_w = mid_w + earl_mid_w - (1 if width == 11 else 2)
    earl_mid_h = mid_h // 2 + 1
    late_mid_h = mid_h + earl_mid_h - (1 if height == 11 else 2)
    # Vertical and horizontal block lines with removed bottlenecks
    blocks = [[mid_h, i] for i in range(width)] + [[i, mid_w] for i in range(height)]
    bottlenecks = [
        [mid_h, earl_mid_w],
        [mid_h, late_mid_w],
        [earl_mid_h, mid_w],
        [late_mid_h, mid_w],
    ]
    blocks = [b for b in blocks if b not in bottlenecks]
    objects = get_empty_objects()
    return blocks, agent_start, objects


def four_rooms_split(height: int, width: int):
    """Creates a four rooms layout with split design."""
    mid_w = width // 2
    mid_h = height // 2
    earl_mid_w = mid_w // 2
    earl_mid_h = mid_h // 2
    late_mid_w = mid_w + earl_mid_w + (1 if width == 11 else 0)
    late_mid_h = mid_h + earl_mid_h + (1 if height == 11 else 0)
    agent_start = [height - 3, width - 3]
    objects = {
        "rewards": [Reward([earl_mid_h, earl_mid_w], 1.0)],
        "markers": [],
        "keys": [Key([earl_mid_h, late_mid_w])],
        "doors": [Door([earl_mid_h, mid_w], "v")],
        "warps": [Warp([late_mid_h, earl_mid_w], [earl_mid_h + 1, late_mid_w])],
        "other": [],
    }
    # Build blocks for multiple rows
    blocks = [[mid_h + delta, i] for delta in (-1, 0, 1) for i in range(width)]
    # Remove bottlenecks
    bottlenecks = [
        [earl_mid_h, mid_w - 1],
        [earl_mid_h, mid_w],
        [earl_mid_h, mid_w + 1],
        [late_mid_h, mid_w - 1],
        [late_mid_h, mid_w],
        [late_mid_h, mid_w + 1],
    ]
    blocks = [b for b in blocks if b not in bottlenecks]
    return blocks, agent_start, objects


def empty(height: int, width: int):
    """Returns an empty grid layout."""
    agent_start = default_agent_start(height, width)
    blocks = []
    objects = get_empty_objects()
    return blocks, agent_start, objects


def outer_ring(height: int, width: int):
    """Creates a layout with an outer ring of empty cells inside a block area."""
    agent_start = default_agent_start(height, width)
    objects = get_empty_objects()
    extra_depth = 2
    blocks = [
        [i, j]
        for i, j in grid_coords(height, width)
        if not (
            extra_depth < i < height - 1 - extra_depth
            and extra_depth < j < width - 1 - extra_depth
        )
    ]
    return blocks, agent_start, objects


def u_maze(height: int, width: int):
    """Creates a U-shaped maze layout."""
    agent_start = default_agent_start(height, width)
    objects = get_empty_objects(reward_pos=[height - 2, 1])
    extra_depth = 2
    blocks = [
        [i, j]
        for i, j in grid_coords(height, width)
        if i > extra_depth and extra_depth < j < width - 1 - extra_depth
    ]
    return blocks, agent_start, objects


def two_rooms(height: int, width: int):
    """Creates a two rooms layout with door and key."""
    mid = height // 2
    half_mid = mid // 2
    agent_start = default_agent_start(height, width)
    objects = {
        "rewards": [Reward([1, mid], 1.0)],
        "markers": [],
        "keys": [Key([mid + half_mid, mid - half_mid])],
        "doors": [Door([mid, mid], "h")],
        "warps": [],
        "other": [],
        "pushable_boxes": [PushableBox([mid - half_mid, mid + half_mid])],
    }
    blocks = [[mid, i] for i in range(width)]
    # Remove door position
    blocks = [b for b in blocks if b != [mid, mid]]
    if height == 17:
        blocks += [[mid + delta, i] for delta in (-1, 1) for i in range(width)]
        # Remove blocks on door positions
        blocks = [b for b in blocks if b not in ([mid - 1, mid], [mid + 1, mid])]
    return blocks, agent_start, objects


def obstacle(height: int, width: int):
    """Creates an obstacle layout."""
    agent_start = default_agent_start(height, width)
    mid = height // 2
    if height == 11:
        blocks = [[mid, i] for i in range(2, width - 2)]
    else:
        blocks = [[mid, i] for i in range(3, width - 3)]
    objects = get_empty_objects()
    return blocks, agent_start, objects


def s_maze(height: int, width: int):
    """Creates an S-shaped maze layout."""
    agent_start = default_agent_start(height, width)
    mid_a = width // 3
    mid_b = (2 * width // 3) + 1
    blocks_a = [[i, mid_a] for i in range(height // 2 + 1 + height // 4)]
    blocks_b = [[i, mid_b] for i in range(height // 4 + 1, height - 1)]
    blocks = blocks_a + blocks_b
    objects = get_empty_objects()
    return blocks, agent_start, objects


def hairpin(height: int, width: int):
    """Creates a hairpin maze layout."""
    agent_start = default_agent_start(height, width)
    mid_a = width // 5
    mid_b = 2 * (width // 5)
    mid_c = 3 * (width // 5)
    mid_d = 4 * (width // 5)
    blocks_a = [[i, mid_a] for i in range(height // 2 + 1 + height // 4)]
    blocks_b = [[i, mid_b] for i in range(height // 4 + 1, height - 1)]
    blocks_c = [[i, mid_c] for i in range(height // 2 + 1 + height // 4)]
    blocks_d = [[i, mid_d] for i in range(height // 4 + 1, height - 1)]
    blocks = blocks_a + blocks_b + blocks_c + blocks_d
    objects = get_empty_objects()
    return blocks, agent_start, objects


def circle(height: int, width: int):
    """Creates a circular layout by blocking outside the circle."""
    agent_start = [height - 2, width // 2]
    objects = get_empty_objects(reward_pos=[1, width // 2])
    mask = base_utils.create_circular_mask(height, width)
    blocks = [[i, j] for i, j in grid_coords(height, width) if mask[i, j] == 0]
    return blocks, agent_start, objects


def ring(height: int, width: int):
    """Creates a ring layout based on two circular masks."""
    agent_start = [height - 2, width // 2]
    objects = get_empty_objects(reward_pos=[1, width // 2])
    big_mask = base_utils.create_circular_mask(height, width)
    small_mask = base_utils.create_circular_mask(
        height, width, radius=min(height, width) // 4
    )
    blocks = [
        [i, j]
        for i, j in grid_coords(height, width)
        if big_mask[i, j] == 0 or small_mask[i, j] != 0
    ]
    return blocks, agent_start, objects


def t_maze(height: int, width: int):
    """Creates a T-shaped maze layout."""
    agent_start = [height - 2, width // 2]
    objects = get_empty_objects()
    corridor_width = 3
    half_width = corridor_width // 2
    middle = width // 2
    blocks = [
        [i, j]
        for i, j in grid_coords(height, width)
        if i >= corridor_width + 1
        and (j < middle - half_width or j > middle + half_width)
    ]
    return blocks, agent_start, objects


def i_maze(height: int, width: int):
    """Creates an I-shaped maze layout."""
    agent_start = default_agent_start(height, width)
    objects = get_empty_objects()
    corridor_width = 3
    half_width = corridor_width // 2
    middle = width // 2
    blocks = [
        [i, j]
        for i, j in grid_coords(height, width)
        if corridor_width + 1 <= i <= height - corridor_width - 2
        and (j < middle - half_width or j > middle + half_width)
    ]
    return blocks, agent_start, objects


def hallways(height: int, width: int):
    """Creates a hallways layout by carving out inner lines."""
    agent_start = default_agent_start(height, width)
    objects = get_empty_objects()
    extra = 1
    blocks = [
        [i, j]
        for i, j in grid_coords(height, width)
        if (extra < i < height - extra - 1 and extra < j < width - extra - 1)
        and not (i == height // 2 or j == width // 2)
    ]
    return blocks, agent_start, objects


def detour(height: int, width: int):
    """Creates a detour layout by removing the center column."""
    agent_start = [height - 2, width // 2]
    objects = get_empty_objects(reward_pos=[1, width // 2])
    extra = 1
    blocks = [
        [i, j]
        for i, j in grid_coords(height, width)
        if extra < i < height - 1 - extra and extra < j < width - 1 - extra
    ]
    # Remove entire center column
    blocks = [b for b in blocks if b[1] != width // 2]
    return blocks, agent_start, objects


def detour_block(height: int, width: int):
    """Creates a detour layout that preserves the center row block."""
    agent_start = [height - 2, width // 2]
    objects = get_empty_objects(reward_pos=[1, width // 2])
    extra = 1
    blocks = [
        [i, j]
        for i, j in grid_coords(height, width)
        if extra < i < height - 1 - extra and extra < j < width - 1 - extra
    ]
    # Remove center column except the middle cell
    blocks = [b for b in blocks if not (b[1] == width // 2 and b[0] != height // 2)]
    return blocks, agent_start, objects


def two_step(height: int, width: int):
    """Creates a two-step layout with multiple rewards and obstacles."""
    agent_start = [height - 2, width // 2]
    objects = {
        "rewards": [
            Reward([1, 1], 0.5),
            Reward([1, 3], -1.0),
            Reward([1, 9], 0.25),
            Reward([1, 7], 0.25),
        ],
        "markers": [],
        "keys": [],
        "doors": [],
        "warps": [],
        "other": [],
    }
    blocks = []
    # Multiple rows of blocks:
    for col in (2, 4, 6, 8):
        blocks.extend([[i, col] for i in range(1, height - 1)])
    for col in (1, 7, 3, 9):
        blocks.extend([[i, col] for i in range(4, height - 1)])
    blocks.extend([[i, 5] for i in range(1, 6)])
    if width > 11:
        for col in range(10, 16):
            blocks.extend([[i, col] for i in range(1, height - 1)])
        agent_start[1] -= 3
    if width > 7:
        for b in (
            [4, 2],
            [4, 8],
            [6, 4],
            [6, 6],
            [6, 3],
            [6, 7],
            [6, 2],
            [6, 8],
            [5, 2],
            [5, 8],
            [3, 2],
            [3, 8],
        ):
            if b in blocks:
                blocks.remove(b)
    return blocks, agent_start, objects


def narrow(height: int, width: int):
    """Creates a narrow layout with selective rewards and obstacles."""
    agent_start = [height - 2, width // 2]
    objects = {
        "rewards": [Reward([1, 5], 1.0), Reward([5, 5], -1.0)],
        "markers": [],
        "keys": [],
        "doors": [],
        "warps": [],
        "other": [],
    }
    # Use a few fixed rows as obstacles
    blocks = []
    for col in (1, 2, 8, 9):
        blocks.extend([[i, col] for i in range(1, height - 1)])
    if width > 11:
        for col in range(10, 16):
            blocks.extend([[i, col] for i in range(1, height - 1)])
        agent_start[1] -= 3
    return blocks, agent_start, objects


TEMPLATES = {
    "empty": empty,
    "four_rooms": four_rooms,
    "outer_ring": outer_ring,
    "two_rooms": two_rooms,
    "u_maze": u_maze,
    "t_maze": t_maze,
    "hallways": hallways,
    "ring": ring,
    "s_maze": s_maze,
    "circle": circle,
    "i_maze": i_maze,
    "hairpin": hairpin,
    "detour": detour,
    "detour_block": detour_block,
    "four_rooms_split": four_rooms_split,
    "obstacle": obstacle,
    "two_step": two_step,
    "narrow": narrow,
}


def generate_layout(
    template: str = "empty",
    height: int = 11,
    width: int = 11,
    add_outer_walls: bool = True,
):
    """Generates the maze layout based on template and grid dimensions."""
    if template not in TEMPLATES:
        raise ValueError(
            f"Unknown template: {template}. Valid templates are: {list(TEMPLATES.keys())}"
        )
    blocks, agent_start, objects = TEMPLATES[template](height, width)
    if add_outer_walls:
        blocks = add_outer(blocks, height, width)
    objects["walls"] = [Wall(pos) for pos in blocks]
    return agent_start, objects


def add_outer(blocks: list, height: int, width: int):
    """Adds an outer border to the blocks."""
    outer_blocks = [
        [i, j]
        for i, j in grid_coords(height, width)
        if i == 0 or i == height - 1 or j == 0 or j == width - 1
    ]
    blocks.extend(outer_blocks)
    return blocks
