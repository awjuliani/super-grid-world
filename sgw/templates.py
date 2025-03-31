import sgw.utils.base_utils as base_utils
from sgw.object import (
    Wall,
    Reward,
    Key,
    Door,
    Warp,
    Marker,
    Other,
    PushableBox,
    LinkedDoor,
    PressurePlate,
    Lever,
)


DEFAULT_AGENT_START_OFFSET = 2
DEFAULT_REWARD_VALUE = 1.0
DEFAULT_REWARD_POS = [1, 1]


def default_agent_start(height, width, offset=DEFAULT_AGENT_START_OFFSET):
    """Returns the default agent start position."""
    return [height - offset, width - offset]


def grid_coords(height, width):
    """Generator for all grid coordinates as (i, j)."""
    return ((i, j) for i in range(height) for j in range(width))


def get_empty_objects(
    reward_pos=None, reward_value=DEFAULT_REWARD_VALUE
) -> dict[str, list]:
    """
    Returns a default objects dictionary initialized with standard keys
    and optionally a single reward.
    """
    if reward_pos is None:
        reward_pos = DEFAULT_REWARD_POS
    return {
        "walls": [],  # Walls will be added later in generate_layout
        "rewards": [Reward(reward_pos, reward_value)] if reward_pos else [],
        "markers": [],
        "keys": [],
        "doors": [],
        "warps": [],
        "other": [],
        "pushable_boxes": [],
        "linked_doors": [],
        "pressure_plates": [],
        "levers": [],
    }


def four_rooms(height: int, width: int):
    """Creates a four rooms layout."""
    agent_start = default_agent_start(height, width)
    mid_w = width // 2
    mid_h = height // 2

    # Calculate passage coordinates, adjusting for grid size parity
    # These ensure passages are roughly centered within each wall segment
    passage_offset_w = mid_w // 2 + 1
    passage_offset_h = mid_h // 2 + 1
    passage_w1 = passage_offset_w
    passage_w2 = (
        mid_w + passage_offset_w - (1 if width == 11 else 2)
    )  # Adjust based on common size
    passage_h1 = passage_offset_h
    passage_h2 = (
        mid_h + passage_offset_h - (1 if height == 11 else 2)
    )  # Adjust based on common size

    # Define wall lines
    vert_wall = [[mid_h, i] for i in range(width)]
    horz_wall = [[i, mid_w] for i in range(height)]
    blocks = vert_wall + horz_wall

    # Define passage coordinates (bottlenecks)
    bottlenecks = [
        [mid_h, passage_w1],
        [mid_h, passage_w2],
        [passage_h1, mid_w],
        [passage_h2, mid_w],
    ]
    # Remove passages from walls
    blocks = [b for b in blocks if b not in bottlenecks]

    objects = get_empty_objects()  # Start with default objects
    return blocks, agent_start, objects


def four_rooms_split(height: int, width: int):
    """Creates a four rooms layout with split design."""
    mid_w = width // 2
    mid_h = height // 2

    # Calculate coordinates relative to the center
    earl_mid_w = mid_w // 2
    earl_mid_h = mid_h // 2
    # Adjust late midpoints based on common size parity
    late_mid_w = mid_w + earl_mid_w + (1 if width == 11 else 0)
    late_mid_h = mid_h + earl_mid_h + (1 if height == 11 else 0)

    agent_start = [height - 3, width - 3]  # Specific start for this layout

    # Define specific objects for this layout
    objects = get_empty_objects(reward_pos=[earl_mid_h, earl_mid_w], reward_value=1.0)
    objects["keys"] = [Key([earl_mid_h, late_mid_w])]
    objects["doors"] = [Door([earl_mid_h, mid_w], "v")]
    objects["warps"] = [Warp([late_mid_h, earl_mid_w], [earl_mid_h + 1, late_mid_w])]

    # Build walls (three rows thick horizontal wall)
    blocks = [[mid_h + delta, i] for delta in (-1, 0, 1) for i in range(width)]

    # Define passage coordinates (bottlenecks)
    bottlenecks = [
        [earl_mid_h, mid_w - 1],
        [earl_mid_h, mid_w],
        [earl_mid_h, mid_w + 1],
        [late_mid_h, mid_w - 1],
        [late_mid_h, mid_w],
        [late_mid_h, mid_w + 1],
    ]
    # Remove passages from walls
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

    # Define specific objects for this layout
    objects = get_empty_objects(reward_pos=[1, mid], reward_value=1.0)
    objects["keys"] = [Key([mid + half_mid, mid - half_mid])]
    objects["doors"] = [Door([mid, mid], "h")]
    objects["pushable_boxes"] = [PushableBox([mid - half_mid, mid + half_mid])]

    # Define wall line
    blocks = [[mid, i] for i in range(width)]
    # Remove door position
    blocks = [b for b in blocks if b != [mid, mid]]

    # Make wall thicker for larger grids (e.g., 17x17)
    if height == 17:
        blocks += [[mid + delta, i] for delta in (-1, 1) for i in range(width)]
        # Remove blocks adjacent to door position
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

    # Define specific rewards for this layout
    objects = get_empty_objects(reward_pos=None)  # No single default reward needed
    objects["rewards"] = [
        Reward([1, 1], 0.5),
        Reward([1, 3], -1.0),
        Reward([1, 9], 0.25),
        Reward([1, 7], 0.25),
    ]

    blocks = []
    # Define vertical wall segments
    for col in (2, 4, 6, 8):
        blocks.extend([[i, col] for i in range(1, height - 1)])
    # Define partial vertical wall segments
    for col in (1, 7, 3, 9):
        blocks.extend([[i, col] for i in range(4, height - 1)])
    # Define central wall segment
    blocks.extend([[i, 5] for i in range(1, 6)])

    # Add more walls for wider grids
    if width > 11:
        for col in range(10, 16):
            blocks.extend([[i, col] for i in range(1, height - 1)])
        agent_start[1] -= 3  # Adjust agent start for wider grid

    # Carve out specific passages if grid width allows (width > 7)
    if width > 7:
        passage_coords = [
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
        ]
        # Use a set for efficient lookup during removal
        passage_set = {tuple(p) for p in passage_coords}
        blocks = [b for b in blocks if tuple(b) not in passage_set]

    return blocks, agent_start, objects


def narrow(height: int, width: int):
    """Creates a narrow layout with selective rewards and obstacles."""
    agent_start = [height - 2, width // 2]

    # Define specific rewards for this layout
    objects = get_empty_objects(reward_pos=[1, 5], reward_value=1.0)
    objects["rewards"].append(Reward([5, 5], -1.0))  # Add a second reward

    # Use a few fixed columns as obstacles
    blocks = []
    for col in (1, 2, 8, 9):
        blocks.extend([[i, col] for i in range(1, height - 1)])

    # Add more walls for wider grids
    if width > 11:
        for col in range(10, 16):
            blocks.extend([[i, col] for i in range(1, height - 1)])
        agent_start[1] -= 3  # Adjust agent start for wider grid

    return blocks, agent_start, objects


def linked_door_test(height: int, width: int):
    """Creates a layout to test PressurePlate, Lever, and LinkedDoor."""
    # Position agent bottom-center
    agent_start = [height - 2, width // 2]

    # Define positions
    door_pos = [height // 2, width // 2]
    plate_pos = [height - 3, width // 2 - 2]  # Left of start
    reward_pos = [1, width // 2]  # Behind the door
    box_pos = [height - 3, width // 2 - 1]

    # Define a common ID for linking
    door_id = "test_door_1"

    # Create the objects dictionary
    objects = get_empty_objects(reward_pos=None)  # Start fresh, no default reward
    objects["rewards"] = [Reward(reward_pos, DEFAULT_REWARD_VALUE)]
    objects["linked_doors"] = [
        LinkedDoor(pos=door_pos, linked_id=door_id, orientation="h")
    ]
    objects["pressure_plates"] = [
        PressurePlate(pos=plate_pos, target_linked_id=door_id)
    ]
    objects["pushable_boxes"] = [PushableBox(box_pos)]
    # Create a simple wall dividing top and bottom, with the door as passage
    blocks = [[door_pos[0], j] for j in range(width) if j != door_pos[1]]

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
    "linked_door_test": linked_door_test,
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

    # Get layout components from the specified template function
    inner_blocks, agent_start, objects = TEMPLATES[template](height, width)

    # Combine inner blocks with outer walls if requested
    all_blocks = (
        add_outer(inner_blocks, height, width) if add_outer_walls else inner_blocks
    )

    # Instantiate Wall objects from block coordinates
    # Ensure 'walls' key exists, even if add_outer_walls is False
    if "walls" not in objects:
        objects["walls"] = []
    objects["walls"].extend([Wall(pos) for pos in all_blocks])

    return agent_start, objects


def add_outer(inner_blocks: list, height: int, width: int) -> list:
    """Adds an outer border to the blocks and returns a new list."""
    outer_blocks = [
        [i, j]
        for i, j in grid_coords(height, width)
        if i == 0 or i == height - 1 or j == 0 or j == width - 1
    ]
    # Return a new list containing both inner and outer blocks
    # Use tuples to allow converting to a set for efficient duplicate removal
    block_set = {tuple(b) for b in inner_blocks} | {tuple(b) for b in outer_blocks}
    return [list(b) for b in block_set]
