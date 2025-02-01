import sgw.utils.base_utils as base_utils
import enum


class GridSize(enum.Enum):
    five = 5
    micro = 7
    small = 11
    large = 17


def default_agent_start(grid_size, offset=2):
    """Returns the default agent start position."""
    return [grid_size - offset, grid_size - offset]


def grid_coords(grid_size):
    """Generator for all grid coordinates as (i, j)."""
    return ((i, j) for i in range(grid_size) for j in range(grid_size))


def four_rooms(grid_size: int):
    """Creates a four rooms layout."""
    agent_start = default_agent_start(grid_size)
    mid = grid_size // 2
    # Adjust earl_mid and late_mid for grid sizes
    earl_mid = mid // 2 + 1
    late_mid = mid + earl_mid - (1 if grid_size == 11 else 2)
    # Vertical and horizontal block lines with removed bottlenecks
    blocks = [[mid, i] for i in range(grid_size)] + [[i, mid] for i in range(grid_size)]
    bottlenecks = [[mid, earl_mid], [mid, late_mid], [earl_mid, mid], [late_mid, mid]]
    blocks = [b for b in blocks if b not in bottlenecks]
    objects = {"rewards": {(1, 1): 1.0}, "markers": {}}
    return blocks, agent_start, objects


def four_rooms_split(grid_size: int):
    """Creates a four rooms layout with split design."""
    mid = grid_size // 2
    earl_mid = mid // 2
    late_mid = mid + earl_mid + (1 if grid_size == 11 else 0)
    agent_start = [grid_size - 3, grid_size - 3]
    objects = {
        "rewards": {(earl_mid, earl_mid): 1.0},
        "markers": {},
        "keys": [(earl_mid, late_mid)],
        "doors": {(earl_mid, mid): "v"},
        "warps": {(late_mid, earl_mid): (earl_mid + 1, late_mid)},
    }
    # Build blocks for multiple rows
    blocks = [[mid + delta, i] for delta in (-1, 0, 1) for i in range(grid_size)]
    # Remove bottlenecks
    bottlenecks = [
        [earl_mid, mid - 1],
        [earl_mid, mid],
        [earl_mid, mid + 1],
        [late_mid, mid - 1],
        [late_mid, mid],
        [late_mid, mid + 1],
    ]
    blocks = [b for b in blocks if b not in bottlenecks]
    return blocks, agent_start, objects


def empty(grid_size: int):
    """Returns an empty grid layout."""
    agent_start = default_agent_start(grid_size)
    blocks = []
    objects = {"rewards": {(1, 1): 1.0}, "markers": {}}
    return blocks, agent_start, objects


def outer_ring(grid_size: int):
    """Creates a layout with an outer ring of empty cells inside a block area."""
    agent_start = default_agent_start(grid_size)
    objects = {"rewards": {(1, 1): 1.0}, "markers": {}}
    extra_depth = 2
    blocks = [
        [i, j]
        for i, j in grid_coords(grid_size)
        if not (
            extra_depth < i < grid_size - 1 - extra_depth
            and extra_depth < j < grid_size - 1 - extra_depth
        )
    ]
    return blocks, agent_start, objects


def u_maze(grid_size: int):
    """Creates a U-shaped maze layout."""
    agent_start = default_agent_start(grid_size)
    objects = {"rewards": {(grid_size - 2, 1): 1.0}, "markers": {}}
    extra_depth = 2
    blocks = [
        [i, j]
        for i, j in grid_coords(grid_size)
        if i > extra_depth and extra_depth < j < grid_size - 1 - extra_depth
    ]
    return blocks, agent_start, objects


def two_rooms(grid_size: int):
    """Creates a two rooms layout with door and key."""
    mid = grid_size // 2
    half_mid = mid // 2
    agent_start = default_agent_start(grid_size)
    objects = {
        "rewards": {(1, mid): 1.0},
        "markers": {},
        "keys": [(mid + half_mid, mid - half_mid)],
        "doors": {(mid, mid): "h"},
    }
    blocks = [[mid, i] for i in range(grid_size)]
    # Remove door position
    blocks = [b for b in blocks if b != [mid, mid]]
    if grid_size == 17:
        blocks += [[mid + delta, i] for delta in (-1, 1) for i in range(grid_size)]
        # Remove blocks on door positions
        blocks = [b for b in blocks if b not in ([mid - 1, mid], [mid + 1, mid])]
    return blocks, agent_start, objects


def obstacle(grid_size: int):
    """Creates an obstacle layout."""
    agent_start = default_agent_start(grid_size)
    mid = grid_size // 2
    if grid_size == 11:
        blocks = [[mid, i] for i in range(2, grid_size - 2)]
    else:
        blocks = [[mid, i] for i in range(3, grid_size - 3)]
    objects = {"rewards": {(1, 1): 1.0}, "markers": {}}
    return blocks, agent_start, objects


def s_maze(grid_size: int):
    """Creates an S-shaped maze layout."""
    agent_start = default_agent_start(grid_size)
    mid_a = grid_size // 3
    mid_b = (2 * grid_size // 3) + 1
    blocks_a = [[i, mid_a] for i in range(grid_size // 2 + 1 + grid_size // 4)]
    blocks_b = [[i, mid_b] for i in range(grid_size // 4 + 1, grid_size - 1)]
    blocks = blocks_a + blocks_b
    objects = {"rewards": {(1, 1): 1.0}, "markers": {}}
    return blocks, agent_start, objects


def hairpin(grid_size: int):
    """Creates a hairpin maze layout."""
    agent_start = default_agent_start(grid_size)
    mid_a = grid_size // 5
    mid_b = 2 * (grid_size // 5)
    mid_c = 3 * (grid_size // 5)
    mid_d = 4 * (grid_size // 5)
    blocks_a = [[i, mid_a] for i in range(grid_size // 2 + 1 + grid_size // 4)]
    blocks_b = [[i, mid_b] for i in range(grid_size // 4 + 1, grid_size - 1)]
    blocks_c = [[i, mid_c] for i in range(grid_size // 2 + 1 + grid_size // 4)]
    blocks_d = [[i, mid_d] for i in range(grid_size // 4 + 1, grid_size - 1)]
    blocks = blocks_a + blocks_b + blocks_c + blocks_d
    objects = {"rewards": {(1, 1): 1.0}, "markers": {}}
    return blocks, agent_start, objects


def circle(grid_size: int):
    """Creates a circular layout by blocking outside the circle."""
    agent_start = [grid_size - 2, grid_size // 2]
    objects = {"rewards": {(1, grid_size // 2): 1.0}, "markers": {}}
    mask = base_utils.create_circular_mask(grid_size, grid_size)
    blocks = [[i, j] for i, j in grid_coords(grid_size) if mask[i, j] == 0]
    return blocks, agent_start, objects


def ring(grid_size: int):
    """Creates a ring layout based on two circular masks."""
    agent_start = [grid_size - 2, grid_size // 2]
    objects = {"rewards": {(1, grid_size // 2): 1.0}, "markers": {}}
    big_mask = base_utils.create_circular_mask(grid_size, grid_size)
    small_mask = base_utils.create_circular_mask(
        grid_size, grid_size, radius=grid_size // 4
    )
    blocks = [
        [i, j]
        for i, j in grid_coords(grid_size)
        if big_mask[i, j] == 0 or small_mask[i, j] != 0
    ]
    return blocks, agent_start, objects


def t_maze(grid_size: int):
    """Creates a T-shaped maze layout."""
    agent_start = [grid_size - 2, grid_size // 2]
    objects = {"rewards": {(1, 1): 1.0}, "markers": {}}
    width = 3
    half_width = width // 2
    middle = grid_size // 2
    blocks = [
        [i, j]
        for i, j in grid_coords(grid_size)
        if i >= width + 1 and (j < middle - half_width or j > middle + half_width)
    ]
    return blocks, agent_start, objects


def i_maze(grid_size: int):
    """Creates an I-shaped maze layout."""
    agent_start = default_agent_start(grid_size)
    objects = {"rewards": {(1, 1): 1.0}, "markers": {}}
    width = 3
    half_width = width // 2
    middle = grid_size // 2
    blocks = [
        [i, j]
        for i, j in grid_coords(grid_size)
        if width + 1 <= i <= grid_size - width - 2
        and (j < middle - half_width or j > middle + half_width)
    ]
    return blocks, agent_start, objects


def hallways(grid_size: int):
    """Creates a hallways layout by carving out inner lines."""
    agent_start = default_agent_start(grid_size)
    objects = {"rewards": {(1, 1): 1.0}, "markers": {}}
    extra = 1
    blocks = [
        [i, j]
        for i, j in grid_coords(grid_size)
        if (extra < i < grid_size - extra - 1 and extra < j < grid_size - extra - 1)
        and not (i == grid_size // 2 or j == grid_size // 2)
    ]
    return blocks, agent_start, objects


def detour(grid_size: int):
    """Creates a detour layout by removing the center column."""
    agent_start = [grid_size - 2, grid_size // 2]
    objects = {"rewards": {(1, grid_size // 2): 1.0}, "markers": {}}
    extra = 1
    blocks = [
        [i, j]
        for i, j in grid_coords(grid_size)
        if extra < i < grid_size - 1 - extra and extra < j < grid_size - 1 - extra
    ]
    # Remove entire center column
    blocks = [b for b in blocks if b[1] != grid_size // 2]
    return blocks, agent_start, objects


def detour_block(grid_size: int):
    """Creates a detour layout that preserves the center row block."""
    agent_start = [grid_size - 2, grid_size // 2]
    objects = {"rewards": {(1, grid_size // 2): 1.0}, "markers": {}}
    extra = 1
    blocks = [
        [i, j]
        for i, j in grid_coords(grid_size)
        if extra < i < grid_size - 1 - extra and extra < j < grid_size - 1 - extra
    ]
    # Remove center column except the middle cell
    blocks = [
        b for b in blocks if not (b[1] == grid_size // 2 and b[0] != grid_size // 2)
    ]
    return blocks, agent_start, objects


def two_step(grid_size):
    """Creates a two-step layout with multiple rewards and obstacles."""
    agent_start = [grid_size - 2, grid_size // 2]
    objects = {
        "rewards": {(1, 1): 0.5, (1, 3): -1.0, (1, 9): 0.25, (1, 7): 0.25},
        "markers": {},
    }
    blocks = []
    # Multiple rows of blocks:
    for col in (2, 4, 6, 8):
        blocks.extend([[i, col] for i in range(1, grid_size - 1)])
    for col in (1, 7, 3, 9):
        blocks.extend([[i, col] for i in range(4, grid_size - 1)])
    blocks.extend([[i, 5] for i in range(1, 6)])
    if grid_size > 11:
        for col in range(10, 16):
            blocks.extend([[i, col] for i in range(1, grid_size - 1)])
        agent_start[1] -= 3
    if grid_size > 7:
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


def narrow(grid_size):
    """Creates a narrow layout with selective rewards and obstacles."""
    agent_start = [grid_size - 2, grid_size // 2]
    objects = {"rewards": {(1, 5): 1.0, (5, 5): -1.0}, "markers": {}}
    # Use a few fixed rows as obstacles
    blocks = []
    for col in (1, 2, 8, 9):
        blocks.extend([[i, col] for i in range(1, grid_size - 1)])
    if grid_size > 11:
        for col in range(10, 16):
            blocks.extend([[i, col] for i in range(1, grid_size - 1)])
        agent_start[1] -= 3
    return blocks, agent_start, objects


class LayoutTemplate(enum.Enum):
    empty = "empty"
    four_rooms = "four_rooms"
    outer_ring = "outer_ring"
    two_rooms = "two_rooms"
    u_maze = "u_maze"
    t_maze = "t_maze"
    hallways = "hallways"
    ring = "ring"
    s_maze = "s_maze"
    circle = "circle"
    i_maze = "i_maze"
    hairpin = "hairpin"
    detour = "detour"
    detour_block = "detour_block"
    four_rooms_split = "four_rooms_split"
    obstacle = "obstacle"
    two_step = "two_step"
    narrow = "narrow"


template_map = {
    LayoutTemplate.empty: empty,
    LayoutTemplate.four_rooms: four_rooms,
    LayoutTemplate.outer_ring: outer_ring,
    LayoutTemplate.two_rooms: two_rooms,
    LayoutTemplate.u_maze: u_maze,
    LayoutTemplate.t_maze: t_maze,
    LayoutTemplate.hallways: hallways,
    LayoutTemplate.ring: ring,
    LayoutTemplate.s_maze: s_maze,
    LayoutTemplate.circle: circle,
    LayoutTemplate.i_maze: i_maze,
    LayoutTemplate.hairpin: hairpin,
    LayoutTemplate.detour: detour,
    LayoutTemplate.detour_block: detour_block,
    LayoutTemplate.four_rooms_split: four_rooms_split,
    LayoutTemplate.obstacle: obstacle,
    LayoutTemplate.two_step: two_step,
    LayoutTemplate.narrow: narrow,
}


def generate_layout(
    template: LayoutTemplate = LayoutTemplate.empty,
    grid_size: GridSize = GridSize.small,
):
    """Generates the maze layout based on template and grid size."""
    grid_val = grid_size.value
    if isinstance(template, str):
        template = LayoutTemplate(template)
    blocks, agent_start, objects = template_map[template](grid_val)
    return blocks, agent_start, objects


def add_outer(blocks: list, grid_size: int):
    """Adds an outer border to the blocks."""
    outer_blocks = [
        [i, j]
        for i, j in grid_coords(grid_size)
        if i == 0 or i == grid_size - 1 or j == 0 or j == grid_size - 1
    ]
    blocks.extend(outer_blocks)
    return blocks
