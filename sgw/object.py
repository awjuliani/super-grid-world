from typing import List, Tuple, Optional, Any
import copy


class Object:
    """Base class for all objects in the environment."""

    def __init__(
        self,
        pos: List[int],
        obstacle: bool = True,
        consumable: bool = False,
        terminal: bool = False,
    ):
        """
        Initialize an object.

        Args:
            pos: [x, y] position of the object
            obstacle: Whether the object blocks movement
            consumable: Whether the object can be consumed/collected
            terminal: Whether interacting with this object ends the episode
        """
        self.pos = pos
        self.obstacle = obstacle
        self.consumable = consumable
        self.terminal = terminal
        self.name = "object"

    def __eq__(self, other: Any) -> bool:
        """Check if this object is at the same position as another object or coordinate."""
        if isinstance(other, (list, tuple)):
            return self.pos[0] == other[0] and self.pos[1] == other[1]
        elif isinstance(other, Object):
            return self.pos[0] == other.pos[0] and self.pos[1] == other.pos[1]
        return False

    def copy(self) -> "Object":
        """Create a deep copy of this object."""
        return type(self)(
            list(self.pos),
            self.obstacle,
            self.consumable,
            self.terminal,
        )

    def interact(self, agent: Any, env: Any = None) -> Optional[str]:
        """
        Called when agent interacts with object.

        Args:
            agent: The agent interacting with this object
            env: The environment instance (optional)

        Returns:
            Optional event message string
        """
        if self.terminal:
            agent.done = True
        return None

    def step(self, env: Any) -> None:
        """
        Called each step of the environment.

        Args:
            env: The environment instance
        """
        pass


class Wall(Object):
    """A wall that blocks movement."""

    def __init__(self, pos: List[int]):
        super().__init__(pos, obstacle=True, consumable=False)
        self.name = "obstacle"

    def copy(self) -> "Wall":
        return type(self)(list(self.pos))


class Reward(Object):
    """A reward object that gives the agent a specified value when collected."""

    def __init__(self, pos: List[int], value: float):
        super().__init__(pos, obstacle=False, consumable=True)
        self.value = value
        self.name = "reward"

    def copy(self) -> "Reward":
        return type(self)(list(self.pos), self.value)

    def interact(self, agent: Any, env: Any = None) -> str:
        super().interact(agent, env)
        agent.collect_reward(self.value)
        return f"{agent.name} collected {self.name} of {self.value}"


class Key(Object):
    """A key that can be collected and used to unlock doors."""

    def __init__(self, pos: List[int]):
        super().__init__(pos, obstacle=False, consumable=True)
        self.name = "door key"

    def copy(self) -> "Key":
        return type(self)(list(self.pos))

    def interact(self, agent: Any, env: Any = None) -> str:
        super().interact(agent, env)
        agent.collect_object(self)
        return f"{agent.name} collected a {self.name}"


class Door(Object):
    """A door that can be unlocked with a key."""

    def __init__(self, pos: List[int]):
        super().__init__(pos, obstacle=True, consumable=True)
        self.name = "locked door"

    def copy(self) -> "Door":
        return type(self)(list(self.pos))

    def try_unlock(self, agent: Any) -> Tuple[bool, Optional[str]]:
        """
        Attempt to unlock the door with a key from the agent's inventory.

        Args:
            agent: The agent trying to unlock the door

        Returns:
            Tuple of (success, message)
        """
        if not self.obstacle:  # Already unlocked
            return True, None
        if agent.use_key():
            self.obstacle = False  # Unlock the door
            return True, f"{agent.name} unlocked and went through a {self.name}"
        return False, f"{agent.name} tried to open a {self.name} but had no key"

    def interact(self, agent: Any, env: Any = None) -> Optional[str]:
        super().interact(agent, env)
        success, message = self.try_unlock(agent)
        return message

    def pre_step_interaction(
        self, agent: Any, direction: Any, env: Any = None
    ) -> Tuple[bool, Optional[str]]:
        return self.try_unlock(agent)


class Warp(Object):
    """A teleportation pad that moves the agent to a target location."""

    def __init__(self, pos: List[int], target: List[int]):
        super().__init__(pos, obstacle=False, consumable=False)
        self.target = target
        self.name = "warp pad"

    def copy(self) -> "Warp":
        return type(self)(list(self.pos), list(self.target))

    def interact(self, agent: Any, env: Any = None) -> str:
        super().interact(agent, env)
        agent.teleport(self.target)
        return f"{agent.name} used a {self.name} to teleport"


class Marker(Object):
    """A colored marker on the ground."""

    def __init__(self, pos: List[int], color: Tuple[int, int, int]):
        super().__init__(pos, obstacle=False, consumable=False)
        self.color = color
        self.name = "marker"

    def copy(self) -> "Marker":
        return type(self)(list(self.pos), tuple(self.color))


class Other(Object):
    """A generic collectible object with a custom name."""

    def __init__(self, pos: List[int], name: str):
        super().__init__(pos, obstacle=False, consumable=True, terminal=False)
        self.name = name

    def copy(self) -> "Other":
        return type(self)(list(self.pos), self.name)

    def interact(self, agent: Any, env: Any = None) -> str:
        super().interact(agent, env)
        agent.collect_object(self)
        return f"{agent.name} collected {self.name}"


class Tree(Object):
    """A tree that can spawn fruits in nearby locations."""

    def __init__(self, pos: List[int], spawn_rate: float = 0.5, spawn_radius: int = 2):
        super().__init__(pos, obstacle=True, consumable=False, terminal=False)
        self.name = "tree"
        self.spawn_rate = spawn_rate
        self.spawn_radius = spawn_radius

    def copy(self) -> "Tree":
        return type(self)(list(self.pos), self.spawn_rate, self.spawn_radius)

    def step(self, env: Any) -> Optional[str]:
        """
        Potentially spawn a fruit in a nearby location.

        Args:
            env: The environment instance

        Returns:
            Optional event message string
        """
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
                return "A fruit fell from a tree"
        return None


class Fruit(Object):
    """A fruit that can be collected by the agent."""

    def __init__(self, pos: List[int]):
        super().__init__(pos, obstacle=False, consumable=True, terminal=False)
        self.name = "fruit"

    def copy(self) -> "Fruit":
        return type(self)(list(self.pos))

    def interact(self, agent: Any, env: Any = None) -> str:
        super().interact(agent, env)
        agent.collect_object(self)
        return f"{agent.name} collected a {self.name}"


class Box(Object):
    """A box that can store and provide items."""

    def __init__(self, pos: List[int]):
        super().__init__(pos, obstacle=False, consumable=False, terminal=False)
        self.name = "box"
        self.contents = []

    def copy(self) -> "Box":
        new_box = type(self)(list(self.pos))
        new_box.contents = self.contents.copy()
        return new_box

    def interact(self, agent: Any, env: Any = None) -> str:
        super().interact(agent, env)
        if agent.inventory:
            # Put item in box
            item = agent.inventory.pop(0)  # Remove and get first item from inventory
            self.contents.append(item)
            return f"{agent.name} put {item.name} in the box"
        elif self.contents:
            # Take item from box
            item = self.contents.pop()  # Remove and get last item from box
            agent.collect_object(item)
            return f"{agent.name} took {item.name} from the box"
        return f"{agent.name} found a {self.name} but it was empty"


class Sign(Object):
    """A sign with a readable message."""

    def __init__(self, pos: List[int], message: str):
        super().__init__(pos, obstacle=False, consumable=False, terminal=False)
        self.message = message
        self.name = "sign"

    def copy(self) -> "Sign":
        return type(self)(list(self.pos), self.message)

    def interact(self, agent: Any, env: Any = None) -> str:
        super().interact(agent, env)
        return f"{agent.name} read the sign: '{self.message}'"


class PushableBox(Object):
    """A box that can be pushed by the agent."""

    def __init__(self, pos: List[int]):
        super().__init__(pos, obstacle=True, consumable=False, terminal=False)
        self.name = "pushable box"
        self.being_pushed = False

    def copy(self) -> "PushableBox":
        new_box = type(self)(list(self.pos))
        new_box.being_pushed = self.being_pushed
        return new_box

    def pre_step_interaction(
        self, agent: Any, direction: Any, env: Any = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Called before the agent moves onto the box's position.
        Calculates the new position for the box based on the direction the agent is looking.

        Args:
            agent: The agent trying to push the box
            direction: The direction the agent is moving
            env: The environment instance, needed to check if the new position is valid

        Returns:
            Tuple of (allowed, message)
        """
        # Get the direction the agent is looking
        looking_idx = agent.looking

        # Get the direction vector from the agent's direction map
        push_direction = agent.direction_map[looking_idx].tolist()

        # Calculate the new position for the box
        new_box_pos = [self.pos[0] + push_direction[0], self.pos[1] + push_direction[1]]

        # Check if the new position is valid (if env is provided)
        if env:
            # Check grid bounds
            if not (
                -1 < new_box_pos[0] < env.grid_shape[0]
                and -1 < new_box_pos[1] < env.grid_shape[1]
            ):
                return (
                    False,
                    f"{agent.name} tried to push a {self.name} but it hit the boundary",
                )

            # Check for blocking objects or other agents at the target location
            target_object = env._find_object_at(new_box_pos)
            if target_object:
                # Block if the object is an obstacle or another pushable box
                if target_object.obstacle or isinstance(target_object, PushableBox):
                    return (
                        False,
                        f"{agent.name} tried to push a {self.name} but it's blocked by {target_object.name}",
                    )
                # Allow pushing onto non-obstacle objects (like pressure plates)

            # Check for other agents
            for other_agent in env.agents:
                # Check if another agent (not the one pushing) is at the target spot
                if other_agent != agent and other_agent.get_position() == new_box_pos:
                    return (
                        False,
                        f"{agent.name} tried to push a {self.name} but {other_agent.name} is in the way",
                    )

        # Update the box's position
        self.pos = new_box_pos
        self.being_pushed = True  # Mark the box as being pushed for this step

        # Return True to allow the agent to move to the box's original position
        return True, f"{agent.name} pushed a {self.name}"

    def interact(self, agent: Any, env: Any = None) -> None:
        """Called when the agent interacts with the box directly (e.g., steps off it)."""
        # Reset being_pushed when agent moves off the box's original square or interacts differently
        # This might need adjustment based on exact interaction timing, but for now,
        # let's assume interact is called when the push action concludes or agent moves away.
        super().interact(agent, env)
        self.being_pushed = (
            False  # No longer being actively pushed in this immediate interaction
        )
        return None

    def step(self, env: Any) -> None:
        """Called each environment step."""
        # Reset being_pushed flag at the beginning of the next step if it wasn't reset by interact
        # Ensures the state is correct even if no direct interaction happened
        super().step(env)
        self.being_pushed = False


class LinkedDoor(Object):
    """A door that is linked to and activated by other objects like levers or plates."""

    def __init__(self, pos: List[int], linked_id: Any):
        super().__init__(pos, obstacle=True, consumable=False, terminal=False)
        self.linked_id = linked_id
        self.name = "linked door"
        self.is_open = False

    def copy(self) -> "LinkedDoor":
        new_door = type(self)(list(self.pos), self.linked_id)
        new_door.obstacle = self.obstacle
        new_door.is_open = self.is_open
        return new_door

    def activate(self, activator: Any = None, env: Any = None) -> Optional[str]:
        """Opens the door."""
        if not self.is_open:
            self.obstacle = False
            self.is_open = True
            # Add activator name if available
            activator_name = getattr(activator, "name", "Something")
            return f"{self.name} (ID: {self.linked_id}) was opened by {activator_name}"
        return None

    def deactivate(self, deactivator: Any = None, env: Any = None) -> Optional[str]:
        """Closes the door."""
        if self.is_open:
            self.obstacle = True
            self.is_open = False
            # Add deactivator info if available
            deactivator_info = ""
            if isinstance(deactivator, PressurePlate):
                deactivator_info = (
                    f" because pressure was released from {deactivator.name}"
                )
            elif isinstance(deactivator, Lever):
                deactivator_info = f" by {deactivator.name}"

            return f"{self.name} (ID: {self.linked_id}) was closed{deactivator_info}"
        return None

    def interact(self, agent: Any, env: Any = None) -> Optional[str]:
        """Called when agent tries to interact (e.g., walk onto) the door."""
        super().interact(agent, env)
        if self.is_open:
            return None  # Agent walks through freely
        else:
            return f"{agent.name} tried to open a locked {self.name}"


class PressurePlate(Object):
    """A plate that activates a linked object when stepped on, and deactivates when empty."""

    def __init__(self, pos: List[int], target_linked_id: Any):
        super().__init__(pos, obstacle=False, consumable=False, terminal=False)
        self.target_linked_id = target_linked_id
        self.name = "pressure plate"
        self.is_pressed = False  # Track activation state

    def copy(self) -> "PressurePlate":
        new_plate = type(self)(list(self.pos), self.target_linked_id)
        new_plate.is_pressed = self.is_pressed  # Copy state
        return new_plate

    def _find_target(self, env: Any) -> Optional[LinkedDoor]:
        """Finds the linked door in the environment."""
        if "linked_doors" in env.objects:
            for door in env.objects["linked_doors"]:
                if (
                    hasattr(door, "linked_id")
                    and door.linked_id == self.target_linked_id
                ):
                    return door
        return None

    def _check_if_pressed(self, env: Any) -> bool:
        """Check if an agent or pushable box is currently on the plate."""
        # Check agents
        for agent in env.agents:
            if agent.pos == self.pos:
                return True
        # Check pushable boxes
        if "pushable_boxes" in env.objects:
            for box in env.objects["pushable_boxes"]:
                if box.pos == self.pos:
                    return True
        return False

    def interact(self, agent: Any, env: Any = None) -> Optional[str]:
        """Called when agent steps onto the plate."""
        # Activation happens implicitly via step logic now,
        # interact can just call the base method or be removed if no specific on-step-on event needed.
        # Let's keep it simple and let step handle it.
        super().interact(agent, env)
        # We could potentially force an immediate check/activation here,
        # but relying on the step ensures consistency with boxes too.
        # The activation will happen in the step() method called shortly after.
        return None

    def step(self, env: Any) -> Optional[str]:
        """Called each step to check activation status."""
        super().step(env)
        currently_pressed = self._check_if_pressed(env)
        message = None

        if currently_pressed and not self.is_pressed:
            # Plate just became pressed
            self.is_pressed = True
            target_door = self._find_target(env)
            if target_door:
                message = target_door.activate(self, env)
                if message:
                    # Generate a clear event message
                    activator_name = "Something"
                    for agent in env.agents:
                        if agent.pos == self.pos:
                            activator_name = agent.name
                            break
                    if (
                        activator_name == "Something"
                        and "pushable_boxes" in env.objects
                    ):
                        for box in env.objects["pushable_boxes"]:
                            if box.pos == self.pos:
                                activator_name = box.name
                                break
                    message = f"{activator_name} pressed {self.name}. {message}"

        elif not currently_pressed and self.is_pressed:
            # Plate just became unpressed
            self.is_pressed = False
            target_door = self._find_target(env)
            if target_door:
                message = target_door.deactivate(self, env)
                # Message from deactivate already includes cause

        return message


class Lever(Object):
    """A lever that toggles the state of a linked object when interacted with."""

    def __init__(self, pos: List[int], target_linked_id: Any):
        super().__init__(pos, obstacle=False, consumable=False, terminal=False)
        self.target_linked_id = target_linked_id
        self.name = "lever"
        self.activated = False  # Represents the lever's own state (e.g., pulled or not)

    def copy(self) -> "Lever":
        new_lever = type(self)(list(self.pos), self.target_linked_id)
        new_lever.activated = self.activated
        return new_lever

    def _find_target(self, env: Any) -> Optional[LinkedDoor]:
        """Finds the linked door in the environment."""
        # Assumes target is always a LinkedDoor for now
        if "linked_doors" in env.objects:
            for door in env.objects["linked_doors"]:
                if (
                    hasattr(door, "linked_id")
                    and door.linked_id == self.target_linked_id
                ):
                    return door
        return None

    def interact(self, agent: Any, env: Any = None) -> Optional[str]:
        """Called when agent uses the INTERACT action on the lever."""
        super().interact(agent, env)

        # Toggle the lever's state
        self.activated = not self.activated
        lever_state_msg = "activated" if self.activated else "deactivated"
        base_message = f"{agent.name} {lever_state_msg} the {self.name}."

        door_message = None
        if env:
            target_door = self._find_target(env)
            if target_door:
                if self.activated:
                    # If lever is now active, try to activate the door
                    door_message = target_door.activate(self, env)
                else:
                    # If lever is now inactive, try to deactivate the door
                    door_message = target_door.deactivate(self, env)

        # Combine messages
        if door_message:
            return f"{base_message} {door_message}"
        else:
            return base_message


class ResetButton(Object):
    """A button that resets all objects in the environment to their initial state."""

    def __init__(self, pos: List[int]):
        super().__init__(pos, obstacle=False, consumable=False, terminal=False)
        self.name = "reset button"

    def copy(self) -> "ResetButton":
        return type(self)(list(self.pos))

    def interact(self, agent: Any, env: Any = None) -> Optional[str]:
        """Resets all objects in the environment to their state at the start of the episode."""
        super().interact(agent, env)

        if env and hasattr(env, "initial_objects") and env.initial_objects is not None:
            # Perform a deep copy to avoid modifying the stored initial state
            env.objects = copy.deepcopy(env.initial_objects)
            # Note: This does not reset agent positions or states, only objects.
            return (
                f"{agent.name} pressed the {self.name}. Objects reset to initial state."
            )
        else:
            # This case should ideally not happen if reset was called correctly
            return f"{agent.name} tried to press the {self.name}, but the initial state was not found."
