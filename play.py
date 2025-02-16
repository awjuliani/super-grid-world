import cv2
import numpy as np
from sgw.env import SuperGridWorld
from sgw.enums import Action, ControlType, ObsType
import argparse


def get_action_from_key(key, control_type, manual_collect):
    """Map keyboard input to game actions."""
    if control_type == ControlType.allocentric:
        # Allocentric controls (absolute directions)
        key_to_action = {
            ord("w"): Action.MOVE_NORTH,
            ord("d"): Action.MOVE_EAST,
            ord("s"): Action.MOVE_SOUTH,
            ord("a"): Action.MOVE_WEST,
        }
    else:
        # Egocentric controls (relative to agent's orientation)
        key_to_action = {
            ord("w"): Action.MOVE_FORWARD,
            ord("d"): Action.ROTATE_RIGHT,
            ord("a"): Action.ROTATE_LEFT,
        }

    # Add collect action if manual collection is enabled
    if manual_collect:
        key_to_action[ord("e")] = Action.COLLECT

    # Add quit key
    key_to_action[ord("q")] = "QUIT"

    return key_to_action.get(key)


def play_game():
    parser = argparse.ArgumentParser(description="Play Super Grid World")
    parser.add_argument(
        "--template",
        type=str,
        default="two_rooms",
        help="Template name for the environment (default: two_rooms)",
    )
    parser.add_argument(
        "--obs_type",
        type=str,
        default="visual_2d",
        choices=["visual_2d", "visual_3d"],
        help="Observation type (default: visual_2d)",
    )
    parser.add_argument(
        "--control",
        type=str,
        default="allocentric",
        choices=["allocentric", "egocentric"],
        help="Control type (default: allocentric)",
    )
    parser.add_argument(
        "--manual_collect",
        action="store_true",
        help="Enable manual collection with E key",
    )
    parser.add_argument(
        "--resolution", type=int, default=512, help="Display resolution (default: 512)"
    )
    args = parser.parse_args()

    # Create environment
    env = SuperGridWorld(
        template_name=args.template,
        control_type=ControlType(args.control),
        manual_collect=args.manual_collect,
        resolution=args.resolution,
        obs_type=ObsType(args.obs_type),
    )

    # Reset environment
    obs = env.reset()

    # Create window
    cv2.namedWindow("Super Grid World", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Super Grid World", args.resolution, args.resolution)

    print("\nControls:")
    if args.control == "allocentric":
        print("W: Move North")
        print("S: Move South")
        print("A: Move West")
        print("D: Move East")
    else:
        print("W: Move Forward")
        print("A: Rotate Left")
        print("D: Rotate Right")
    if args.manual_collect:
        print("E: Collect/Interact")
    print("Q: Quit")
    print("\nEvents will be displayed in the terminal.")

    while True:
        # Display the game
        frame = obs[0]  # Get first agent's observation
        if isinstance(frame, np.ndarray):
            # Convert to BGR if necessary
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Super Grid World", frame)

        # Get keyboard input
        key = cv2.waitKey(100) & 0xFF
        action = get_action_from_key(
            key, ControlType(args.control), args.manual_collect
        )

        if action == "QUIT":
            break
        elif action is not None:
            # Find action index
            action_idx = env.valid_actions.index(action)
            obs, rewards, dones, info = env.step([action_idx])

            # Display events
            if info["events"]:
                print("\nEvents:", info["events"])

            # Display rewards
            if rewards[0] != 0:
                print(f"\nReward: {rewards[0]}")

            # Check if episode is done
            if dones[0]:
                print("\nEpisode finished! Resetting environment...")
                obs = env.reset()

    cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    play_game()
