import argparse

from arena.policies import REGISTRY
from arena.utilities import Runner, Saver, logging


def main():
    parser = argparse.ArgumentParser(description="Run battle arena simulations")

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--policies",
        type=str,
        nargs="+",
        default=[
            "aggressive",
            "aggressive",
            "scanner",
            "scanner",
            "zombie",
            "zombie",
            "coward",
            "coward",
            "random",
            "random",
            "custom",
            "custom",
        ],
        help="List of policy types to use (default: 2 of each)",
    )

    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1_000,
        help="Number of episodes to run (default: 5)",
    )

    parser.add_argument(
        "--arena-height",
        type=int,
        default=15,
        help="Arena height (default: 15)",
    )

    parser.add_argument(
        "--arena-width",
        type=int,
        default=15,
        help="Arena width (default: 15)",
    )

    parser.add_argument(
        "--min-steps",
        type=int,
        default=100,
        help="Minimum steps per episode (default: 100)",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode (default: 500)",
    )

    parser.add_argument(
        "--vision-radius",
        type=int,
        default=2,
        help="Vision radius for agents (default: 2)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    parser.add_argument(
        "--list-policies",
        action="store_true",
        help="List available policies and exit",
    )

    args = parser.parse_args()

    # Setup logging
    logging.setup_logging(args.log_level)
    logger = logging.get_logger(__name__)

    if args.list_policies:
        print("Available policies:")
        for policy_name in REGISTRY.keys():
            print(f"  - {policy_name}")
        return

    # Convert policy names to policy classes
    try:
        policy_types = [REGISTRY[policy.lower()] for policy in args.policies]
    except KeyError as e:
        print(
            f"Error: Unknown policy {e}. Use --list-policies to see available options."
        )
        return

    logger.info(
        f"Starting simulation with {len(policy_types)} agents, "
        f"{args.num_episodes} episodes"
    )

    runner = Runner(
        policy_types=policy_types,
        num_episodes=args.num_episodes,
        arena_height=args.arena_height,
        arena_width=args.arena_width,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        vision_radius=args.vision_radius,
        seed=args.seed,
    )

    trajectories = runner.run()

    logger.info(f"Completed {len(trajectories)} episodes")

    saver = Saver()
    saver.save(trajectories)

    logger.info(f"Saved to {saver.session_dir}")


if __name__ == "__main__":
    main()
