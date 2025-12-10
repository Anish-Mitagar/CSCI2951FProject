import argparse
from pathlib import Path

from arena.utilities import Loader, Renderer


def render_all(sessions_dir: str, fps: int = 2):
    loader = Loader(input_dir=sessions_dir)
    renderer = Renderer()

    sessions = loader.list_sessions()
    if not sessions:
        print("No sessions found")
        return

    print(f"Found {len(sessions)} session(s)\n")

    for session_dir in sessions:
        session_name = session_dir.name
        print(f"Rendering session: {session_name}")
        trajectories_with_paths = loader.load_session(session_name)

        for trajectory, episode_dir in trajectories_with_paths:
            print(f"  Rendering episode {trajectory.episode}...")
            renderer.save_gif(trajectory, episode_dir, fps=fps)

        print()


def render_session(session_path: str, fps: int = 2):
    session_path = Path(session_path)
    input_dir = str(session_path.parent)
    session_name = session_path.name

    loader = Loader(input_dir=input_dir)
    renderer = Renderer()

    print(f"Rendering session: {session_name}")
    trajectories_with_paths = loader.load_session(session_name)

    if not trajectories_with_paths:
        print(f"No trajectories found in session: {session_name}")
        return

    print(f"Found {len(trajectories_with_paths)} episode(s)\n")

    for trajectory, episode_dir in trajectories_with_paths:
        print(f"Rendering episode {trajectory.episode}...")
        renderer.save_gif(trajectory, episode_dir, fps=fps)

    print("Done!")


def render_episode(episode_path: str, fps: int = 2):
    episode_path = Path(episode_path)
    session_path = episode_path.parent
    input_dir = str(session_path.parent)
    session_name = session_path.name

    loader = Loader(input_dir=input_dir)
    renderer = Renderer()

    print(f"Rendering episode from {episode_path}")
    trajectory, episode_dir = loader.load_trajectory(session_name, episode_path.name)

    renderer.save_gif(trajectory, episode_dir, fps=fps)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Render battle arena episodes to GIFs")

    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Frames per second for GIF (default: 2)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Render all sessions
    all_parser = subparsers.add_parser(
        "all",
        help="Generate GIFs for all sessions",
    )
    all_parser.add_argument(
        "--path",
        type=str,
        default="sessions",
        help="Path to sessions directory",
    )

    # Render specific session
    session_parser = subparsers.add_parser(
        "session",
        help="Generate GIFs for a specific session",
    )
    session_parser.add_argument(
        "--path",
        type=str,
        help="Path to session directory",
    )

    # Render specific episode
    episode_parser = subparsers.add_parser(
        "episode",
        help="Generate GIF for a specific episode",
    )
    episode_parser.add_argument(
        "--path",
        type=str,
        help="Path to episode directory",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "all":
        render_all(args.path, fps=args.fps)

    elif args.command == "session":
        render_session(args.path, fps=args.fps)

    elif args.command == "episode":
        render_episode(args.path, fps=args.fps)


if __name__ == "__main__":
    main()
