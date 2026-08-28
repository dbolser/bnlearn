"""Manage the bnlearn Agent Skill."""

from pathlib import Path
import argparse
import shutil


KNOWN_HARNESSES = {
    "claude": ".claude",
    "opencode": ".opencode",
    "agents": ".agents",
}


def skill_path():
    """Return the path to the bundled bnlearn Agent Skill."""
    return Path(__file__).resolve().parent / "skills" / "bnlearn"


def install_skill(harness="claude"):
    """Install the bnlearn Agent Skill into the current project.

    Parameters
    ----------
    harness : str, default='claude'
        Name of the AI coding harness. The skill is installed to:

            ./.<harness>/skills/bnlearn/

        Any harness name is accepted.
    """
    source = skill_path()

    # Keep the harness generic. A leading dot is added automatically.
    harness = harness.lstrip(".")
    destination = Path.cwd() / f".{harness}" / "skills" / "bnlearn"

    if harness not in KNOWN_HARNESSES:
        print(
            f"Warning: '{harness}' is not a known AI harness. "
            f"Installing anyway to:\n{destination}"
        )

    if not source.exists():
        raise FileNotFoundError(
            f"Bundled bnlearn skill not found: {source}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        shutil.rmtree(destination)

    shutil.copytree(source, destination)

    print(f"bnlearn skill installed to:\n{destination}")


def main():
    """Command-line interface for bnlearn skills."""
    parser = argparse.ArgumentParser(
        prog="bnlearn",
        description="bnlearn command-line interface.",
    )

    subparsers = parser.add_subparsers(dest="command")

    skill_parser = subparsers.add_parser(
        "skill",
        help="Manage the bnlearn Agent Skill.",
    )

    skill_subparsers = skill_parser.add_subparsers(
        dest="skill_command",
    )

    install_parser = skill_subparsers.add_parser(
        "install",
        help="Install the bnlearn Agent Skill.",
    )

    install_parser.add_argument(
        "--harness",
        default="claude",
        help="AI coding harness name (default: claude).",
    )

    skill_subparsers.add_parser(
        "path",
        help="Show the path to the bundled bnlearn Agent Skill.",
    )

    args = parser.parse_args()

    if args.command == "skill":
        if args.skill_command == "install":
            install_skill(harness=args.harness)

        elif args.skill_command == "path":
            print(skill_path())

        else:
            skill_parser.print_help()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()