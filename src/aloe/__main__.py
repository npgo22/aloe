"""Entry point for aloe.

aloe gui            → launch NiceGUI web interface (default)
aloe cli [options]  → run batch parameter sweep from the command line
"""

import sys

HELP_TEXT = """Usage:

    aloe gui            Launch NiceGUI web interface (default)
    aloe cli [options]  Run CLI batch/single simulation (see aloe cli --help)

Examples:
    uv run aloe gui
    uv run aloe cli --help
    uv run aloe cli --single --thrust 12000 --burn-time 15 --output-dir runs
"""


def main():
    """Run the aloe application."""
    args = sys.argv[1:]

    # Global help that does not import NiceGUI
    if "-h" in args or "--help" in args:
        print(HELP_TEXT)
        sys.exit(0)

    # CLI mode
    if args and args[0] == "cli":
        from .cli import run_cli

        run_cli(args[1:])
        return

    # GUI mode (default)
    if args and args[0] == "gui":
        args = args[1:]
    # Preserve any extra args for NiceGUI (if supplied)
    sys.argv = [sys.argv[0], *args]
    from nicegui import ui

    from . import gui  # noqa: F401

    ui.run(host="0.0.0.0", port=8080, title="Aloe: Nolan's Sim", reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
