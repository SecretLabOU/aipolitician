"""Shared subprocess utilities for launcher scripts."""

import subprocess
import sys


def run_script(command: list[str]) -> int:
    """Run a command and return its exit status."""
    try:
        completed = subprocess.run(command, check=False)
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        print(f"Error: failed to execute command: {exc}", file=sys.stderr)
        return 127
    if completed.returncode < 0:
        return 128 + abs(completed.returncode)
    return completed.returncode
