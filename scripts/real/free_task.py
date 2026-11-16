#!/usr/bin/env python3
"""Free task on the real FR3: the operator writes the task, the model gets every primitive of its condition.

    python scripts/real/free_task.py --instruction "..." \\
        --out runs/real/<name>/episode --relay-dir runs/real/<name>/relay            # synthetic demo
    python scripts/real/free_task.py --live --execute --auto-confirm \\
        --calibration configs/real/force_calibration.json --condition A --instruction "..." \\
        --out runs/real/<name>/episode --relay-dir runs/real/<name>/relay
    python scripts/real/free_task.py score --out runs/real/<name>/episode --outcome success --note "..."

The model answers through the relay folder (the web monitor, or a person writing cmd_<n>.json); see
docs/real_robot.md for every flag.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forceharness.real.runner import main as run  # noqa: E402


def main(argv=None):
    return run(argv)


if __name__ == '__main__':
    raise SystemExit(main())
