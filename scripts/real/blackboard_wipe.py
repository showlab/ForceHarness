#!/usr/bin/env python3
"""Whiteboard wiping on the real FR3: ForceHarness (--condition A) or the no-force control (--condition B).

Hold the eraser, find the board by touch, press and slide; judge from new images whether the writing is gone.
The instruction is fixed (forceharness/prompts/real/blackboard_wipe.md, shared by both conditions); the flags are those of
scripts/real/free_task.py (python scripts/real/blackboard_wipe.py --help). Without --live it runs the synthetic demo, which
only checks the interface and ends with done(success=false).

    python scripts/real/blackboard_wipe.py --live --execute --auto-confirm \\
        --calibration configs/real/force_calibration.json --condition A \\
        --out runs/real/<name>/episode --relay-dir runs/real/<name>/relay
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forceharness.real.runner import main as run  # noqa: E402
from forceharness.real.tasks import TASKS  # noqa: E402

TASK = TASKS['blackboard_wipe']


def main(argv=None):
    return run(argv, task=TASK)


if __name__ == '__main__':
    raise SystemExit(main())
