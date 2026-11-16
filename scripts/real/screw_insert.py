#!/usr/bin/env python3
"""Screw driving on the real FR3: ForceHarness (--condition A) or the no-force control (--condition B).

Find the threaded hole by touch, press the tip in, turn with the thread, and grasp again at the wrist limit.
The instruction is fixed (forceharness/prompts/real/screw_insert.md, shared by both conditions); the flags are those of
scripts/real/free_task.py (python scripts/real/screw_insert.py --help). Without --live it runs the synthetic demo, which
only checks the interface and ends with done(success=false).

    python scripts/real/screw_insert.py --live --execute --auto-confirm \\
        --calibration configs/real/force_calibration.json --condition A --max-actions 60 --max-minutes 60 \\
        --out runs/real/<name>/episode --relay-dir runs/real/<name>/relay
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forceharness.real.runner import main as run  # noqa: E402
from forceharness.real.tasks import TASKS  # noqa: E402

TASK = TASKS['screw_insert']


def main(argv=None):
    return run(argv, task=TASK)


if __name__ == '__main__':
    raise SystemExit(main())
