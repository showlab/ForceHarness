#!/usr/bin/env python3
"""Peeling on the real FR3: ForceHarness (--condition A) or the no-force control (--condition B).

Keep the clamped peeler blade in the skin of a cucumber or eggplant while moving along +x.
The instruction is fixed (forceharness/prompts/real/peel_vegetable.md, shared by both conditions); the flags are those of
scripts/real/free_task.py (python scripts/real/peel_vegetable.py --help). Without --live it runs the synthetic demo, which
only checks the interface and ends with done(success=false).

    python scripts/real/peel_vegetable.py --live --execute --auto-confirm \\
        --calibration configs/real/force_calibration.json --condition A --max-actions 60 --max-minutes 60 --force-max 20 --slide-force-default 10 \\
        --out runs/real/<name>/episode --relay-dir runs/real/<name>/relay
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forceharness.real.runner import main as run  # noqa: E402
from forceharness.real.tasks import TASKS  # noqa: E402

TASK = TASKS['peel_vegetable']


def main(argv=None):
    return run(argv, task=TASK)


if __name__ == '__main__':
    raise SystemExit(main())
