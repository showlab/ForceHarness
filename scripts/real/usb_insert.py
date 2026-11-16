#!/usr/bin/env python3
"""USB-A plug insertion on the real FR3: ForceHarness (--condition A) or the no-force control (--condition B).

Reach a port, find it by a light sliding search until the plug drops, and seat it.
The instruction is fixed (forceharness/prompts/real/usb_insert.md, shared by both conditions); the flags are those of
scripts/real/free_task.py (python scripts/real/usb_insert.py --help). Without --live it runs the synthetic demo, which
only checks the interface and ends with done(success=false).

    python scripts/real/usb_insert.py --live --execute --auto-confirm \\
        --calibration configs/real/force_calibration.json --condition A --force-max 20 --insert-force-max 20 \\
        --out runs/real/<name>/episode --relay-dir runs/real/<name>/relay
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forceharness.real.runner import main as run  # noqa: E402
from forceharness.real.tasks import TASKS  # noqa: E402

TASK = TASKS['usb_insert']


def main(argv=None):
    return run(argv, task=TASK)


if __name__ == '__main__':
    raise SystemExit(main())
