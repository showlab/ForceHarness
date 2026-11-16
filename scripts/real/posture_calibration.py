#!/usr/bin/env python3
"""One-time empty-hand posture calibration of the force zero (optional; about 20 minutes). MOVES THE ARM.

The runner runs it by itself the first time (when configs/real/posture_bias_fr3.json is missing, unless
--no-auto-calibration) and reuses the file afterwards; use this script to recalibrate on purpose, e.g. after the
camera mount, the pads or the cable routing changed. The shipped file has use_model false: on the FR3 a posture
model did not beat a constant zero, so only the per-episode zero reading is used. Needs the gripper open and
empty and the space 25-40 cm above the table clear; motion goes only through the runner's primitives with every
guard, at a raw force cap of 15 N.

    python scripts/real/posture_calibration.py --demo --path /tmp/posture.json      # synthetic plant
    python scripts/real/posture_calibration.py --live --execute                     # the real arm
"""
import argparse
from datetime import date
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from forceharness.real import posture_bias  # noqa: E402
from forceharness.real import posture_calibration as posture_cal  # noqa: E402
from forceharness.real.config import BackendConfig, load_robot_config  # noqa: E402
from forceharness.real.runner import TCP_TRANSLATION, FreePrimitives, demo  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--live', action='store_true')
    mode.add_argument('--demo', action='store_true')
    ap.add_argument('--execute', action='store_true', help='required with --live')
    ap.add_argument('--path', type=Path, default=posture_bias.DEFAULT_FILE)
    ap.add_argument('--calibration', type=Path, default=ROOT/'configs/real/force_calibration.json',
                    help='force calibration the backend uses for its guards')
    ap.add_argument('--robot-config', type=Path, default=None)
    ap.add_argument('--out', type=Path, default=ROOT/f'runs/real/posture_calibration_{date.today():%Y%m%d}')
    a = ap.parse_args(argv)
    if a.live:
        if not a.execute:
            ap.error('--live requires --execute')
        try:
            robot = load_robot_config(a.robot_config).require_live()
        except (OSError, ValueError) as exc:
            ap.error(str(exc))
        from forceharness.real.backend import ContinuousBackend
        from forceharness.real.transport import RobotPort
        cfg = BackendConfig(motion_enabled=True, f_cap=15, tcp_translation=TCP_TRANSLATION)
        a.out.mkdir(parents=True, exist_ok=False)
        port = RobotPort(cfg, a.out/'nuc', robot=robot)
        port.next_weigh_trial(1)
        b = ContinuousBackend(port, config=cfg, calibration=json.loads(a.calibration.read_text()), keep_controller=True)
    else:
        b, _, _ = demo()
    with b:
        p = FreePrimitives(b, confirm=lambda kind, state: False, tick=lambda: None, stop_protocol='r9')
        doc = posture_cal.run_calibration(p, a.path, out_dir=a.out, log=lambda m: print(m, flush=True))
    print(json.dumps(dict(path=str(a.path), use_model=doc['use_model'], stats=doc['stats']), ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
