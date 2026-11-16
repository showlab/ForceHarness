#!/usr/bin/env python3
"""Read, close or open the Franka Hand by hand from the workstation, outside an episode. Never moves the arm.

Changed for the public release; not yet validated on hardware. It runs robot_side/franka_polymetis/gripper.py on
the robot PC over SSH (deploy the bundle first: scripts/real/deploy_robot_side.py --deploy), the same path the
runner uses for grasp and release. Typical use: close the gripper on an object by hand, then start an episode
that already holds it with --resume-held "<object>". The operator runs it at the robot with the hands clear of the
fingers, and never while a live episode runs (the runner owns the gripper then; this script refuses on Linux
when it sees one).

    python scripts/real/gripper_check.py width               # read only
    python scripts/real/gripper_check.py close               # grasp with the runner's fixed force (60 N)
    python scripts/real/gripper_check.py close --force 30    # any force up to 140 N (above 70 N only briefly)
    python scripts/real/gripper_check.py open                # open to 80 mm

Every command is appended to runs/real_gripper/gripper_log.jsonl.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from forceharness.real.config import load_robot_config  # noqa: E402
from forceharness.real.gripper import RemoteGripper  # noqa: E402

LOG = ROOT/'runs/real_gripper/gripper_log.jsonl'
EMPTY_CLOSE_M = .008     # closed at or below this: the fingers met, nothing is held
OPEN_M = .078            # the runner counts a gripper as open from 78 mm
CONTINUOUS_N = 70.       # Franka Hand continuous rating
PEAK_N = 140.            # and its peak


def live_runner():
    """Command line of a running live runner, if any (it owns the gripper during an episode). Linux only."""
    proc_root = Path('/proc')
    if not proc_root.is_dir():
        return None
    for proc in proc_root.iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc/'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        text = [a.decode(errors='replace') for a in args if a]
        if '--live' in text and any('scripts/real/' in a and not a.endswith('gripper_check.py') for a in text):
            return ' '.join(text)
    return None


def verdict(command, w):
    mm = w*1000
    if command == 'close':
        if w <= EMPTY_CLOSE_M:
            return (f'Reading {mm:.0f} mm: the two fingers met, so nothing seems to be held (open, put the object right '
                    'between the fingers and hold it, then close again). If you can see that it is held, the width '
                    'reading is off: see the hint for opening.')
        if w < .075:
            return (f'Holding something, width {mm:.0f} mm. To start an episode holding it, pass --resume-held '
                    '"<what it holds>" (in the monitor: start the episode as holding that object).')
        return f'Width {mm:.0f} mm: it does not look closed; check the gripper and its state in the vendor\'s Desk tool.'
    if w >= OPEN_M:
        return f'Open ({mm:.0f} mm); an episode can start normally.'
    return (f'Only {mm:.0f} mm (needs >= 78 mm). If the fingers are visibly fully open, the width reading is too small '
            'overall: the gripper server homed with something between the fingers and took its width as 0. Remove it, '
            'clear the space around the fingers, and restart the Polymetis gripper server on the robot PC (it homes on '
            'start). If the Desk tool shows the end effector as not connected, deal with that first.')


def main(argv=None, *, gripper=None, log=LOG):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=['width', 'close', 'open'])
    ap.add_argument('--force', type=float, default=None, help=f'close force in N (default: the runner\'s, up to {PEAK_N:g})')
    ap.add_argument('--robot-config', type=Path, default=None)
    a = ap.parse_args(argv)
    if a.force is not None and not 0 < a.force <= PEAK_N:
        ap.error(f'--force must be in (0, {PEAK_N:g}] N')
    running = live_runner()
    if running and a.action != 'width':
        print('A live episode is running; the runner owns the gripper, so nothing is sent:\n  '+running[:200])
        return 2
    if gripper is None:
        robot = load_robot_config(a.robot_config)
        if robot.is_example or robot.placeholders():
            ap.error('fill in configs/real/robot.json first (copy configs/real/robot.example.json)')
        gripper = RemoteGripper(robot)
    op = 'state' if a.action == 'width' else a.action
    got = gripper.run(op, force=a.force if a.action == 'close' else None)
    record = dict(time=datetime.now().astimezone().isoformat(timespec='seconds'), command=a.action, **got)
    if a.action != 'width':
        record['verdict'] = verdict(a.action, got['width_m'])
        if got.get('settled') is False:
            record['verdict'] += ' (the hand had not finished moving within 8 s, so this may not be the final width; read it again)'
        if a.action == 'close' and got.get('force_N', 0) > CONTINUOUS_N:
            record['verdict'] += (f' ({got["force_N"]:g} N is above the continuous rating of {CONTINUOUS_N:g} N: do not keep '
                                  'it closed for long, the hand may overheat, report an error and open)')
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False)+'\n')
    if a.action == 'width':
        print(f'width {got["width_m"]*1000:.1f} mm')
    else:
        print(f'{got["width_before_m"]*1000:.0f} mm -> {got["width_m"]*1000:.0f} mm. {record["verdict"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
