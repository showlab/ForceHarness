#!/usr/bin/env python3
"""Force gate for a newly connected arm: read the wrench estimate exactly as the harness does, without moving.

The robot-side worker runs read-only (no --enable-motion): it never starts a controller, and Polymetis' own
default controller keeps holding the arm. The reading is the harness's: w = (J^T)^+ tau_ext at the flange, with
force_sign from the force calibration file. Results are printed and written as JSON (default
runs/check_force/<timestamp>.json, the robot-side log next to it). The exit code is 0 unless the stream cannot be
read; verdicts are the caller's, and the paper's FR3 values are printed for comparison.

    python scripts/real/check_force.py                         # 30 s still: noise, drift, rate, residual, sigma_min
    python scripts/real/check_force.py --sign                  # 3 presses on the fingertips: sign of the reaction z
    python scripts/real/check_force.py --mass --known-grams 320          # a known mass hung from the hand
    python scripts/real/check_force.py --mass --known-grams 320 --close-gripper   # ... gripped by the hand

--sign: after a countdown, push the fingertips UP from below with your hand for about 3 s, gently (5-10 N), as a
table pushes on a hand pressing it: the reaction z should come out positive. With --press down you press on the
fingertips from above instead, and it should come out negative. --mass: an empty reading first, then the known mass
in or hanging from the gripper; the change of the still reading gives the weight, as in the harness's weight line.
Only with --close-gripper does anything move: the gripper closes (and later opens) after you type CLOSE (OPEN),
through the same robot-side gripper command as the runner. Keep every reading well below the 25 N raw cap: at the
cap the worker stops and the check ends.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from forceharness.real import force_check as fc  # noqa: E402
from forceharness.real.config import BackendConfig, load_robot_config  # noqa: E402
from forceharness.real.types import BackendError  # noqa: E402


def open_port(robot, log_dir):
    from forceharness.real.transport import SSHPort
    port = SSHPort(BackendConfig(motion_enabled=False), log_dir, robot=robot)
    port.open()
    return port


def countdown(say, seconds=3, sleep=time.sleep):
    for k in range(seconds, 0, -1):
        say(f'  {k} ...')
        sleep(1.)


def run_checks(a, port, *, ask=input, say=print, sleep=time.sleep, clock=time.monotonic, gripper=None,
               calibration=None):
    """All requested checks on an open port; returns the result dict (no files)."""
    cal = calibration or {}
    sign = cal.get('force_sign')
    if sign not in (-1, 1):
        raise ValueError('the force calibration has no force_sign of -1 or +1')
    out = dict(mode='mass' if a.mass else 'sign' if a.sign else 'still', force_sign=sign,
               calibration=str(a.calibration), guard_limits=port.limits,
               metadata={k: port.metadata.get(k) for k in ('hz', 'frame', 'reference', 'controller')},
               reference_FR3=fc.REFERENCE)
    if not (a.sign or a.mass):
        say(f'Reading for {a.seconds:g} s: keep everything away from the arm and the hand.')
        rows = fc.sample(port, a.seconds, a.hz, clock=clock, sleep=sleep)
        out['still'] = fc.stream_stats(rows, sign)
        bias = cal.get('bias_force')
        if bias is not None:
            out['still']['raw_minus_calibration_bias_N'] = [round(m-b, 4) for m, b in
                                                            zip(out['still']['raw_force_mean_N'], bias)]
        return out
    if a.sign:
        trials = []
        where = 'UP from below' if a.press == 'up' else 'DOWN from above'
        for i in range(a.repeats):
            ask(f'Trial {i+1}/{a.repeats}: hands off the robot, then press Enter for the rest reading. ')
            rest = fc.sample(port, 1.5, a.hz, clock=clock, sleep=sleep)
            say(f'Get ready to push the fingertips {where}, gently (5-10 N), for 3 s, when it says NOW.')
            countdown(say, sleep=sleep)
            say('  NOW: push and hold ...')
            pressed = fc.sample(port, 3., a.hz, clock=clock, sleep=sleep)
            say('  release.')
            result = fc.press_result(rest, pressed, sign, a.press)
            say(f'  reaction change {result["delta_N"]} N: z {result["delta_z_N"]:+.2f} N '
                f'(expected {"+" if result["expected_sign_z"] > 0 else "-"}), '
                + ('as expected' if result['matches'] else 'NOT as expected (or too small to tell)'))
            trials.append(result)
        out['sign'] = dict(trials=trials, all_match=all(t['matches'] for t in trials))
        return out
    # mass
    ask('Empty reading: nothing may touch the hand or the fingers; press Enter. ')
    empty = fc.sample(port, a.seconds, a.hz, clock=clock, sleep=sleep)
    width_empty = empty[-1]['gripper_width']
    closed_here = False
    if a.close_gripper:
        if gripper is None:
            raise ValueError('no gripper')
        answer = ask('Hold the object between the fingers (by its cap or handle), keep your fingers clear, and type '
                     'CLOSE to close the gripper at the runner\'s fixed force (anything else aborts): ')
        if answer.strip() != 'CLOSE':
            out['mass'] = dict(aborted='gripper close not confirmed')
            return out
        reply = gripper.run('close')
        closed_here = True
        out['gripper_close'] = reply
        if reply.get('width_m', 0) <= .008:
            out['mass'] = dict(aborted=f'empty grasp (width {reply.get("width_m", 0)*1000:.0f} mm)')
            return out
        ask('Let go of the object so that it hangs from the gripper, then press Enter. ')
    else:
        ask('Place or hang the known mass in or from the gripper, let go of it, then press Enter. ')
    sleep(1.)
    loaded = fc.sample(port, a.seconds, a.hz, clock=clock, sleep=sleep)
    width_loaded = loaded[-1]['gripper_width']
    closed = width_empty >= .078 and width_loaded < .075
    correction = cal.get('gripper_delta_wrench') if closed else None
    out['mass'] = dict(fc.mass_result(empty, loaded, sign, a.known_grams, correction, closed=closed),
                       width_empty_m=round(width_empty, 4), width_loaded_m=round(width_loaded, 4),
                       empty=fc.stream_stats(empty, sign) if len(empty) >= 12 else None,
                       loaded=fc.stream_stats(loaded, sign) if len(loaded) >= 12 else None)
    if closed_here:
        answer = ask('Hold the object and type OPEN to open the gripper (anything else leaves it closed): ')
        if answer.strip() == 'OPEN':
            out['gripper_open'] = gripper.run('open')
        else:
            out['gripper_open'] = 'left closed by the operator'
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--sign', action='store_true', help='sign check by pressing on the fingertips (read-only)')
    mode.add_argument('--mass', action='store_true', help='weigh a known mass held by the hand')
    ap.add_argument('--seconds', type=float, default=None, help='still reading length (default 30; 5 per --mass reading)')
    ap.add_argument('--hz', type=float, default=20., help='request rate (the harness steps at about 15-20 Hz)')
    ap.add_argument('--repeats', type=int, default=3, help='--sign trials')
    ap.add_argument('--press', choices=['up', 'down'], default='up',
                    help='--sign: push the fingertips up from below (expected +z) or down from above (expected -z)')
    ap.add_argument('--known-grams', type=float, default=None, help='--mass: the true mass')
    ap.add_argument('--close-gripper', action='store_true',
                    help='--mass: close the gripper on the object after an explicit CLOSE (the only motion)')
    ap.add_argument('--calibration', type=Path, default=None,
                    help='force calibration file (default: configs/real/force_calibration.json if it exists, '
                         'else the example file with the reference FR3 sign and bias)')
    ap.add_argument('--robot-config', type=Path, default=None)
    ap.add_argument('--out', type=Path, default=None, help='result JSON (default runs/check_force/<timestamp>.json)')
    a = ap.parse_args(argv)
    if a.seconds is None:
        a.seconds = 5. if a.mass else 30.
    if not 1 <= a.seconds <= 600 or not 1 <= a.hz <= 50 or not 1 <= a.repeats <= 20:
        ap.error('--seconds 1..600, --hz 1..50, --repeats 1..20')
    if a.close_gripper and not a.mass:
        ap.error('--close-gripper belongs to --mass')
    stamp = datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')
    out_path = a.out or ROOT/f'runs/check_force/{stamp}.json'
    if a.calibration is None:
        own = ROOT/'configs/real/force_calibration.json'
        a.calibration = own if own.exists() else ROOT/'configs/real/force_calibration.example.json'
        print(f'calibration file: {a.calibration}', flush=True)
    calibration = json.loads(a.calibration.read_text())
    robot = load_robot_config(a.robot_config)
    if robot.is_example or robot.placeholders():
        ap.error('fill in configs/real/robot.json first (copy configs/real/robot.example.json)')
    gripper = None
    if a.close_gripper:
        from forceharness.real.gripper import RemoteGripper
        gripper = RemoteGripper(robot)
    try:
        port = open_port(robot, out_path.with_name(out_path.stem+'_robot_side'))
    except (BackendError, OSError) as exc:
        print(f'The wrench stream could not be opened: {exc}', file=sys.stderr)
        return 2
    try:
        result = run_checks(a, port, gripper=gripper, calibration=calibration)
    except BackendError as exc:
        print(f'The wrench stream stopped: {exc}', file=sys.stderr)
        return 2
    finally:
        port.close()
    result.update(time=datetime.now().astimezone().isoformat(timespec='seconds'), robot_config=robot.source)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('guard_limits', 'metadata')}, indent=1))
    print('FR3 reference values (paper): '+'; '.join(f'{k}: {v}' for k, v in fc.REFERENCE.items()))
    print('written:', out_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
