"""One Franka Hand action through the Polymetis gripper server on this computer; prints one JSON line.

Changed for the public release; not yet validated on hardware. The runs reported in the paper actuated the
hand through a site-specific gripper service (a fixed 60 N grasp and an 80 mm open). This script is its
replacement on the standard Polymetis path: the workstation runs it over SSH (forceharness/real/gripper.py)
once per grasp or release, while the worker keeps the arm under its guarded hold, so the gripper never
shares the worker's control loop. It never commands the arm.

    python gripper.py state
    python gripper.py close --force 60 --speed 0.05     # grasp: close until the fingers stop, at this force
    python gripper.py open --width 0.08 --speed 0.05    # move the fingers to this opening

GripperInterface.grasp(blocking=True) only waits until the server has accepted the command; the gripper
client executes it in its own loop. So this script watches is_moving and the width until the hand has
finished moving before it reads back. Above 70 N the Franka Hand runs beyond its continuous rating (140 N
is its peak): hold such a grasp only briefly.
"""
import argparse
import json
import time

CONTINUOUS_N = 70.   # Franka Hand: 70 N continuous
PEAK_N = 140.        # and 140 N peak
MAX_WIDTH_M = .08


def wait_done(gripper, before, timeout=8., poll=.05):
    """Until the hand has moved and stopped (is_moving seen True, then False), or the width changed by more
    than 2 mm and then stayed still for 0.5 s. Returns (state, settled)."""
    begun, seen_moving, last, still_since = time.time(), False, before, None
    while time.time()-begun < timeout:
        s = gripper.get_state()
        if s.is_moving:
            seen_moving, still_since = True, None
        elif seen_moving:
            return s, True
        else:
            still_since = (still_since or time.time()) if abs(s.width-last) < 1e-4 else None
            if abs(s.width-before) > .002 and still_since and time.time()-still_since > .5:
                return s, True
        last = s.width
        time.sleep(poll)
    return gripper.get_state(), False


def describe(s):
    return dict(width_m=round(float(s.width), 5), is_grasped=bool(s.is_grasped), is_moving=bool(s.is_moving),
                prev_command_successful=bool(s.prev_command_successful), error_code=int(s.error_code))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('op', choices=['state', 'close', 'open'])
    ap.add_argument('--force', type=float, default=60., help='grasp force in N (close only)')
    ap.add_argument('--speed', type=float, default=.05, help='finger speed in m/s')
    ap.add_argument('--width', type=float, default=MAX_WIDTH_M, help='opening in m (open only)')
    ap.add_argument('--port', type=int, default=50052, help='Polymetis gripper server port on this computer')
    a = ap.parse_args(argv)
    if not 0 < a.force <= PEAK_N:
        ap.error('--force must be in (0, %g] N' % PEAK_N)
    if not 0 < a.speed <= .1:
        ap.error('--speed must be in (0, 0.1] m/s')
    if not 0 < a.width <= MAX_WIDTH_M:
        ap.error('--width must be in (0, %g] m' % MAX_WIDTH_M)
    from polymetis import GripperInterface
    gripper = GripperInterface(ip_address='localhost', port=a.port)
    begun = time.time()
    first = gripper.get_state()
    before = float(first.width)
    out = dict(op=a.op, width_before_m=round(before, 5))
    if a.op == 'state':
        out.update(describe(first), settled=not first.is_moving)
    else:
        if a.op == 'close':
            gripper.grasp(speed=a.speed, force=a.force, blocking=True)
            out.update(force_N=a.force, above_continuous_rating=a.force > CONTINUOUS_N)
        else:
            gripper.goto(width=a.width, speed=a.speed, force=a.force, blocking=True)
            out.update(target_width_m=a.width)
        s, settled = wait_done(gripper, before)
        out.update(describe(s), settled=settled, speed_m_s=a.speed)
    out['elapsed_s'] = round(time.time()-begun, 3)
    print(json.dumps(out), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
