"""Franka Hand on the real robot: one robot-side gripper command per grasp or release, over SSH.

Changed for the public release; not yet validated on hardware. The runs reported in the paper actuated the hand
through a site-specific executor and gripper service. This module replaces them with the standard Polymetis
gripper server, reached by running robot_side/franka_polymetis/gripper.py from the same deployed bundle as the
worker (one SSH command per action), and keeps the semantics the runner relies on:

- close = grasp with a fixed force (60 N by default), open = move to 80 mm; at least 3 s between commands;
- while the hand moves (at least 3.2 s), the robot state is read every 50 ms through the worker's read-only
  `state` request, which applies every robot-side state check (timestamp advancing, hardware error codes, joint
  margins, raw force caps, wrench conditioning, gripper error code); this takes the place of a separate hardware
  health monitor. A state round trip slower than 0.3 s or a width outside 0..80.2 mm is an error;
- after a close the width must be above the minimum grasp width (8 mm in the runner) and below 75 mm, else
  "No plausible grasp width" is raised (the runner turns fingers that met, <= 8 mm, into a known empty grasp);
  after an open the width must be at least 78 mm; a command the server reports as unsuccessful is an error;
- one close per trial (the ledger), and nothing is retried: any error leaves the held state unknown in the
  backend, which then refuses every further gripper action.
"""
import json
import subprocess
import time

from forceharness.real.types import BackendError

MIN_ACTION_S = 3.2          # read the state for at least this long after a command, even if the hand is done sooner
INTERVAL_S = 3.             # between two gripper commands
STATE_RTT_S = .3            # a state read slower than this is an error
MAX_WIDTH_M = .0802         # a width reading above this is not plausible for the Franka Hand
OPEN_MIN_M = .078
CLOSED_MAX_M = .075


class NoCameras:
    """Stand-in when no camera evidence is recorded around gripper actions."""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def check(self):
        pass

    def save(self, root, label, state, health):
        return {}


class RemoteGripper:
    """Runs robot_side/franka_polymetis/gripper.py on the robot PC for one action and returns its JSON reply."""

    def __init__(self, robot, *, popen=subprocess.Popen, clock=time.monotonic, sleep=time.sleep):
        self.robot, self.popen, self.clock, self.sleep = robot, popen, clock, sleep

    def command(self, op, force=None):
        from forceharness.real.transport import bundle_command, ssh_argv
        g, pc = self.robot.gripper, self.robot.robot_pc
        args = [op, '--port', str(pc.gripper_port), '--speed', f'{g.speed_m_s:g}']
        if op == 'close':
            args += ['--force', f'{g.force_N if force is None else force:g}']
        elif op == 'open':
            args += ['--width', f'{g.open_width_m:g}']
        return ssh_argv(self.robot, bundle_command(self.robot, 'gripper.py', args))

    def run(self, op, *, poll=None, min_s=0., force=None):
        """Run one action; call poll() every 50 ms until it has finished and at least min_s has passed.
        force overrides the configured grasp force (operator tools only; the runner never passes it)."""
        if op not in ('state', 'close', 'open'):
            raise ValueError('Unknown gripper operation')
        if force is not None and not 0 < force <= 140:
            raise ValueError('Grasp force must be in (0, 140] N')
        begun = self.clock()
        proc = self.popen(self.command(op, force), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            while proc.poll() is None or self.clock()-begun < min_s:
                if self.clock()-begun > self.robot.gripper.command_timeout_s:
                    raise BackendError('Gripper command timed out; outcome unknown')
                if poll is not None:
                    poll()
                self.sleep(.05)
            out, err = proc.communicate(timeout=5)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
        lines = [line for line in (out or '').splitlines() if line.startswith('{')]
        if proc.returncode or not lines:
            detail = (err or out or '').strip().splitlines()
            raise BackendError(f'Gripper command failed: {detail[-1] if detail else "exit %s" % proc.returncode}')
        reply = json.loads(lines[-1])
        if reply.get('op') != op:
            raise BackendError('Unexpected gripper reply: '+lines[-1][:200])
        return reply


def check_result(closed, reply, width, min_width_m):
    """The acceptance rules of one action (see the module docstring); raises RuntimeError."""
    if not reply.get('prev_command_successful'):
        raise RuntimeError('Gripper command did not succeed')
    if closed:
        if not min_width_m < width < CLOSED_MAX_M:
            raise RuntimeError('No plausible grasp width; stop without retry')
    elif width < OPEN_MIN_M:
        raise RuntimeError('Gripper did not open')


def run_action(port, closed, cameras, target):
    """One gripper action through the port's worker and gripper, with its ledger and evidence files."""
    ledger = port.ledger
    record = dict(close=closed)
    port_records = [record]

    def read():
        cameras.check()
        begun = time.monotonic()
        reply = port.request('state')
        if time.monotonic()-begun > STATE_RTT_S:
            raise RuntimeError('RPC state acquisition too slow')
        state = reply['state']
        if not 0 <= state['gripper_width'] <= MAX_WIDTH_M:
            raise RuntimeError('Invalid measured gripper width')
        if port.monitor:
            port.monitor()
        return state

    def health(state):
        return dict(source='robot-side worker state', error_code=state.get('error_code'),
                    policy_running=state.get('policy_running'), timestamp_ns=state.get('timestamp_ns'))

    try:
        if ledger.get('last_gripper_time') and time.time()-ledger['last_gripper_time'] < INTERVAL_S:
            raise RuntimeError('Gripper command interval too short')
        before = read()
        cameras.save(target, 'before', before, health(before))
        if closed:
            if ledger['grasp_attempts']:
                raise RuntimeError('Grasp retry prohibited')
            ledger['grasp_attempts'] = 1
        ledger['last_gripper_time'] = time.time()
        (target/'run.json').write_text(json.dumps(ledger, indent=2)+'\n')
        record.update(before=before, time_unix=time.time())
        record['response'] = port.gripper.run('close' if closed else 'open', poll=read, min_s=MIN_ACTION_S)
        after = read()
        record['after'] = after
        record['previous_command_success'] = bool(record['response'].get('prev_command_successful'))
        check_result(closed, record['response'], after['gripper_width'], port.grasp_min_width_m)
        cameras.save(target, 'after', after, health(after))
    finally:
        (target/'gripper.json').write_text(json.dumps(port_records, indent=2, default=str)+'\n')
