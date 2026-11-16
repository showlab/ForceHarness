"""Transport to the robot-side worker: one foreground SSH session per episode, JSON lines (layer 1/2 plumbing).

SSHPort starts robot_side/franka_polymetis/worker.py on the real-time computer from a content-addressed bundle
(deployed by scripts/real/deploy_robot_side.py and verified with sha256sum on every start), sends one JSON
request per line and waits for the matching reply. No hardware dependency is imported here. JSON telemetry is
drained on a reader thread even while the caller processes a primitive report. Closing stdin ends the session:
while the worker is alive, EOF stops an owned controller; a crashed worker cannot terminate the stock
controller, which then retains its last target. RobotPort adds the gripper (see gripper.py) and its per-episode
ledger. The protocol is documented in docs/robot_protocol.md.
"""
import hashlib
import json
import math
from pathlib import Path
import queue
import shlex
import subprocess
import threading
import time

from forceharness.real.config import load_robot_config
from forceharness.real.types import BackendError

ROOT = Path(__file__).resolve().parents[2]
ROBOT_SIDE = ROOT/'robot_side/franka_polymetis'
FILES = ('guard.py','driver.py','worker.py','selftest.py','wrench.py','gripper.py')
SSH_OPTIONS = ('-o','BatchMode=yes','-o','ConnectTimeout=5')


def bundle_info():
    """Bundle name (robot_side_<first 12 hex of the hash of the file hashes>) and the per-file sha256."""
    hashes = {name:hashlib.sha256((ROBOT_SIDE/name).read_bytes()).hexdigest() for name in FILES}
    digest = hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:12]
    return 'robot_side_'+digest,hashes


def remote_path(path):
    """A shell word for a path on the robot PC; a leading ~/ expands to the remote $HOME."""
    if path == '~':
        return '"$HOME"'
    if path.startswith('~/'):
        return '"$HOME"/'+shlex.quote(path[2:])
    return shlex.quote(path)


def bundle_dir(robot,version):
    return robot.robot_pc.deploy_dir.rstrip('/')+'/'+version


def ssh_argv(robot,command):
    pc = robot.robot_pc
    return ['ssh',*SSH_OPTIONS,*pc.ssh_options,pc.ssh_host,command]


def bundle_command(robot,script,args):
    """Remote shell command: enter the deployed bundle, verify its checksums, run one bundle script."""
    version,_ = bundle_info()
    pc = robot.robot_pc
    # All shell words are constants or quoted values. No user shell code.
    return ('cd '+remote_path(bundle_dir(robot,version))
            +' && sha256sum --status -c SHA256SUMS'
            +' && export CONDA_PREFIX='+remote_path(pc.conda_prefix)
            +' && exec '+remote_path(pc.python)+' -u '+script+(' '+' '.join(args) if args else ''))


def worker_command(config,robot=None,*,compile_only=False):
    robot = robot or load_robot_config()
    pc = robot.robot_pc
    limits = dict(speed=config.speed_m_s,f_cap=config.f_cap,t_cap=config.t_cap)
    if not config.gripper_width_check:
        limits['width_check'] = 0   # only when a task asks for it; otherwise the worker's default
    args = ['--hardware-config',remote_path(pc.hardware_config),'--limits-json',shlex.quote(json.dumps(limits)),
            '--robot-port',str(pc.robot_port),'--gripper-port',str(pc.gripper_port)]
    if compile_only:
        args += ['--compile-only']
    elif config.motion_enabled:
        args += ['--enable-motion']
    return bundle_command(robot,'worker.py',args)


class SSHPort:
    def __init__(self,config,log_dir,*,robot=None):
        self.config,self.log_dir = config,Path(log_dir)
        self.robot = robot or load_robot_config()
        self.proc = None
        self.metadata = self.limits = None
        self.inbox = queue.Queue()
        self.ident = 0
        self.closed = False
        self._lock = threading.Lock()

    def open(self):
        if self.proc is not None or self.closed:
            raise BackendError('Transport is single use')
        self.log_dir.mkdir(parents=True,exist_ok=True)
        self.log = (self.log_dir/'nuc.jsonl').open('x')
        self.stderr = (self.log_dir/'nuc.stderr').open('x')
        try:
            self.proc = subprocess.Popen(
                ssh_argv(self.robot,worker_command(self.config,self.robot)),
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,bufsize=1)
            self.thread = threading.Thread(target=self._read,daemon=True)
            self.thread.start()
            ready = self._next(20.)
            if (ready.get('type')!='ready' or ready.get('version')!='fh_continuous/2'
                    or ready.get('motion_enabled')!=self.config.motion_enabled
                    or ready.get('metadata',{}).get('frame')!='world'
                    or ready.get('metadata',{}).get('reference')!='panda_link8'):
                raise BackendError('Unexpected worker handshake: '+str(ready))
            self.metadata,self.limits = ready['metadata'],ready['limits']
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            for line in self.proc.stdout:
                self.log.write(line); self.log.flush()
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # Pinocchio/torch import diagnostics can use stdout.
                if row.get('type') in ('ready','reply','error','fault','fault_release_start'):
                    self.inbox.put(row)
        finally:
            self.inbox.put(dict(type='error',error='NUC worker EOF'))

    def _next(self,timeout):
        deadline=time.monotonic()+timeout
        releasing=False
        while True:
            try:
                row = self.inbox.get(timeout=max(0.,deadline-time.monotonic()))
            except queue.Empty as exc:
                raise BackendError('NUC reply timeout; outcome unknown') from exc
            if row.get('type')=='fault_release_start':
                if not releasing:
                    # Keep stdin open until the robot side finishes its guarded ramp.
                    # Repeated notifications cannot extend this bounded wait.
                    deadline=time.monotonic()+3.2;releasing=True
                continue
            if row.get('type') in ('error','fault'):
                raise BackendError('NUC fault: '+str(row.get('error')))
            if releasing:
                continue  # A queued old reply cannot hide the latched fault.
            return row

    def request(self,op,**kwargs):
        if self.proc is None or self.closed or self.proc.poll() is not None:
            raise BackendError('NUC worker is unavailable')
        if not self._lock.acquire(blocking=False):
            raise BackendError('Overlapping control requests')
        try:
            self.ident += 1
            cmd = dict(id=self.ident,op=op,**kwargs)
            self.proc.stdin.write(json.dumps(cmd,allow_nan=False)+'\n')
            self.proc.stdin.flush()
            # A robot-side release may take up to about 3.6 s (a 25 mm pressing lead at 8 mm/s).
            row = self._next(15. if op=='begin' else (.30 if op=='step' else (4.5 if op in ('finish','hold','close') else 1.)))
            if row.get('type')!='reply' or row.get('id')!=self.ident or row.get('ok') is not True:
                raise BackendError('Unexpected/replayed reply')
            return row
        except BaseException:
            self.close()
            raise
        finally:
            self._lock.release()

    def grip(self,closed):
        raise BackendError('Gripper disabled')   # RobotPort owns the gripper

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.proc is not None:
            try:
                self.proc.stdin.close()
            except (OSError,ValueError):
                pass
            try:
                self.proc.wait(timeout=2.)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=1.)
                except subprocess.TimeoutExpired:
                    self.proc.kill(); self.proc.wait(timeout=1.)
            if hasattr(self,'thread'):
                self.thread.join(timeout=1.)
            self.proc.stdout.close()
        for name in ('log','stderr'):
            if hasattr(self,name):
                getattr(self,name).close()


class RobotPort(SSHPort):
    """SSHPort plus the Franka Hand, with the per-episode gripper ledger the runner relies on.

    Gripper actions run only while this worker holds the arm or no controller runs at all. Each grasp is
    one "trial": a new trial needs the previous release confirmed, and a trial allows one close. The
    runner's port has no trial budget (max_trials=None); a budget may still be set. The width checks of
    each action are those of gripper.check_result."""
    monitor=None

    def __init__(self, *args, grasp_min_width_m=.025, camera_factory=None, max_trials=6, gripper=None, **kwargs):
        if not math.isfinite(grasp_min_width_m) or not .008 <= grasp_min_width_m <= .025:
            raise ValueError('Grasp minimum width must be between 8 and 25 mm')
        if max_trials is not None and not (isinstance(max_trials, int) and max_trials >= 1):
            raise ValueError('max_trials must be a positive integer or None')
        super().__init__(*args, **kwargs)
        self.grasp_min_width_m = grasp_min_width_m
        self.camera_factory = camera_factory
        self.max_trials = max_trials
        self._gripper = gripper

    @property
    def gripper(self):
        if self._gripper is None:
            from forceharness.real.gripper import RemoteGripper
            self._gripper = RemoteGripper(self.robot)
        return self._gripper

    def next_weigh_trial(self,trial):
        limit=getattr(self,'max_trials',6)
        if trial!=getattr(self,'trial',0)+1 or trial<1 or (limit is not None and trial>limit):
            raise BackendError('Sequential six-trial gripper budget required')
        if trial>1 and not getattr(self,'released',False):
            raise BackendError('Previous bottle release not confirmed; no retry')
        self.trial=trial;self.released=False
        self.ledger=dict(grasp_attempts=0,phase='gripper_trial',trial=trial,
                         last_gripper_time=getattr(self,'last_gripper_time',0.))

    def grip(self,closed):
        if not self.config.motion_enabled or self.closed or not getattr(self,'trial',0):
            raise BackendError('Gripper disabled or no gripper trial')
        state=self.request('state')
        if state['session']['mode'] not in ('holding','idle'):
            raise BackendError('Gripper only during own hold or stopped controller')
        if state['session']['mode']=='idle' and state['state']['policy_running']:
            raise BackendError('Another controller is active')
        from forceharness.real.gripper import NoCameras, run_action
        target=self.log_dir/('gripper_'+str(time.monotonic_ns()));target.mkdir()
        factory = self.camera_factory or NoCameras
        with factory() as cameras:
            try:
                run_action(self,closed,cameras,target)
                self.released=not closed
            finally:
                self.last_gripper_time=self.ledger.get('last_gripper_time',0.)
