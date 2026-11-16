"""Robot-side JSON-lines worker: one guarded session per process, started by the workstation over SSH.

Read-only unless started with --enable-motion. One 50 Hz loop on the real-time computer checks the raw
wrench and state (guard.GuardedSession.tick) and updates the controller target; workstation step requests
arrive at about 20 Hz on stdin and are answered on stdout. The workstation's lease is enforced only while
this worker is alive: the installed stock controller has no independent heartbeat watchdog. Closing stdin
(the SSH session ending) ends the session and stops an owned controller. Protocol: docs/robot_protocol.md.
"""
import argparse
import fcntl
import json
import os
import queue
import sys
import threading
import time

from guard import GuardedSession,Limits


def emit(row):
    print(json.dumps(row,allow_nan=False,separators=(',',':')),flush=True)


def serve(driver,allow_motion=False,limits=None):
    # Logging is outside the servo/server mutex; raw timings identify stalls.
    driver.telemetry = emit
    inbox = queue.Queue(maxsize=4)
    lost = threading.Event()
    def reader():
        try:
            for line in sys.stdin:
                if len(line)>16384:
                    raise ValueError('Command too long')
                inbox.put_nowait(json.loads(line))
        finally:
            lost.set()
    threading.Thread(target=reader,daemon=True).start()
    session = GuardedSession(driver,limits=limits,emit=emit)
    session.release_cancel=lost.is_set
    emit(dict(type='ready',version='fh_continuous/2',motion_enabled=allow_motion,
              metadata=driver.metadata(),limits=session.limits.as_dict()))
    previous_id = 0
    pending = None
    due = time.monotonic()
    def reply(ident,**fields):
        emit(dict(type='reply',id=ident,ok=True,session=session.status(),**fields))
    try:
        while not lost.is_set():
            now = time.monotonic()
            if now>=due:
                session.tick()
                due = max(due+.02,time.monotonic())
            if pending is not None and now>=pending[1]:
                reply(pending[0],state=session.read_checked())
                pending = None
            try:
                cmd = inbox.get(timeout=max(.001,min(.01,due-time.monotonic())))
            except queue.Empty:
                continue
            ident,op = cmd.get('id'),cmd.get('op')
            if type(ident) is not int or ident<=previous_id or pending is not None:
                raise ValueError('Overlapping/replayed command')
            previous_id = ident
            if op=='close':
                session.end(); reply(ident); break
            if op=='state':
                reply(ident,state=session.read_checked())
                continue
            if not allow_motion:
                raise RuntimeError('Read-only worker; motion is disabled')
            if op=='begin':
                state = session.begin(cmd['budget'])
                due = time.monotonic()
                reply(ident,state=state)
            elif op=='resume':
                state=session.resume(cmd['budget'],cmd['kp'])
                due=time.monotonic()
                reply(ident,state=state)
            elif op=='end':
                session.end(); reply(ident,state=session.read_checked())
            elif op in ('finish','hold'):
                getattr(session,op)(cancel=lost.is_set)
                due = time.monotonic()
                reply(ident,state=session.read_checked())
            elif op=='set_kp':
                result = session.set_kp(cmd['kp'])
                if session.active:
                    session.tick()
                reply(ident,**result)
            elif op=='step':
                session.command(cmd['dpos'],cmd['drot'],cmd['tcp_translation'],cmd['tcp_quaternion'],
                                cmd.get('lead'))
                session.tick()
                pending = (ident,time.monotonic()+session.limits.dt)
            else:
                raise ValueError('Unsupported operation')
    except BaseException as exc:
        session.fail(exc)
        emit(dict(type='error',error=session.fault))
    finally:
        session.end()


def lock_path():
    """One lock for every bundle in the deploy directory (the parent of this bundle's folder)."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'.worker.lock')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--enable-motion',action='store_true')
    ap.add_argument('--hardware-config',required=True,help='Polymetis franka_hardware.yaml (joint limits)')
    ap.add_argument('--compile-only',action='store_true',help='Read state and compile policy, never send it')
    ap.add_argument('--limits-json',default='{}',help='Only tighter limits are accepted')
    ap.add_argument('--robot-port',type=int,default=50051,help='Polymetis robot server port on this computer')
    ap.add_argument('--gripper-port',type=int,default=50052,help='Polymetis gripper server port on this computer')
    args = ap.parse_args()
    # Exclusive among ForceHarness workers; other clients of the robot still need the operator's
    # coordination and the controller-ownership checks. No background service is left behind.
    with open(lock_path(),'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from driver import PolymetisDriver
        driver = PolymetisDriver(args.hardware_config,args.robot_port,args.gripper_port)
        if args.compile_only:
            import io
            policy = driver.torch.jit.script(driver.make_policy())
            serialized = io.BytesIO()
            driver.torch.jit.save(policy,serialized)
            serialized.seek(0)
            policy = driver.torch.jit.load(serialized)
            torch = driver.torch
            state = driver.robot.get_robot_state()
            sample = dict(joint_positions=torch.tensor(state.joint_positions),
                          joint_velocities=torch.tensor(state.joint_velocities))
            result = policy(sample)
            if policy.is_terminated() or not bool(torch.isfinite(result['joint_torques']).all()):
                raise RuntimeError('Offline forward rejected state or produced non-finite output')
            policy.update({'joint_pos_desired':policy.joint_pos_desired.detach().clone()})
            result = policy(sample)
            if policy.is_terminated() or not bool(torch.isfinite(result['joint_torques']).all()):
                raise RuntimeError('Offline target update rejected or produced non-finite output')
            emit(dict(type='compile_check',ok=True,hardware_motion=False,metadata=driver.metadata(),
                      serialization=True,forward_without_send=True,target_update=True,
                      runtime_gain_update=False,controller_watchdog=False))
        else:
            serve(driver,args.enable_motion,Limits(**json.loads(args.limits_json)))


if __name__=='__main__':
    main()
