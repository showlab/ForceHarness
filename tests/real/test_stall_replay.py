"""The production tracking-stall guard replayed on a logged held-weighing motion (not a position-follower approximation)."""
import json
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pytest

from forceharness.real.plant import DemoPort, core

TRACE=Path(__file__).with_name('fixtures')/'loaded_weighing_trace.json'


def guard_at(now=0.,actual=(0.,0.,0.),target=(.004,0.,0.)):
    # Isolate only the production stall predicate. These replay trajectories
    # are never sent through a driver; other guards are tested by DemoPort.
    g=core.GuardedSession.__new__(core.GuardedSession)
    g.limits=core.Limits()
    g.progress_pos=np.array(actual);g.progress_at=now
    g.desired=np.array(target)
    g.target_history=deque([(now,g.desired.copy())]);g.target_moving=False
    return g


def replay_trace():
    trace=json.loads(TRACE.read_text());init=trace['initial']
    g=guard_at(init['time'],init['actual'],init['target'])
    phases=Counter();static_samples=0
    for kind,now,value,*phase in trace['events']:
        if kind=='target':
            g.desired=np.array(value);g.target_history.append((now,g.desired.copy()))
        else:
            g.check_stall(value,now)
            if phase[0]:phases[phase[0]]+=1
            if phase[0] in ('loaded_baseline','loaded_measure') and not g.target_moving:
                static_samples+=1
    return dict(no_stall=True,phase_samples=dict(phases),static_weigh_samples=static_samples,
                elapsed_s=now-init['time'],scope=trace['scope'])


def test_logged_loaded_motion_replay_has_no_stall():
    result=replay_trace()
    assert set(result['phase_samples'])=={'lift_to_H','loaded_baseline','loaded_measure','return_bottle'}
    assert result['static_weigh_samples']>100


def test_same_recorded_trace_reproduces_old_per_command_fault():
    """Regression control: a per-command stall predicate fails on this evidence."""
    trace=json.loads(TRACE.read_text());init=trace['initial']
    g=guard_at(init['time'],init['actual'],init['target']);actual=g.progress_pos.copy()
    failed=False
    for kind,now,value,*phase in trace['events']:
        if kind=='target':
            g.target_moving=np.linalg.norm(value-g.desired)>1e-9
            if not g.target_moving:
                g.progress_pos,g.progress_at=actual.copy(),now;g.target_history.clear()
            g.desired=np.array(value);g.target_history.append((now,g.desired.copy()))
        else:
            actual=np.array(value)
            if np.linalg.norm(actual-g.progress_pos)>.0005:
                g.progress_pos,g.progress_at=actual.copy(),now
            error=np.linalg.norm(g.desired-actual)
            if not g.target_moving or error<=.002:g.progress_at=now
            while len(g.target_history)>1 and g.target_history[1][0]<=now-4.:
                g.target_history.popleft()
            recent=np.array([p for _,p in g.target_history])
            if now-g.progress_at>4. and error>.002 and np.max(np.linalg.norm(recent-recent[0],axis=1))>.0005:
                failed=True;break
    assert failed,'Trace must expose the per-command predicate defect'


def test_blocked_hand_and_target_advancing_4mm_s_stalls_at_four_seconds():
    g=guard_at()
    for i in range(1,251):
        now=i*.02;g.desired=np.array([.004+.004*now,0,0])
        g.target_history.append((now,g.desired.copy()))
        try:g.check_stall([0,0,0],now)
        except RuntimeError as exc:
            assert 'Cartesian tracking stalled' in str(exc)
            assert 4.<=now<=4.2
            return
    pytest.fail('Moving blocked hand was not stopped')


def test_micrometre_jitter_10s_through_full_session_does_not_stall():
    p=DemoPort();p.driver.frozen=True
    with p.backend() as b:
        with b.motion(30.):
            for _ in range(30):b.step([.004,0,0])
            for i in range(220):b.step([.004+(i%2)*.0000048,0,0])
            assert p.session.fault is None and not p.session.target_moving
            assert p.clock()-p.session.progress_at<.051


def test_accumulated_path_not_net_change_and_window_expires():
    g=guard_at();start=g.desired.copy()
    for i in range(1,21):
        now=i*.05;g.desired=start+np.array([.00004*(i%2),0,0])
        g.target_history.append((now,g.desired.copy()));g.check_stall([0,0,0],now)
    np.testing.assert_array_equal(g.desired,start)
    assert g.target_moving  # 0.8 mm path, zero net change.
    g.check_stall([0,0,0],2.01)
    assert not g.target_moving and g.progress_at==2.01


def test_static_window_clears_previous_stall_time():
    g=guard_at()
    for i in range(1,151):
        now=i*.02;g.desired=np.array([.004+.004*min(now,2.),0,0])
        g.target_history.append((now,g.desired.copy()));g.check_stall([0,0,0],now)
    assert not g.target_moving and g.progress_at==3.
    for i in range(151,501):
        now=i*.02;g.target_history.append((now,g.desired.copy()));g.check_stall([0,0,0],now)
    assert g.progress_at==10.
