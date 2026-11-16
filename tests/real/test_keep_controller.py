"""Keeping the controller between primitives: lifecycle, unchanged guards and episode closure (offline plants)."""
import json
import numpy as np
import pytest

from forceharness.real.primitives import ContinuousPrimitives
from forceharness.real.plant import DemoPort
from test_transport import port


def test_three_primitives_start_once_and_default_still_starts_three_times():
    for keep,starts in [(False,3),(True,1)]:
        p=DemoPort()
        with p.backend(keep_controller=keep) as b:
            prim=ContinuousPrimitives(b)
            for delta in ([0,0,.02],[.02,0,0],[-.02,0,0]):
                report=prim.move_by(*delta)
                assert report.termination=='completed'
                assert b.holding is keep
                if keep:
                    b.sleep(5.);b.refresh()
                    assert report.derived['episode']['controller_starts']==1
            assert p.driver.starts==starts
        assert p.driver.stops==starts and not p.driver.running
        if keep:
            assert len(b.episode_report()['holds'])==3
            assert all(h['max_displacement_m']<.001 for h in b.episode_report()['holds'])


def test_sixty_second_idle_timeout_is_not_fault_and_next_primitive_restarts():
    events=[];p=DemoPort(emit=events.append)
    with p.backend(keep_controller=True) as b:
        ContinuousPrimitives(b).move_by(0,0,.002)
        p.sleep(59.9);b.refresh();assert b.holding and p.driver.running
        p.sleep(.2);b.refresh();assert not b.holding and not p.driver.running
        assert b.fault is p.session.fault is None
        timeout=next(x for x in events if x['type']=='hold_timeout')
        assert 60.<=timeout['elapsed_s']<60.03
        ContinuousPrimitives(b).move_by(0,0,-.002)
        assert p.driver.starts==2
    assert p.driver.stops==2


def test_six_actions_use_two_starts_without_hardware():
    from forceharness.real.config import BackendConfig
    p=DemoPort(BackendConfig(motion_enabled=True,f_cap=10.),plane=.34)
    # The ideal fixture only models Cartesian gains (750 N/m); use the same
    # contact fixture as the contact tests to represent the stock hybrid controller.
    update=p.driver.update
    def contact_response(*args):
        update(*args);p.driver.w[:3]=p.driver.bias[:3]+2*(p.driver.w[:3]-p.driver.bias[:3])
    p.driver.update=contact_response
    with p.backend(keep_controller=True) as b:
        prim=ContinuousPrimitives(b,empty_contact_check=True,stop_protocol='r9')
        for delta in ([0,0,.02],[.02,0,0],[-.02,0,0]):
            r=prim.move_by(*delta);assert r.termination=='completed'
            b.sleep(5.);b.refresh()
        r=prim.move_until_contact([0,0,-1],max_dist=.05,f_threshold=5.,speed=.004)
        assert r.termination=='contact'
        b.sleep(5.);b.refresh()
        r=prim.move_by(0,0,.03);assert r.termination=='completed'
        assert p.driver.starts==1
        b.sleep(65.);b.refresh();assert not b.holding and b.fault is None
        r=prim.move_by(0,0,-.01);assert r.termination=='completed'
        assert p.driver.starts==2
    assert p.driver.stops==2


@pytest.mark.parametrize('stale_timestamp',[False,True])
def test_hold_does_not_disable_guard_deadline_or_timestamp_checks(stale_timestamp):
    p=DemoPort();s=p.session;s.begin();s.hold(sleep=p.clock.sleep)
    if stale_timestamp:
        p.driver.override['timestamp_ns']=s.last_stamp
        with pytest.raises(RuntimeError,match='timestamp stalled'):p.sleep(.2)
    else:
        p.clock.sleep(.16)
        with pytest.raises(RuntimeError,match='deadline'):s.tick()
    assert p.driver.stops==1 and not p.driver.running


def test_hold_force_change_requires_three_consecutive_ticks_and_soft_stops():
    events=[];p=DemoPort(emit=events.append);s=p.session
    s.begin();s.hold(sleep=p.clock.sleep)
    p.driver.override['wrench']=(p.driver.bias+[8.1,0,0,0,0,0]).tolist()
    for _ in range(2):p.sleep(.02)
    assert s.holding
    p.driver.override['wrench']=p.driver.bias.tolist();p.sleep(.02)
    p.driver.override['wrench']=(p.driver.bias+[8.1,0,0,0,0,0]).tolist()
    for _ in range(2):p.sleep(.02)
    with pytest.raises(RuntimeError,match='hold_contact'):p.sleep(.02)
    assert p.driver.stops==1 and not s.holding
    assert any(x['type']=='normal_stop' for x in events)


@pytest.mark.parametrize('change,match',[
    ({'flange_pos':[.4051,0,.35]},'hold_unexpected_motion'),
    ({'wrench':[25.,0,0,0,0,0]},'cap'),
    ({'wrench':[0,0,0,3.,0,0]},'cap'),
    ({'error_code':1},'hardware'),
    ({'flange_pos':[.7,0,.35]},'workspace'),
])
def test_hold_safety_faults_stop_and_latch(change,match):
    p=DemoPort();p.session.begin();p.session.hold(sleep=p.clock.sleep)
    p.driver.override.update(change)
    with pytest.raises((RuntimeError,ValueError),match=match):p.sleep(.02)
    assert p.driver.stops==1 and not p.driver.running and p.session.fault


def test_hold_freezes_target_updates_telemetry_and_allows_width_change():
    p=DemoPort();s=p.session;s.begin();s.hold(sleep=p.clock.sleep)
    target=s.desired.copy();p.driver.p[1]+=.001
    p.driver.width=.05;p.sleep(.1)
    assert s.holding and s.status()['holds'][-1]['max_displacement_m']==pytest.approx(.001)
    np.testing.assert_array_equal(s.desired,target)
    assert p.driver.updates[-1]['heartbeat']>p.driver.updates[-2]['heartbeat']
    s.resume(3.,[150.]*6);assert s.initial_width==.05
    p.driver.width=.03;p.clock.sleep(.02)
    with pytest.raises(RuntimeError,match='width changed'):s.tick()


def test_resume_rejects_different_gains_and_requires_hold():
    p=DemoPort();s=p.session
    with pytest.raises(RuntimeError,match='holding'):s.resume(3.,[150.]*6)
    s.begin();s.hold(sleep=p.clock.sleep);target=s.desired.copy()
    with pytest.raises(ValueError,match='gains differ'):s.resume(3.,[75.]*6)
    assert s.holding and p.driver.starts==1
    s.resume(3.,[150.]*6)
    np.testing.assert_array_equal(s.desired,target)
    assert s.last_step==p.clock() and s.started==p.clock()


def test_resume_rebases_primitive_displacement_joint_budget_and_lease():
    p=DemoPort();s=p.session;s.begin(10.)
    for _ in range(135):
        s.command([.008,0,0],[0,0,0],[0,0,0],[0,0,0,1]);p.clock.sleep(.05);s.tick()
    s.hold(sleep=p.clock.sleep);p.sleep(5.);previous=p.driver.p.copy()
    p.driver.q[0]+=.3
    s.resume(30.,[150.]*6)   # the budget covers the longer walk to the 120 mm envelope
    np.testing.assert_allclose(s.initial['flange_pos'],previous)
    p.driver.q[0]+=.3
    for _ in range(135):
        s.command([.008,0,0],[0,0,0],[0,0,0],[0,0,0,1]);p.clock.sleep(.05);s.tick()
    assert p.driver.p[0]>.5 and p.driver.starts==1
    assert s.limits.displacement==.15 and s.limits.joint_delta==.9 and s.limits.lead==.008   # per-primitive envelopes; lead unchanged
    # The new primitive still has the SAME target bound (120 mm from the resume point).
    with pytest.raises(ValueError,match='120 mm'):
        for _ in range(200):
            s.command([.008,0,0],[0,0,0],[0,0,0],[0,0,0,1]);p.clock.sleep(.05);s.tick()


def test_gain_change_while_holding_finishes_before_new_start():
    p=DemoPort()
    with p.backend(keep_controller=True) as b:
        with b.motion():b.step([0,0,0])
        b.set_kp(75)
        assert not p.driver.running and not b.holding
        with b.motion():b.step([0,0,0])
        assert p.driver.starts==2 and p.driver.start_gains[-1][0]==375


def test_airborne_tare_while_holding_and_gripper_confirmation_remains():
    p=DemoPort()
    with p.backend(keep_controller=True) as b:
        with b.motion():b.step([0,0,0])
        b.rezero_bias(steps=10,unloaded=True)
        assert b.holding and p.driver.starts==1
        with pytest.raises(RuntimeError,match='confirmation'):b.command_gripper(True)
        b.command_gripper(True,confirmed=True)
        assert b.holding and b._closed is True
        with pytest.raises(RuntimeError,match='Explicit'):b.rezero_bias(steps=10,unloaded=True)


@pytest.mark.parametrize('reason',['force_cap','torque_cap','timeout','unexpected_motion'])
def test_unsuccessful_primitive_never_enters_hold(reason):
    p=DemoPort()
    with p.backend(keep_controller=True) as b:
        with b.motion():
            b.step([0,0,0]);b.motion_result(reason)
        assert not p.driver.running and not b.holding


def test_timeout_while_holding_an_object_keeps_the_controller():
    """Only the held-object timeout is retained; other failures still finish."""
    for reason,kept in (('timeout',True),('force_cap',False)):
        p=DemoPort()
        with p.backend(keep_controller=True) as b:
            with b.motion():b.step([0,0,0])
            b.command_gripper(True,confirmed=True)
            with b.motion():
                b.step([.001,0,0]);b.motion_result(reason)
            assert b.holding is kept and p.driver.running is kept
            assert p.driver.starts==1


@pytest.mark.parametrize('reason',['contact','stalled_short'])
def test_orderly_unsuccessful_result_soft_finishes_in_both_modes(reason):
    for keep in (False,True):
        events=[];p=DemoPort(emit=events.append);ops=[];request=p.request
        def logged(op,**kwargs):
            ops.append(op);return request(op,**kwargs)
        p.request=logged
        with p.backend(keep_controller=keep) as b:
            with b.motion():
                b.step([.001,0,0]);b.motion_result(reason)
            assert ops[-1]=='finish' and 'hold' not in ops and 'end' not in ops
            assert any(e['type']=='normal_stop' for e in events)
            assert not p.driver.running and not b.holding


def test_contact_end_enters_hold_and_next_move_resumes():
    p=DemoPort(plane=.348)
    with p.backend(keep_controller=True) as b:
        prim=ContinuousPrimitives(b)
        r=prim.move_until_contact([0,0,-1],max_dist=.006,f_threshold=.5)
        assert r.termination=='contact' and b.holding
        b.sleep(1.)
        r=prim.move_by(0,0,.003)
        assert r.termination=='completed' and b.holding and p.driver.starts==1


@pytest.mark.parametrize('error',[KeyboardInterrupt,RuntimeError])
def test_episode_exception_while_holding_soft_finishes(error):
    events=[];p=DemoPort(emit=events.append)
    with pytest.raises(error):
        with p.backend(keep_controller=True) as b:
            with b.motion():b.step([0,0,0])
            raise error('script exits')
    assert not p.driver.running and p.driver.stops==1
    assert any(x['type']=='normal_stop' for x in events)


def test_worker_hold_survives_lease_and_eof_stops(tmp_path,monkeypatch):
    p,cfg=port(tmp_path,monkeypatch);p.open()
    p.request('begin',budget=3.);p.request('hold')
    import time
    time.sleep(.4)
    state=p.request('state')
    assert state['session']['mode']=='holding' and state['state']['policy_running']
    p.request('resume',budget=3.,kp=[150.]*6)
    p.request('hold');p.close()
    rows=[json.loads(x) for x in (tmp_path/'nuc.jsonl').read_text().splitlines()]
    assert rows[0]['version']=='fh_continuous/2'
    assert sum(x['type']=='begin' for x in rows)==1
    assert rows[-1]['type']=='end' and p.proc.poll()==0
    assert not any(x['type']=='normal_stop' for x in rows)  # EOF is immediate.


class SaggingPlant(DemoPort):
    """A lifted load hangs below the target (6-7 mm for 0.42 kg on the stock stiffness). 4 mm here: well past the 1 mm goal tolerance, inside the fixture's 5 mm hold and stop checks."""
    def __init__(self, **kw):
        super().__init__(**kw)
        update, self.grip_z = self.driver.update, None
        def sag(pose, kp, kd, heartbeat):
            update(pose, kp, kd, heartbeat)
            if self.driver.held:                    # supported at the grasp height, fully hanging 4 mm above it
                self.grip_z = pose[2] if self.grip_z is None else self.grip_z
                self.driver.p = self.driver.p-[0, 0, min(.004, max(0., pose[2]-self.grip_z))]
        self.driver.update = sag


def test_held_move_completes_at_the_goal_instead_of_dithering_into_a_stall():
    p = SaggingPlant(payload=.42)
    with p.backend(keep_controller=True) as b:
        prim = ContinuousPrimitives(b)
        with b.motion(): b.step([0, 0, 0])
        b.command_gripper(True, confirmed=True)
        rep = prim.move_by(0, 0, .03)
        assert rep.termination == 'completed' and b.fault is None
        assert abs(b.eef_pos[2]-rep.derived['active_end_pos'][2]) < .01
