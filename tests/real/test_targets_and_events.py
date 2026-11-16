"""Stall-compensated free-motion targets, target release, and the real contact debounce (offline plants)."""
import numpy as np
import pytest

from forceharness.real.events import RealEventDetector
from forceharness.real.plant import DemoPort
from forceharness.real.primitives import ContinuousPrimitives
from forceharness.real.targets import StallCompensatedTarget


def test_stall_adds_bounded_lead_and_reached_position_retracts_bias():
    c=StallCompensatedTarget([.005,0,0],[0,0,0])
    targets=[c.update(np.zeros(3),c.goal,.05) for _ in range(100)]
    assert targets[0][0]==pytest.approx(.005)
    assert targets[-1][0]==pytest.approx(.008)
    assert max(np.linalg.norm(p) for p in targets)<=.008000001
    bias=c.bias.copy()
    c.update(c.goal,c.goal,.05)
    assert np.linalg.norm(c.bias)<np.linalg.norm(bias)
    for _ in range(20): c.update(c.goal,c.goal,.05)
    assert c.retracted


def test_moving_robot_does_not_accumulate_stall_bias():
    c=StallCompensatedTarget([.05,0,0],[0,0,0])
    for i in range(20):
        p=np.array([i*.0002,0,0])
        c.update(p,p+[.004,0,0],.05)
    np.testing.assert_allclose(c.bias,0,atol=1e-12)


def test_release_large_lead_extends_duration_without_exceeding_speed():
    events=[]
    p=DemoPort(emit=events.append)
    p.driver.frozen=True
    with p.backend() as b:
        with b.motion(5.):
            for _ in range(20): b.step([.008,0,0])
    stop=next(r for r in events if r['type']=='normal_stop')
    assert .9<stop['release_elapsed_s']<1.2
    targets=np.array([r['pose'][:3] for r in p.driver.updates])
    assert np.max(np.linalg.norm(np.diff(targets,axis=0),axis=1))<=.000400001
    np.testing.assert_allclose(targets[-1],p.driver.p,atol=.0001)


def test_release_force_fault_terminates_immediately():
    p=DemoPort()
    p.session.begin(5.)
    update=p.driver.update
    def force_fault(*args):
        update(*args)
        p.driver.override['wrench']=[25.,0,0,0,0,0]
    p.driver.update=force_fault
    with pytest.raises(RuntimeError,match='cap'):
        p.session.finish(sleep=p.clock.sleep)
    assert len(p.driver.updates)==1
    assert p.driver.stops==1 and not p.driver.running


def test_compensation_continues_inside_acceptance_until_one_mm():
    c=StallCompensatedTarget([.005,0,0],[.003,0,0])
    for _ in range(10):c.update([.003,0,0],c.goal,.05)
    assert c.bias[0]>0  # 2 mm satisfies acceptance, but not the internal goal.
    before=c.bias.copy()
    c.update([.0041,0,0],c.goal,.05)
    assert np.linalg.norm(c.bias)<np.linalg.norm(before)


def test_cap_stall_is_bounded_and_progress_resets_timer():
    c=StallCompensatedTarget([.005,0,0],[0,0,0])
    for _ in range(60):
        c.update([0,0,0],c.goal,.05)
        if c.stalled_short:break
    assert c.stalled_short and c.at_cap_s==pytest.approx(1.)
    c.update([.0003,0,0],c.goal,.05)
    assert not c.stalled_short and c.at_cap_s==0


def feed(detector,t,f,torque=0):
    return {e.kind for e in detector.update(t,np.array([f,0,0]),
        np.array([0,0,torque]),np.zeros(3),np.zeros(3),np.zeros(3),False)}


def test_contact_grace_and_three_consecutive_raw_samples():
    d=RealEventDetector(lambda:0.)
    for t in (.05,.1,.15,.2,.25):assert 'contact_start' not in feed(d,t,6.)
    assert 'contact_start' not in feed(d,.30,5.)
    assert 'contact_start' not in feed(d,.35,5.)
    assert 'contact_start' not in feed(d,.40,4.9)  # resets even in hysteresis band
    assert 'contact_start' not in feed(d,.45,5.)
    assert 'contact_start' not in feed(d,.50,5.)
    assert 'contact_start' in feed(d,.55,5.)
    d.reset()
    for i in range(20):assert 'contact_start' not in feed(d,i*.05,3.1)


def test_grace_never_masks_force_or_torque_cap():
    d=RealEventDetector(lambda:0.)
    kinds=feed(d,.05,25.,3.)
    assert {'force_cap','torque_cap'}<=kinds and 'contact_start' not in kinds


def test_probe_uses_compensation_only_on_free_return(monkeypatch):
    p=DemoPort()
    start_z=p.driver.p[2]
    update_plant=p.driver.update
    released=[False]
    def return_deadband(pose,*args):
        update_plant(pose,*args)
        if p.driver.starts==2 and not released[0]:
            if pose[2]>start_z+.001:released[0]=True
            else:p.driver.p[2]=min(p.driver.p[2],start_z-.002)
    p.driver.update=return_deadband
    with p.backend() as b:
        prim=ContinuousPrimitives(b)
        assert prim.f_contact==5. and isinstance(prim.detector,RealEventDetector)
        seen=[]
        original=StallCompensatedTarget.update
        def update(self,*args,**kwargs):
            seen.append((prim._inside_probe,prim._preserve_bias,kwargs.get('enabled',True)))
            return original(self,*args,**kwargs)
        monkeypatch.setattr(StallCompensatedTarget,'update',update)
        rep=prim.probe(amplitude=.01)
        assert seen and any(enabled for _,_,enabled in seen)
        assert all(not inside and preserving for inside,preserving,_ in seen)
        assert rep.derived['return_report']['derived']['compensation_peak_m']>0
