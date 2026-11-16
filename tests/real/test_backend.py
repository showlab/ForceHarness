"""ContinuousBackend contracts, faults and compatibility with the shared primitives (synthetic plant)."""
import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from forceharness.real.config import BackendConfig
from forceharness.real.backend import ContinuousBackend
from forceharness.real.primitives import ContinuousPrimitives
from forceharness.real.plant import DemoPort,core


def test_construction_readonly_and_scope():
    p = DemoPort(BackendConfig())
    with p.backend() as b:
        ContinuousPrimitives(b)
        assert p.driver.starts==0 and not b.active
        with pytest.raises(RuntimeError,match='disabled'):
            with b.motion(): pass
    assert p.driver.grips==[]


def test_continuous_step_and_real_gain_mapping():
    p = DemoPort()
    with p.backend() as b:
        gains = b.set_kp([40,75,150,30,150,150])
        with b.motion():
            np.testing.assert_allclose(gains['Kx'],[200,375,750,3,15,15])
            np.testing.assert_allclose(gains['Kxd'],np.array([37]*3+[2]*3)*np.sqrt(np.array([40,75,150,30,150,150])/150))
            start = b.eef_pos.copy()
            stamps = []
            for _ in range(20):
                stamps.append(b.step([.001,0,0]).robot_timestamp_ns)
            assert .0079<np.linalg.norm(b.eef_pos-start)<.0081
            assert len(set(stamps))==20 and p.driver.starts==1
    assert p.driver.stops==1 and not p.driver.running
    assert p.driver.updates[-1]['Kx']==gains['Kx']


@pytest.mark.parametrize('kp',[0,-1,151,float('nan'),[150]*5,[150,150,150,150,150,151],True])
def test_bad_gains_stop_and_latch(kp):
    p = DemoPort()
    with p.backend() as b:
        with pytest.raises((ValueError,RuntimeError)):
            with b.motion():
                b.set_kp(kp)
        assert not p.driver.running
        with pytest.raises(RuntimeError,match='fault'):
            with b.motion(): pass


@pytest.mark.parametrize('override,match',[
    ({'wrench':[25,0,0,0,0,0]},'cap'),
    ({'wrench':[0,0,0,3,0,0]},'cap'),
    ({'wrench':[float('nan')]*6},'finite'),
    ({'flange_pos':[.7,0,.35]},'workspace'),
    ({'flange_pos':[.42,0,.35]},'Tracking'),
    ({'joints':[0,0,0,-2.6,0,.44,0]},'margin'),
    ({'joints':[1.0,-.48,.03,-2.6,.03,2.08,.75]},'joint delta'),        # joint_delta limit 0.9 rad
    ({'flange_quat':R.from_rotvec([2.3,0,0]).as_quat().tolist()},'rotation'),   # 0.84 rad from the fixture's initial pose (> 0.60)
    ({'policy_running':False},'controller stopped'),
    ({'error_code':1},'hardware'),
    ({'smin':.001},'conditioning'),
    ({'residual':2},'conditioning'),
    ({'gripper_width':.05},'width changed'),
])
def test_fault_stops_owned_controller(override,match):
    p = DemoPort()
    p.session.begin()
    p.driver.override.update(override)
    with pytest.raises((RuntimeError,ValueError),match=match):
        p.session.tick()
    assert not p.driver.running and p.session.fault and p.driver.stops==1


def test_live_worker_checks_workstation_lease_and_guard_period():
    p = DemoPort()
    p.session.begin()
    for _ in range(16):
        p.clock.sleep(.02)
        if p.clock()-p.session.last_step>.3:
            with pytest.raises(RuntimeError,match='lease'): p.session.tick()
        else: p.session.tick()
    assert not p.driver.running
    p = DemoPort()
    p.session.begin()
    p.clock.sleep(.16)
    with pytest.raises(RuntimeError,match='deadline'): p.session.tick()
    assert not p.driver.running


def test_stale_robot_stream_and_foreign_controller():
    p = DemoPort()
    p.driver.override['timestamp_ns']=100
    p.session.read_checked()
    p.clock.sleep(.16)
    with pytest.raises(RuntimeError,match='stalled'): p.session.tick()
    p = DemoPort()
    p.driver.running = True
    with pytest.raises(RuntimeError,match='Another'): p.session.begin()
    assert p.driver.stops==0 and p.driver.starts==0 and p.driver.running


def test_combined_tcp_flange_rate_limit():
    p = DemoPort()
    p.session.begin()
    offset = np.array([0,0,.2466])
    for _ in range(20):
        old = p.session.desired.copy()
        new = p.session.command([.003,0,0],[0,.05,0],offset,[0,0,0,1])
        oldtcp = old[:3]+R.from_quat(old[3:]).apply(offset)
        newtcp = new[:3]+R.from_quat(new[3:]).apply(offset)
        assert np.linalg.norm(new[:3]-old[:3])<=.000400001
        assert np.linalg.norm(newtcp-oldtcp)<=.000400001
        p.clock.sleep(.05); p.session.tick()


def test_free_empty_hold_can_tare_but_contact_still_requires_calibration():
    p = DemoPort()
    with ContinuousBackend(p,config=p.config,clock=p.clock,sleep=p.clock.sleep) as b:
        prim = ContinuousPrimitives(b)
        with pytest.raises(RuntimeError,match='calibration'):
            prim.move_until_contact([0,0,-1],max_dist=.001)
        assert p.driver.starts==0
        with pytest.raises(RuntimeError,match='Explicit'): b.rezero_bias()
        b.rezero_bias(unloaded=True)
        assert not b.calibration_valid()
        rep=prim.hold(.1)
        assert rep.derived['force_sign_calibrated'] is False
        assert not b.calibration_valid() and b.force_sign is None
        assert p.driver.starts==1


def test_unknown_holding_and_gripper_one_attempt():
    p = DemoPort()
    p.driver.width = .05
    with p.backend() as b:
        with pytest.raises(RuntimeError,match='unknown'): b.command_gripper(False,confirmed=True)
        assert not p.driver.grips
    p = DemoPort()
    with p.backend() as b:
        b.command_gripper(True,confirmed=True)
        b.command_gripper(True,confirmed=True)
        assert p.driver.grips==[True]
        with pytest.raises(RuntimeError,match='explicitly'):
            with b.motion(): b.step([0,0,0],grip=-1)
        with pytest.raises(RuntimeError,match='fault'): b.command_gripper(False,confirmed=True)
    assert p.driver.grips==[True]


def test_exception_and_delayed_reply_stop_without_retry():
    p = DemoPort()
    with p.backend() as b:
        with pytest.raises(KeyboardInterrupt):
            with b.motion(): raise KeyboardInterrupt
        assert p.driver.stops==1
    p = DemoPort()
    with p.backend() as b:
        original = p.request
        def delayed(op,**kw):
            result = original(op,**kw)
            if op=='step': p.clock.sleep(.21)
            return result
        p.request = delayed
        with pytest.raises(RuntimeError,match='250 ms'):
            with b.motion(): b.step([0,0,0])
        assert not p.driver.running


@pytest.mark.parametrize('delay',[0.,.21])
def test_stale_step_reports_timestamp_independently_of_roundtrip(delay):
    p = DemoPort()
    with p.backend() as b:
        original = p.request
        def stale_reply(op,**kw):
            before = b._state['timestamp_ns']
            result = original(op,**kw)
            if op=='step':
                result['state']['timestamp_ns'] = before
                p.clock.sleep(delay)
            return result
        p.request = stale_reply
        with pytest.raises(RuntimeError,match='Robot timestamp did not advance') as error:
            with b.motion():
                b.step([0,0,0])
        assert ('round trip exceeded 250 ms' in str(error.value)) == bool(delay)
        assert not p.driver.running and p.driver.stops==1 and b.fault


def test_stall_and_limit_tightening():
    with pytest.raises(ValueError): core.Limits(f_cap=26)
    with pytest.raises(ValueError): core.Limits(joint_margin=.02)
    p = DemoPort()
    p.driver.frozen = True
    with p.backend() as b:
        with pytest.raises(RuntimeError,match='stalled'):
            with b.motion():
                # the target must keep advancing, not sit at the lead cap.
                for i in range(120): b.step([.0025+.00003*i,0,0])
        assert not p.driver.running


def test_invalid_arguments_do_not_activate():
    p = DemoPort()
    with p.backend() as b:
        prim = ContinuousPrimitives(b)
        for name,args in [('move_by',dict(dx=.1,dy=0,dz=0,speed=.1)),
                          ('hold',dict(secconds=1)),('push',dict(direction=[0,0,-1],target_force=30)),
                          ('rotate_z',dict(angle_deg=40))]:   # per-primitive rotation cap 0.55 rad (31.5 deg)
            with pytest.raises(ValueError): prim.call(name,**args)
        assert not p.driver.starts


def test_contact_controller_uses_the_default_gain_and_search_hop():
    p = DemoPort(plane=.349)
    with p.backend() as b:
        prim = ContinuousPrimitives(b)
        rep = prim.move_until_contact([0,0,-1],max_dist=.003,f_threshold=.5)
        assert rep.termination=='contact'
        assert p.driver.start_gains[0]==[750,750,750,15,15,15]
    p = DemoPort()
    with p.backend() as b:
        prim = ContinuousPrimitives(b)
        rep = prim.call('search',mode='hop',radius_max=.001,pitch=.002,hop=.002,drop=.001)
        assert rep.termination=='no_surface'   # nothing under the tool is no surface, not a hole
        assert not p.driver.running


def test_real_torque_sign_and_tcp_com_conversion():
    p = DemoPort()
    with p.backend() as b:
        b.command_gripper(True,confirmed=True)
        # 0.6 kg, COM 2 cm +X and 1 cm -Y from TCP:
        # reaction moment = r x F; Z TCP translation does not change this.
        mg = .6*9.81
        p.driver.override['wrench']=(p.driver.bias+np.array([0,0,-mg,.01*mg,.02*mg,0])).tolist()
        b.calibration['torque_verified']=True
        prim = ContinuousPrimitives(b)
        rep = prim.hold(.5)
        com = next(q for q in rep.quantities if q['name']=='com_offset')
        assert com['extra']['x_cm']==2. and com['extra']['y_cm']==-1.
        assert rep.final_wrench[3]<0 and rep.final_wrench[4]<0


CASES = [
    ('move_to',dict(x=.403,y=0,z=.2466),None),
    ('move_by',dict(dx=.003,dy=0,dz=0),None),
    ('move_until_contact',dict(direction=[0,0,-1],max_dist=.006,f_threshold=.5),.348),
    ('push',dict(direction=[0,0,-1],target_force=1,duration=1.5),.35),
    ('slide',dict(direction=[1,0,0],distance=.003,normal_force=1),.35),
    ('search',dict(radius_max=.002,pitch=.002,drop=.002,mode='slide',normal_force=.5),None),
    ('insert',dict(depth=.003,wiggle=.0002),None),
    ('rotate_z',dict(angle_deg=1),None),
    ('lift',dict(height=.003,hold_time=.3),None),
    ('probe',dict(direction=[0,0,-1],amplitude=.003),None),
    ('grasp',{},None),
    ('release',{},None),
    ('hold',dict(seconds=.2),None),
    ('pull',dict(direction=[1,0,0],distance=.003,force=1,speed=.004),None),
    ('turn',dict(axis=[0,0,1],pivot=[.4,0,.2466],angle_deg=1),None),
    ('point_tool',dict(direction=[.02,0,-1]),None),
]


@pytest.mark.parametrize('name,args,plane',CASES,ids=[c[0] for c in CASES])
def test_every_primitive_returns_report_and_stops(name,args,plane):
    p = DemoPort(plane=plane)
    with p.backend() as b:
        prim = ContinuousPrimitives(b,confirm=lambda *_:True)
        if name in ('release','lift','pull','turn'):
            prim.grasp()
        rep = prim.call(name,**args)
        assert rep.primitive==name
        assert rep.n_steps>0 and rep.wrench_rows
        assert rep.truth['contact_available'] is False
        assert rep.truth['peak_object_contact_force'] is None
        assert not p.driver.running
        json.dumps(rep.to_dict(),allow_nan=False)
        if name=='lift': assert rep.derived['payload_estimate_kg']==.6
    assert p.driver.starts==p.driver.stops


def test_begin_waits_for_the_arm_to_settle():
    # after a controller stop under load the wrist can still swing at 0.1 rad/s and the robot side
    # refuses the next begin ('Start requires stationary arm'); the backend waits for it first.
    p = DemoPort()
    with p.backend() as b:
        p.driver.override = {'velocities':[0,0,0,0,0,.1,0]}
        b.refresh()
        naps = []
        def sleep(s):
            naps.append(s); p.clock.sleep(s)
            if len(naps)==4: p.driver.override = {}
        b.sleep = sleep
        with b.motion(): pass
        assert len(naps)==4 and p.driver.starts==1 and not b.fault
