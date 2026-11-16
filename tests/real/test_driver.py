"""Driver target validation with a fake robot: no RPCs or hardware imports."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
torch=pytest.importorskip('torch')
from real_helpers import robot_side
DeadlineStub,PolymetisDriver=robot_side('driver').DeadlineStub,robot_side('driver').PolymetisDriver


def test_only_controller_start_gets_prewarm_budget_and_explicit_timeouts_survive():
    calls=[]
    names=('SetController','UpdateController','TerminateController',
           'GetRobotState','GetRobotClientMetadata','GetEpisodeInterval','GetState')
    def rpc(request,**kwargs):
        calls.append((request,kwargs))
        return 'response'
    stub=DeadlineStub(SimpleNamespace(**{name:rpc for name in names}))
    request=object()
    for name in names:
        expected=3. if name=='SetController' else .10
        assert getattr(stub,name)(request)=='response'
        assert calls[-1]==(request,{'timeout':expected}),name
        assert getattr(stub,name)(request,timeout=.025,metadata=(('test','value'),))=='response'
        assert calls[-1]==(request,{'timeout':.025,'metadata':(('test','value'),)}),name


class Model:
    def __init__(self):
        self.bad=None
        self.calls=0
    def inverse_kinematics(self,p,r,**kwargs):
        self.calls+=1
        q=torch.zeros(7);q[:3]=p-torch.tensor([.4,0.,.35])
        if self.bad is not None: q=self.bad.clone()
        return q
    def forward_kinematics(self,q):
        return torch.tensor([.4,0.,.35])+q[:3],torch.tensor([0.,0.,0.,1.])


def driver():
    d=PolymetisDriver.__new__(PolymetisDriver)
    d.torch=torch
    d.low=np.ones(7)*-2
    d.high=np.ones(7)*2
    d.start_q=d.last_q=d.last_joint_target=np.zeros(7)
    d.last_flange=np.array([.4,0,.35])
    d.last_target=None
    d.last_update_index=None
    d.own_start=99
    d.kp_max=np.array([750.]*3+[15.]*3)
    d.kd_max=np.array([37.]*3+[2.]*3)
    d.active_kp,d.active_kd=d.kp_max.copy(),d.kd_max.copy()
    d.start_pose=np.array([.4,0,.35,0,0,0,1.])
    d.telemetry=lambda row:None
    d.sent=[]
    d.robot=SimpleNamespace(robot_model=Model(),update_desired_joint_positions=lambda p:d.sent.append(p) or 0)
    return d


def send(d):
    d.update(np.array([.404,0,.35,0,0,0,1]),np.array([750]*3+[15]*3),np.array([37]*3+[2]*3),1)


def test_ik_uses_stock_joint_target_api_and_unchanged_target_is_not_resent():
    d=driver()
    send(d);send(d)
    assert d.robot.robot_model.calls==1
    assert len(d.sent)==1
    assert d.sent[0].shape==(7,)
    assert d.sent[0][0].item()==pytest.approx(.004,abs=1e-7)
    torch.testing.assert_close(d.sent[0][1:],torch.zeros(6))


@pytest.mark.parametrize('bad',[torch.ones(7)*float('nan'),torch.ones(7)*3,
                                torch.tensor([.004,0,0,0,0,0,.06]),torch.zeros(7)])
def test_unsafe_or_wrong_ik_never_sends(bad):
    d=driver();d.robot.robot_model.bad=bad
    with pytest.raises(RuntimeError,match='IK'):
        send(d)
    assert not d.sent and d.last_target is None


def test_ik_deadline_never_sends(monkeypatch):
    driver_module = robot_side("driver")
    times=iter([1.,1.11])
    monkeypatch.setattr(driver_module.time,'monotonic',lambda:next(times))
    d=driver()
    with pytest.raises(RuntimeError,match='100 ms'):
        send(d)
    assert not d.sent


def test_rejected_update_does_not_change_last_accepted_target():
    d=driver()
    logs=[]
    d.telemetry=logs.append
    d.robot.update_desired_joint_positions=lambda params:-1
    with pytest.raises(RuntimeError,match='rejected'):
        send(d)
    assert d.last_target is None
    np.testing.assert_array_equal(d.last_joint_target,np.zeros(7))
    assert d.last_update_index is None
    assert not any(row['type']=='joint_target_accepted' for row in logs)


def test_changed_gains_rejected_before_ik_or_rpc():
    d=driver()
    with pytest.raises(RuntimeError,match='gains are fixed'):
        d.update(d.start_pose,d.kp_max*.5,d.kd_max*np.sqrt(.5),1)
    assert not d.sent and d.robot.robot_model.calls==0


def test_resume_rebases_only_primitive_bounds_without_rpc_or_target_change():
    d=driver();send(d)
    target=d.last_target.copy();joints=d.last_joint_target.copy();gains=d.active_kp.copy()
    d.reset_primitive(dict(flange_pos=[.45,0,.35],flange_quat=[0,0,0,1],joints=[.05,0,0,0,0,0,0]))
    np.testing.assert_array_equal(d.start_pose,[.45,0,.35,0,0,0,1])
    np.testing.assert_array_equal(d.start_q,[.05,0,0,0,0,0,0])
    np.testing.assert_array_equal(d.last_target,target)
    np.testing.assert_array_equal(d.last_joint_target,joints)
    np.testing.assert_array_equal(d.active_kp,gains)
    assert d.own_start==99 and len(d.sent)==1


@pytest.mark.parametrize('index,value',[
    (0,float('nan')),(0,.7),(1,.31),(2,.15),(6,1.01),
])
def test_invalid_pose_never_reaches_update_rpc(index,value):
    d=driver()
    pose=d.start_pose.copy();pose[index]=value
    with pytest.raises((ValueError,RuntimeError)):
        d.update(pose,d.kp_max,d.kd_max,1)
    assert not d.sent


def test_factory_returns_installed_class_with_selected_gains(monkeypatch):
    d=driver()
    class InstalledPolicy:
        def __init__(self,**kwargs): self.arguments=kwargs
    monkeypatch.setitem(sys.modules,'torchcontrol',SimpleNamespace(
        policies=SimpleNamespace(HybridJointImpedanceControl=InstalledPolicy)))
    d.robot.use_grav_comp=True
    d.robot.metadata=SimpleNamespace(default_Kq=[40,30,50,25,35,25,10],
                                    default_Kqd=[4,6,5,5,3,2,1])
    d.robot.get_joint_positions=lambda:torch.zeros(7)
    policy=d.make_policy(d.kp_max*.5,d.kd_max*np.sqrt(.5))
    assert type(policy) is InstalledPolicy
    arguments=policy.arguments
    torch.testing.assert_close(arguments['Kx'],torch.tensor(d.kp_max*.5,dtype=torch.float32))
    torch.testing.assert_close(arguments['Kq'],torch.tensor([80.,60.,100.,50.,70.,50.,40.]))
    torch.testing.assert_close(arguments['Kqd'],torch.tensor(
        np.array([4,6,5,5,3,2,1])*np.sqrt(np.array([4,4,4,4,4,4,8])*.5),dtype=torch.float32))
    assert arguments['robot_model'] is d.robot.robot_model
    assert arguments['ignore_gravity'] is True


@pytest.mark.parametrize('field,value',[('kp',0.),('kp',751.),('kd',float('nan')),('kd',38.)])
def test_invalid_gains_rejected_before_any_rpc(field,value):
    d=driver()
    kp,kd=d.kp_max.copy(),d.kd_max.copy()
    (kp if field=='kp' else kd)[0]=value
    with pytest.raises((ValueError,RuntimeError)):
        d.update(d.start_pose,kp,kd,1)
    assert not d.sent


def test_does_not_stop_a_foreign_controller():
    d=driver()
    d.Empty=lambda:None
    d.own_start=99
    d.start_attempted=True
    d.robot.grpc_connection=SimpleNamespace(
        GetEpisodeInterval=lambda _:SimpleNamespace(start=100,end=-1))
    d.robot.terminate_current_policy=lambda **kwargs:pytest.fail('Must not stop another owner')
    with pytest.raises(RuntimeError,match='foreign'):
        d.stop()


def test_target_diagnostics_record_exact_rpc_payload_once():
    d=driver()
    logs=[]
    d.telemetry=logs.append
    def update(q):
        d.sent.append(q)
        return 123
    d.robot.update_desired_joint_positions=update
    send(d);send(d)
    accepted=[row for row in logs if row['type']=='joint_target_accepted']
    assert len(accepted)==1
    row=accepted[0]
    assert row['update_index']==123 and row['episode_start']==99
    assert row['joint_target']==d.sent[0].tolist()==d.last_joint_target.tolist()
    assert row['checked_joint_state']==[0.]*7
    np.testing.assert_allclose(row['fk_pos'],row['flange_target'][:3],atol=1e-7)


@pytest.mark.parametrize('running',[True,False])
def test_state_diagnostics_reuse_same_read_and_label_inactive_target(running,monkeypatch):
    driver_module = robot_side("driver")
    d=driver()
    d.Empty=lambda:None
    d.last_update_index=123
    names=('joint_torques_computed','prev_joint_torques_computed',
           'prev_joint_torques_computed_safened','motor_torques_measured',
           'motor_torques_external','motor_torques_desired')
    stamp=SimpleNamespace(seconds=20,nanos=123)
    s=SimpleNamespace(timestamp=stamp,joint_positions=[0.]*7,joint_velocities=[0.]*7,
                      error_code=0,prev_controller_latency_ms=.3,prev_command_successful=True,
                      **{name:[float(i)]*7 for i,name in enumerate(names)})
    calls=[]
    def read():
        calls.append('state');return s
    def interval(_):
        calls.append('interval');return SimpleNamespace(start=99,end=-1 if running else 150)
    def grip(_):
        calls.append('gripper');return SimpleNamespace(error_code=0,timestamp=stamp,width=.08)
    d.robot.get_robot_state=read
    d.robot.grpc_connection=SimpleNamespace(GetEpisodeInterval=interval)
    d.gripper=SimpleNamespace(GetState=grip)
    d.kin=SimpleNamespace(fk=lambda q:(np.array([.4,0,.35]),np.array([0,0,0,1.])),
                          jacobian=lambda q:np.eye(6,7))
    monkeypatch.setattr(driver_module,'solve_wrench',
                        lambda j,t:(np.zeros(6),0.,1.))
    row=d.read()
    assert calls==['state','interval','gripper']
    for name in names:
        assert row[name]==getattr(s,name)
    assert row['timestamp_ns']==20_000_000_123
    assert row['rpc_accepted_joint_target']==[0.]*7
    assert row['last_update_index']==123
    assert row['target_controller_active'] is running


def test_lead2_measured_joint_check_grows_with_target_distance_only():
    kp,kd=np.array([750]*3+[15]*3),np.array([37]*3+[2]*3)
    d=driver();d.robot.robot_model.bad=torch.tensor([.014,0,0,.07,0,0,0.])
    d.last_joint_target=np.array([.014,0,0,.04,0,0,0])
    d.update(np.array([.414,0,.35,0,0,0,1]),kp,kd,1)      # 14 mm ahead: 0.05*14/8 allows 0.07 rad
    assert len(d.sent)==1
    d=driver();d.robot.robot_model.bad=torch.tensor([.004,0,0,.07,0,0,0.])
    d.last_joint_target=np.array([.004,0,0,.04,0,0,0])
    with pytest.raises(RuntimeError,match='IK'):            # 4 mm ahead: still 0.05 rad
        send(d)
    d=driver();d.robot.robot_model.bad=torch.tensor([.014,0,0,.07,0,0,0.])
    with pytest.raises(RuntimeError,match='IK'):            # the jump from the last joint target stays 0.05
        d.update(np.array([.414,0,.35,0,0,0,1]),kp,kd,1)
