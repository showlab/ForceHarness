"""An empty-downward-contact force calibration: it certifies only empty, downward contact finding and pressing,
never payloads, other directions or torque, and cannot be promoted to a full calibration."""
import numpy as np
import pytest

from forceharness.real.config import BackendConfig
from forceharness.real.backend import ContinuousBackend
from forceharness.real.plant import DemoPort
from forceharness.real.primitives import ContinuousPrimitives


EMPTY_DOWNWARD = dict(verified=False, force_sign=-1, force_scope='empty_downward_contact', scope_verified=True,
                      gripper_verified=False, torque_verified=False,
                      source='sign from empty-hand presses on a wooden table',
                      allowed_primitives=['push', 'slide', 'move_until_contact', 'probe'], allowed_normal=[0, 0, -1])


def scoped_fixture(plane=None):
    cfg=BackendConfig(motion_enabled=True,f_cap=10.)
    port=DemoPort(cfg,plane=plane)
    update=port.driver.update
    def negative_contact(*args):
        update(*args)
        port.driver.w[:3]=port.driver.bias[:3]-(port.driver.w[:3]-port.driver.bias[:3])
    port.driver.update=negative_contact
    c=dict(EMPTY_DOWNWARD)
    backend=ContinuousBackend(port,config=cfg,calibration=c,clock=port.clock,sleep=port.clock.sleep)
    return port,backend


@pytest.mark.parametrize('name',['push','slide','move_until_contact','probe'])
def test_scope_flips_only_force_inside_empty_downward_use(name):
    p,b=scoped_fixture()
    with b:
        b.rezero_bias(10,unloaded=True)
        p.driver.w=p.driver.bias+np.array([.1,-.2,-4.,.1,.2,.3]);b.refresh()
        w=b.wrench();torque=w.net_torque_world.copy()
        with pytest.raises(RuntimeError,match='calibration'):b.reaction_force(w)
        with b.force_calibration_scope(name,[0,0,-1]):
            np.testing.assert_allclose(b.reaction_force(b.wrench()),[-.1,.2,4.])
            np.testing.assert_allclose(b.wrench().net_torque_world,torque)
            assert not b.calibration_valid() and not b.wrench().bias_valid
        with pytest.raises(RuntimeError,match='calibration'):b.reaction_force(w)


@pytest.mark.parametrize('name,direction',[('push',[1,0,0]),('push',[0,0,1]),
    ('slide',[.1,0,-1]),('pull',[0,0,-1]),('lift',[0,0,-1]),('search',[0,0,-1])])
def test_scope_does_not_unlock_other_force_uses(name,direction):
    p,b=scoped_fixture()
    with b:
        b.rezero_bias(10,unloaded=True)
        with b.force_calibration_scope(name,direction):
            with pytest.raises(RuntimeError,match='calibration'):b.reaction_force(b.wrench())
    assert p.driver.starts==0


@pytest.mark.parametrize('closed,width',[(True,.05),(None,.08),(False,.06)])
def test_scope_rejects_held_unknown_or_not_open(closed,width):
    p,b=scoped_fixture()
    with b:
        b.rezero_bias(10,unloaded=True)
        b._closed=closed;p.driver.width=width;b.refresh()
        with b.force_calibration_scope('push',[0,0,-1]):
            with pytest.raises(RuntimeError,match='calibration'):b.reaction_force(b.wrench())


def test_scoped_push_and_slide_work_without_certifying_payload_or_torque():
    p,b=scoped_fixture(plane=.35)
    with b:
        prim=ContinuousPrimitives(b,stop_protocol='r9')
        for name,args in [('push',dict(direction=[0,0,-1],target_force=.5,duration=2.,lead=.008)),
                          ('slide',dict(direction=[1,0,0],distance=.003,normal_force=.5))]:
            rep=prim.call(name,**args)
            assert rep.derived['scoped_force_sign_calibrated']
            assert rep.quantities==[]
            assert not rep.derived['force_sign_calibrated']
        starts=p.driver.starts
        for name,args in [('push',dict(direction=[1,0,0],target_force=1.)),
                          ('slide',dict(direction=[1,0,0],distance=.003,normal=[0,1,0])),
                          ('pull',dict(direction=[1,0,0])),('lift',dict(height=.002))]:
            with pytest.raises(RuntimeError,match='calibration'):prim.call(name,**args)
        assert p.driver.starts==starts and not p.driver.grips
    assert p.driver.starts==p.driver.stops


def test_scope_cannot_be_promoted_to_full_or_torque_calibration():
    p,b=scoped_fixture()
    for key in ('verified','torque_verified','gripper_verified'):
        with pytest.raises(ValueError,match='scope'):
            ContinuousBackend(p,config=p.config,calibration=dict(b.calibration,**{key:True}))
