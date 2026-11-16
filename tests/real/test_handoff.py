"""Stock termination and fixed gains at controller-session boundaries."""
import pytest
from forceharness.real.plant import DemoPort


def test_normal_completion_retracts_target_before_stock_stop():
    events=[]
    p=DemoPort(emit=events.append)
    with p.backend() as b:
        with b.motion(3.):
            b.step([.001,0,0])
    assert p.driver.stops==1
    stop=next(r for r in events if r['type']=='normal_stop')
    assert stop['mode']=='target_release_then_stock_terminate'
    assert stop['release_elapsed_s']==pytest.approx(.3)
    assert sum(r['type']=='release_start' for r in events)==1


@pytest.mark.parametrize('failure',['force','disconnect'])
def test_normal_stop_guard_failure_still_attempts_termination(failure):
    p=DemoPort()
    p.session.begin(3.)
    if failure=='force': p.driver.override['wrench']=[25.,0,0,0,0,0]
    with pytest.raises(RuntimeError):
        p.session.finish(sleep=p.clock.sleep,cancel=lambda:failure=='disconnect')
    assert p.session.fault and not p.driver.running
    assert p.driver.stops==1


def test_exception_stops_stock_controller():
    p=DemoPort()
    with p.backend() as b:
        with pytest.raises(KeyboardInterrupt):
            with b.motion(3.):
                raise KeyboardInterrupt
    assert p.driver.stops==1


def test_gain_changes_require_separate_sessions():
    p=DemoPort()
    with p.backend() as b:
        for kp in (150,75,40,150):
            b.set_kp(kp)
            with b.motion(3.):
                b.step([0,0,0])
    assert p.driver.starts==4 and p.driver.stops==4
    assert [g[0] for g in p.driver.start_gains]==[750,375,200,750]


def test_direct_active_gain_change_stops_without_silent_application():
    p=DemoPort()
    with p.backend() as b:
        with pytest.raises(RuntimeError,match='gains are fixed'):
            with b.motion(3.):
                b.set_kp(75)
    assert p.driver.stops==1 and not p.driver.running


def test_primitive_restoration_is_deferred_until_after_stop():
    p=DemoPort()
    with p.backend() as b:
        with b.fixed_gains(40):
            with b.motion(3.):
                b.step([0,0,0])
                reply=b.set_kp(150)
                assert reply['applied'] is False
                assert p.session.kp_ratio[0]==40/150
        assert p.session.kp_ratio[0]==1.
    assert p.driver.starts==1 and p.driver.stops==1


def test_deferred_gain_change_cannot_hide_a_mid_primitive_change():
    p=DemoPort()
    with p.backend() as b:
        with pytest.raises(RuntimeError,match='requires a new controller'):
            with b.fixed_gains(40):
                with b.motion(3.):
                    b.set_kp(150)
                    b.step([0,0,0])
    assert p.driver.stops==1 and not p.driver.running
