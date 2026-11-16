"""The wrench residual stops the arm on a 1 N*m rise within one primitive or above 3 N*m, not on the pose-dependent
offset a long wrist turn builds up."""
import pytest

from forceharness.real.plant import DemoPort


def test_high_residual_at_primitive_start_is_accepted():
    # a wrist at q7 = 2.03 rad rested with residual 0.41 and climbed to 1.0 while turning a screw;
    # an absolute 1.0 N*m limit would have ended the episode.
    p = DemoPort()
    p.driver.override['residual'] = 1.8
    p.session.begin()
    p.driver.override['residual'] = 2.6
    p.session.tick()
    assert p.driver.running and not p.session.fault


def test_rise_within_primitive_still_stops():
    p = DemoPort()
    p.driver.override['residual'] = 1.2
    p.session.begin()
    p.driver.override['residual'] = 2.3
    with pytest.raises(RuntimeError, match='rose'):
        p.session.tick()
    assert not p.driver.running and p.session.fault


def test_absolute_ceiling_still_stops():
    p = DemoPort()
    p.driver.override['residual'] = 2.5
    p.session.begin()
    p.driver.override['residual'] = 3.1
    with pytest.raises(RuntimeError, match='conditioning'):
        p.session.tick()
    assert not p.driver.running


def test_each_primitive_takes_a_new_baseline():
    p = DemoPort()
    p.driver.override['residual'] = .05
    p.session.begin()
    p.driver.override['residual'] = .9
    p.session.tick()
    p.session.end()
    assert p.session.initial_residual is None
    p.driver.override['residual'] = 1.6
    p.session.begin()
    assert p.session.initial_residual == pytest.approx(1.6)
    p.session.tick()
    assert p.driver.running
