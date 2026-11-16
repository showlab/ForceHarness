"""Resume launch contract: "Resume from the last episode" passes --resume-held to live runners only, as one argv item."""
import pytest

from fhmon.sessions import LaunchError
from test_server_boundary import server  # noqa: F401

LIVE = dict(task='blackboard_wipe', conditions=['A'], mode='live', model=dict(source='manual'), checklist=[True]*3)


def test_resume_held_reaches_the_live_runner_as_one_argument(server):
    app, _ = server
    app.cfg.allow_live = True
    task = next(t for t in app.info()['tasks'] if t['id'] == 'blackboard_wipe')
    assert task['live']['resume_held']['default'] == 'eraser'
    plain = app.sessions.plan(LIVE)[0]
    assert not any(a.startswith('--resume-held') for a in plain.command) and plain.resume_held is None
    s = app.sessions.plan(dict(LIVE, resume_held='  eraser, blue side down '))[0]
    assert '--resume-held=eraser, blue side down' in s.command and s.public()['resume_held'] == 'eraser, blue side down'
    for tid in ('free', 'screw_insert', 'usb_insert', 'peel_vegetable'):
        assert app.registry[tid]['live']['resume_held']['args'] == ['--resume-held={resume_held}']


def test_resume_held_is_refused_outside_live_or_when_too_long(server):
    app, _ = server
    with pytest.raises(LaunchError, match='Resume from the last episode'):
        app.sessions.plan(dict(LIVE, mode='demo', checklist=None, resume_held='eraser'))
    app.cfg.allow_live = True
    with pytest.raises(LaunchError, match='one line'):
        app.sessions.plan(dict(LIVE, resume_held='eraser\n--live'))
    with pytest.raises(LaunchError, match='40'):
        app.sessions.plan(dict(LIVE, resume_held='e'*41))
    for tid in ('touch_firmness', 'can_weight_sort'):                     # these tasks start with an empty gripper
        with pytest.raises(LaunchError, match='Resume from the last episode'):
            app.sessions.plan(dict(LIVE, task=tid, resume_held='bottle'))
