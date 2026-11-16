"""Peeling launch contract and the per-task force flags, without starting a robot episode."""
import json

from conftest import make_free
from fhmon import episodes
from test_server_boundary import server  # noqa: F401


def flag(command, name):
    return command[command.index(name)+1] if name in command else None


def test_peel_and_usb_tasks_ask_for_twenty_newtons(server):
    app, _ = server
    app.cfg.allow_live = True
    live = dict(conditions=['A'], mode='live', model=dict(source='manual'), checklist=[True] * 3)
    peel = app.sessions.plan(dict(live, task='peel_vegetable', resume_held='peeler'))[0]
    assert peel.command[1].endswith('scripts/real/peel_vegetable.py') and '--resume-held=peeler' in peel.command
    assert flag(peel.command, '--force-max') == '20' and flag(peel.command, '--insert-force-max') is None
    assert flag(peel.command, '--slide-force-default') == '10'
    usb = app.sessions.plan(dict(live, task='usb_insert'))[0]
    assert flag(usb.command, '--force-max') == '20' and flag(usb.command, '--insert-force-max') == '20'
    for tid in ('blackboard_wipe', 'screw_insert', 'can_weight_sort', 'touch_firmness', 'free'):   # these keep the runner defaults
        s = app.sessions.plan(dict(live, task=tid, instruction='x' if tid == 'free' else ''))[0]
        assert flag(s.command, '--force-max') is None


def test_peel_records_have_their_own_family(runs):
    path = make_free(runs, 'peel_A', task='Peel it', scored='partial')/'episode.json'
    record = json.loads(path.read_text())
    record.update(task_id='peel_vegetable')
    assert episodes.adapter_for(record).family == 'peel_vegetable'
