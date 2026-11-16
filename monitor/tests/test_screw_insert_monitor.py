"""Screwing launch contract and separate records, without starting a robot episode."""
import json

from conftest import make_free
from fhmon import episodes, live
from test_server_boundary import server  # noqa: F401


def test_screw_task_launches_with_its_fixed_goal_and_longer_limits(server):
    app, _ = server
    task = next(t for t in app.info()['tasks'] if t['id'] == 'screw_insert')
    assert task['name'] == 'Screw the green screw into the purple hole' and task['fixed_instruction'].startswith('Goal of screw_insert')
    assert task['max_actions'] == 60 and task['live']['resume_held']['default'] == 'green screw'
    a, b = app.sessions.plan(dict(task='screw_insert', conditions=['A', 'B'], mode='demo', model=dict(source='external')))
    assert a.instruction == b.instruction == task['fixed_instruction']
    assert a.command[1].endswith('scripts/real/screw_insert.py') and a.command[a.command.index('--max-actions') + 1] == '60'
    app.cfg.allow_live = True
    s = app.sessions.plan(dict(task='screw_insert', conditions=['A'], mode='live', model=dict(source='manual'),
                               checklist=[True] * 3, resume_held='green screw'))[0]
    assert '--live' in s.command and '--resume-held=green screw' in s.command and '--max-minutes' in s.command


def test_screw_records_are_not_mixed_with_other_tasks(runs):
    for name, condition, outcome in [('screw_A', 'phys2_force', 'partial'), ('screw_B', 'none_guard', 'failed')]:
        path = make_free(runs, name, condition=condition, task='Screw it in', scored=outcome)/'episode.json'
        record = json.loads(path.read_text())
        record.update(task_id='screw_insert')
        path.write_text(json.dumps(record, ensure_ascii=False))
        assert episodes.adapter_for(record).family == 'screw_insert'
    rows = live.Store([runs]).scan()
    assert len(live.compare(rows, 'screw_insert')['episodes']) == 2
    assert not live.compare(rows, 'usb_insert')['episodes'] and not live.compare(rows, 'free')['episodes']
