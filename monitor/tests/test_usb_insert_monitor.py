"""USB insertion launch contract and separate records, without starting a robot episode."""
import json

from conftest import make_free
from fhmon import episodes, live
from test_server_boundary import server  # noqa: F401


def test_usb_task_launches_with_its_fixed_goal(server):
    app, _ = server
    task = next(t for t in app.info()['tasks'] if t['id'] == 'usb_insert')
    assert task['name'] == 'Insert the USB plug into the power strip' and task['live']['resume_held']['default'] == 'USB plug'
    a, b = app.sessions.plan(dict(task='usb_insert', conditions=['A', 'B'], mode='demo', model=dict(source='external')))
    assert a.instruction == b.instruction == task['fixed_instruction'] and a.command[1].endswith('scripts/real/usb_insert.py')
    app.cfg.allow_live = True
    s = app.sessions.plan(dict(task='usb_insert', conditions=['A'], mode='live', model=dict(source='manual'),
                               checklist=[True] * 3, resume_held='USB plug'))[0]
    assert '--live' in s.command and '--resume-held=USB plug' in s.command


def test_usb_records_stay_apart(runs):
    path = make_free(runs, 'usb_A', task='Insert the plug', scored='partial')/'episode.json'
    record = json.loads(path.read_text())
    record.update(task_id='usb_insert')
    path.write_text(json.dumps(record, ensure_ascii=False))
    assert episodes.adapter_for(record).family == 'usb_insert'
    rows = live.Store([runs]).scan()
    assert [e['family'] for e in live.compare(rows, 'usb_insert')['episodes']] == ['usb_insert']
    assert not live.compare(rows, 'free')['episodes']
