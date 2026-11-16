"""Fixed task goals are editable from the page: saved back to the prompt file the runner reads at start
(here the fake repository's copy, never the real forceharness/prompts/)."""
from pathlib import Path

from test_server_boundary import _post, server  # noqa: F401


def prompt_of(app, tid='screw_insert'):
    task = app.registry[tid]
    return task, Path(task['fixed_instruction_path'])


def test_saved_goal_is_written_back_and_used_by_the_next_launch(server):
    app, base = server
    task, path = prompt_of(app)
    assert str(path).startswith(str(app.cfg.harness_root))
    old = task['fixed_instruction']
    status, body = _post(base + '/api/tasks/instruction', dict(task='screw_insert', text='  New goal\nsecond line  ', base=old),
                         token=app.token)
    assert status == 200 and body['fixed_instruction'] == 'New goal\nsecond line'
    assert path.read_text(encoding='utf-8') == 'New goal\nsecond line\n'
    info = next(t for t in app.info()['tasks'] if t['id'] == 'screw_insert')
    assert info['fixed_instruction'] == 'New goal\nsecond line'
    assert info['fixed_instruction_file'] == 'forceharness/prompts/real/screw_insert.md'
    a, b = app.sessions.plan(dict(task='screw_insert', conditions=['A', 'B'], mode='demo', model=dict(source='external')))
    assert a.instruction == b.instruction == 'New goal\nsecond line'


def test_an_edit_made_elsewhere_is_shown_and_not_overwritten(server):
    app, base = server
    task, path = prompt_of(app)
    old = task['fixed_instruction']
    path.write_text('Edited elsewhere\n', encoding='utf-8')
    assert next(t for t in app.info()['tasks'] if t['id'] == 'screw_insert')['fixed_instruction'] == 'Edited elsewhere'
    status, body = _post(base + '/api/tasks/instruction', dict(task='screw_insert', text='Edited on the page', base=old),
                         token=app.token)
    assert status == 400 and 'changed elsewhere' in body['error'] and path.read_text(encoding='utf-8') == 'Edited elsewhere\n'


def test_refused_saves_leave_the_file_alone(server):
    app, base = server
    task, path = prompt_of(app)
    old = task['fixed_instruction']
    before = path.read_text(encoding='utf-8')
    post = lambda **kw: _post(base + '/api/tasks/instruction', dict(task='screw_insert', base=old, **kw), token=app.token)
    assert post(text='   ')[0] == 400
    assert post(text='x' * 4001)[0] == 400
    assert _post(base + '/api/tasks/instruction', dict(task='free', text='x', base=''), token=app.token)[0] == 400
    assert _post(base + '/api/tasks/instruction', dict(task='screw_insert', text='x', base=old))[0] == 403
    # the second half of a pair reads the file when it starts: no edit while it waits
    second = app.sessions.plan(dict(task='screw_insert', conditions=['A', 'B'], mode='demo', model=dict(source='external')))[1]
    app.sessions.sessions[second.id] = second
    status, body = post(text='New goal')
    assert status == 400 and 'queued' in body['error']
    assert path.read_text(encoding='utf-8') == before
