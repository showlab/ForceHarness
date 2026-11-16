"""The six fixed real-robot tasks: instruction, menus and prompts per condition, and honest offline demos.

Offline only: the synthetic plant cannot model any of the tasks, so every demo ends with done(success=false).
"""
import importlib.util
import json
import re

import pytest

from forceharness.real import runner as free
from forceharness.real.tasks import TASKS
from real_helpers import FORCE_ONLY, PROMPTS, ROOT, run_task, task_file

LIVE_ARGS = {'blackboard_wipe': (), 'screw_insert': ('--max-actions', '60', '--max-minutes', '60'),
             'peel_vegetable': ('--max-actions', '60', '--max-minutes', '60', '--force-max', '20',
                                '--slide-force-default', '10'),
             'usb_insert': ('--force-max', '20', '--insert-force-max', '20'),
             'can_weight_sort': ('--max-actions', '80', '--max-minutes', '60'), 'touch_firmness': ('--max-actions', '15')}
VERSIONS = dict(blackboard_wipe='r40', screw_insert='screw1', peel_vegetable='peel2', usb_insert='plug1',
                can_weight_sort='cansort1', touch_firmness='firm1')


def run(tmp_path, task_id, *args):
    code = run_task(task_id, ['--out', str(tmp_path/'ep'), '--relay-dir', str(tmp_path/'relay'), *args])
    return code, json.loads((tmp_path/'ep'/'episode.json').read_text()), (tmp_path/'relay'/'obs_0.md').read_text()


def tool_calls(text):
    """Names of force-only tools the text refers to as tools (a call, a backticked name, or 'use <tool>')."""
    return {name for name in FORCE_ONLY | {'screw', 'wiggle'}
            if re.search(rf'(`{name}`|\b{name}\(|\buse {name}\b|\bwith {name}\b)', text)}


def test_the_six_tasks_and_their_live_flags():
    assert set(TASKS) == set(LIVE_ARGS)
    for task_id, task in TASKS.items():
        assert task.live_args == LIVE_ARGS[task_id] and task.version == VERSIONS[task_id]
        assert task_file(task_id).is_file() and task.instruction() == task_file(task_id).read_text().strip()


@pytest.mark.parametrize('task_id', sorted(LIVE_ARGS))
@pytest.mark.parametrize('condition', ['A', 'B'])
def test_demo_keeps_the_instruction_and_claims_nothing(tmp_path, task_id, condition):
    code, r, obs = run(tmp_path, task_id, '--condition', condition, *LIVE_ARGS[task_id])
    assert code == 0 and r['task_id'] == task_id and r['task_version'] == VERSIONS[task_id]
    assert r['instruction'] == task_file(task_id).read_text().strip() and r['instruction'] in obs
    assert r['model_success'] is False and not r['scored'] and not r['physical_validation']
    assert [a['name'] for a in r['actions']] == ['done'] and r['scene_updates'][0]['objects'] == []
    assert r['model_note'].startswith('Interface check only: the synthetic plant has no physical model')
    assert '{menu}' not in obs and '{table_z}' not in obs and '{instruction}' not in obs
    menu = r['menu']
    for name in menu:
        assert name+'(' in obs
    assert not FORCE_ONLY & set(menu) or condition == 'A' or task_id == 'peel_vegetable'
    assert 'runner_fixes' not in r and 'execution_version' not in r


@pytest.mark.parametrize('task_id', sorted(LIVE_ARGS))
def test_a_fixed_task_cannot_be_replaced(tmp_path, task_id):
    other = tmp_path/'other.txt'
    other.write_text('another task')
    for flag, value in (('--instruction', 'another task'), ('--instruction-file', str(other))):
        with pytest.raises(SystemExit):
            run_task(task_id, ['--out', str(tmp_path/'ep'), '--relay-dir', str(tmp_path/'relay'), flag, value])
        assert not (tmp_path/'ep').exists()


def test_operator_scoring_is_separate_from_the_model_report(tmp_path):
    code, _, _ = run(tmp_path, 'blackboard_wipe')
    assert run_task('blackboard_wipe', ['score', '--out', str(tmp_path/'ep'), '--outcome', 'partial',
                                        '--note', 'residue left']) == 0
    r = json.loads((tmp_path/'ep'/'episode.json').read_text())
    assert r['scored'] and r['success'] is False and r['outcome'] == 'partial' and r['score_note'] == 'residue left'
    with pytest.raises(SystemExit):
        run_task('blackboard_wipe', ['score', '--out', str(tmp_path/'ep'), '--outcome', 'success'])


@pytest.mark.parametrize('task_id', ['peel_vegetable', 'screw_insert', 'can_weight_sort', 'touch_firmness'])
def test_instructions_that_name_no_force_only_tool(task_id):
    assert not tool_calls(task_file(task_id).read_text())


def test_instructions_that_name_tools_as_procedural_hints():
    # The operator's procedural hints in two instructions name tools of the force menu.
    assert tool_calls(task_file('usb_insert').read_text()) == {'search', 'insert', 'wiggle'}
    assert 'slide' in task_file('blackboard_wipe').read_text()


def test_wipe_instruction_is_independent_of_the_scene():
    text = task_file('blackboard_wipe').read_text()
    assert not any(word in text for word in ('left side', 'right side', 'red', 'blue'))
    assert 'more pressure' in text and 'really covers the residue' in text


def test_touch_instruction_sets_the_budget_and_the_output_but_no_method():
    goal = task_file('touch_firmness').read_text()
    assert 'turn 14' in goal and 'left soft, right hard' in goal
    assert not any(word in goal for word in ('push', 'move_until_contact', '3 N', 'sink', 'press', 'guard', 'pad'))


def test_usb_instruction_does_not_fix_the_start():
    goal = task_file('usb_insert').read_text()
    assert 'usually already in the hand' in goal and 'not preset' in goal and 'if it is not in the hand' in goal


def test_screw_instruction_and_menus():
    goal = task_file('screw_insert').read_text()
    assert 'rotate_z' in goal and 'no limit on the number of grasps' in goal and 'shift=0' in goal
    conditions, _ = TASKS['screw_insert'].menus()
    assert 'lift' not in conditions['A']['menu'] and 'screw' in conditions['A']['menu']
    assert conditions['A']['menu'][-1] == 'wiggle' and 'wiggle' not in conditions['B']['menu']
    assert 'lift' in free.CONDITIONS['A']['menu'] and 'wiggle' not in free.CONDITIONS['A']['menu']


def test_plug_search_has_the_slide_mode_only(tmp_path):
    code, r, obs = run(tmp_path, 'usb_insert')
    line = next(l for l in obs.splitlines() if l.startswith('- `search('))
    assert code == 0 and line.startswith('- `search(normal?, radius_max?, pitch?, normal_force?, drop?)`')
    assert 'mode=' not in line and 'hop' not in line and 'guess' not in line
    _, primitives = TASKS['usb_insert'].menus()
    assert set(primitives['search']['args']) == {'normal', 'radius_max', 'pitch', 'normal_force', 'drop'}
    assert 'hop' in free.PRIMITIVES['search']['args']            # the module-level menu keeps the hop mode


@pytest.mark.parametrize('condition', ['A', 'B'])
def test_peel_menus_and_prompts_are_scoped_to_the_task(tmp_path, condition):
    original = {c: list(v['menu']) for c, v in free.CONDITIONS.items()}
    code, r, obs = run(tmp_path, 'peel_vegetable', '--condition', condition)
    assert code == 0
    menu = r['menu']
    assert menu == original[condition]+(['move_until_contact'] if condition == 'B' else [])
    assert ('slide' in menu) == (condition == 'A') and ('lift(' in obs) == (condition == 'A')
    assert 'You are controlling a Franka FR3 to peel (condition' in obs and r['instruction'] in obs
    assert {c: list(v['menu']) for c, v in free.CONDITIONS.items()} == original


def test_peel_records_the_raised_force_limit(tmp_path):
    code, r, _ = run(tmp_path, 'peel_vegetable', *LIVE_ARGS['peel_vegetable'])
    assert code == 0 and r['max_actions'] == 60 and r['max_minutes'] == 60
    assert r['force_limits_N'] == dict(contact=20., insert=free.INSERT_FORCE_MAX_N, slide_default=10.)


def test_the_control_can_find_contact_in_peeling_without_seeing_force(tmp_path, monkeypatch):
    from test_free_task import FakeClient
    FakeClient.script = [('move_until_contact', dict(direction=[0, 0, -1], max_dist=.02, f_threshold=1)),
                         ('done', dict(success=False, note='x'))]
    monkeypatch.setattr(free, 'RelayClient', FakeClient)
    code = run_task('peel_vegetable', ['--out', str(tmp_path/'ep'), '--relay-dir', str(tmp_path/'relay'),
                                       '--condition', 'B', '--demo-model'])
    record = json.loads((tmp_path/'ep/episode.json').read_text())
    assert code == 0 and 'move_until_contact' in record['menu'] and 'slide' not in record['menu']
    first = record['actions'][0]
    assert first['name'] == 'move_until_contact' and first['termination']
    text = first['receipt']['text']
    assert not text.startswith('Invalid') and not re.search(r'\d N\b', text) and 'force' not in text.lower()
    assert "'f_threshold': 5.0" in text          # the model's threshold is ignored: B cannot tune a force
    prompt = json.dumps(FakeClient.last.seen[0])
    assert 'gives no force reading' in prompt and '`move_until_contact(direction, max_dist)`' in prompt


def test_both_conditions_know_how_to_press_harder():
    force = (PROMPTS/'free_task_force.md').read_text()
    baseline = (PROMPTS/'free_task_baseline.md').read_text()
    assert '+2 N each time' in force and 'at most 10 N' in force
    assert 'lower the contact height by about 1 mm each time' in baseline and 'normal_force' not in baseline


def test_slide_force_range_is_unchanged():
    b, _, _ = free.demo()
    with b:
        assert free.validate('slide', dict(direction=[0, 1, 0], distance=.05, normal_force=10), b, 'A')[0]['normal_force'] == 10
        with pytest.raises(ValueError, match='normal_force'):
            free.validate('slide', dict(direction=[0, 1, 0], distance=.05, normal_force=12), b, 'A')


@pytest.mark.parametrize('name', sorted(LIVE_ARGS)+['free_task'])
def test_the_command_line_scripts_run_the_presets(tmp_path, name):
    spec = importlib.util.spec_from_file_location(f'script_{name}', ROOT/f'scripts/real/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = ['--out', str(tmp_path/'ep'), '--relay-dir', str(tmp_path/'relay')]
    if name == 'free_task':
        argv += ['--instruction', 'look at the table', '--max-actions', '1']
    assert module.main(argv) == (0 if name != 'free_task' else 2)
    r = json.loads((tmp_path/'ep'/'episode.json').read_text())
    assert r['task_id'] == ('free' if name == 'free_task' else name)
