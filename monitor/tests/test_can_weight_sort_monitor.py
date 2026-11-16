"""Weight-sort launch contract and separate records, without starting a robot episode."""
import json

import pytest

from conftest import make_free
from fhmon import episodes, live
from fhmon.sessions import LaunchError
from test_server_boundary import server  # noqa: F401


def test_fixed_goal_is_identical_for_both_conditions_and_live_gets_the_longer_budget(server):
    app, _ = server
    task = next(t for t in app.info()['tasks'] if t['id'] == 'can_weight_sort')
    assert task['name'] == 'Sort bottles by weight' and task['max_actions'] == 80
    assert task['fixed_instruction_file'] == 'forceharness/prompts/real/can_weight_sort.md'
    assert task['fixed_instruction'].startswith('Goal of can_weight_sort') and 'resume_held' not in task['live']
    req = dict(task='can_weight_sort', conditions=['A', 'B'], mode='demo', model=dict(source='external'))
    a, b = app.sessions.plan(req)
    assert a.instruction == b.instruction == task['fixed_instruction']
    assert a.command[1].endswith('scripts/real/can_weight_sort.py') and a.command[1] == b.command[1]
    assert '--live' not in a.command and '--execute' not in a.command
    with pytest.raises(LaunchError, match='is fixed in the task file'):
        app.sessions.plan(dict(req, instruction='a different task'))
    with pytest.raises(LaunchError, match='real-robot mode is off'):
        app.sessions.plan(dict(req, conditions=['A'], mode='live'))
    app.cfg.allow_live = True
    real = dict(req, conditions=['A'], mode='live', model=dict(source='manual'))
    with pytest.raises(LaunchError, match='checklist'):
        app.sessions.plan(real)
    planned = app.sessions.plan(dict(real, checklist=[True]*3))[0].command
    assert '--live' in planned and '--auto-confirm' in planned
    assert planned[planned.index('--max-actions')+1] == '80' and planned[planned.index('--max-minutes')+1] == '60'


def test_results_are_not_mixed_with_free_tasks_or_other_fixed_tasks(runs):
    for name, condition, outcome in [('sort_A', 'phys2_force', 'success'), ('sort_B', 'none_guard', 'partial')]:
        folder = make_free(runs, name, condition=condition, task='Sort by weight', scored=outcome)
        path = folder/'episode.json'
        record = json.loads(path.read_text())
        record.update(task_id='can_weight_sort')
        path.write_text(json.dumps(record, ensure_ascii=False))
        assert episodes.adapter_for(record).family == 'can_weight_sort'
    make_free(runs, 'old_free_sort', task='Sort by weight', scored='failed')
    rows = live.Store([runs]).scan()
    comparison = live.compare(rows, 'can_weight_sort')
    assert len(comparison['episodes']) == 2 and len(live.compare(rows, 'free')['episodes']) == 1
    assert comparison['arms']['harness']['n_success'] == 1 and comparison['arms']['baseline']['n_scored'] == 1
