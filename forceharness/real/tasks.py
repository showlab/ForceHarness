"""The six real-robot tasks of the paper (and the free task) as presets of the runner.

A preset fixes the instruction (forceharness/prompts/real/<task_id>.md, read at the start of every episode and
shared by both conditions), records a task version, and may change the per-episode menus or system prompts;
the module-level menus in menu.py never change. live_args are the extra command-line flags the paper's live
runs used (the monitor passes them; scripts/real/<task>.py keeps the runner's defaults otherwise).
"""
from dataclasses import dataclass
import copy
from pathlib import Path

from forceharness.real.menu import CONDITIONS, PRIMITIVES, SEARCH_SLIDE_ONLY

PROMPTS = Path(__file__).resolve().parents[1]/'prompts/real'


@dataclass(frozen=True)
class TaskPreset:
    task_id: str
    version: str | None = None            # recorded as task_version in episode.json
    instruction_file: str | None = None   # under forceharness/prompts/real; None = the operator writes the task
    demo_note: str | None = None          # the synthetic demo cannot check this task: it ends with done(false, note)
    remove: tuple = ()                    # tools removed from both menus
    add: tuple = ()                       # (condition, tool) appended to that condition's menu
    prompt_files: tuple = ()              # (condition, file) system prompts replacing the free-task ones
    search_slide_only: bool = False       # search without the hop mode (plug insertion)
    gripper_width_check: bool = True      # False: the robot-side worker skips its gripper width checks
    live_args: tuple = ()                 # extra flags of the paper's live runs

    def instruction(self):
        """The fixed instruction text (None for the free task)."""
        return None if self.instruction_file is None else (PROMPTS/self.instruction_file).read_text().strip()

    def menus(self):
        """(conditions, primitives) for one episode of this task: copies, so nothing global changes."""
        conditions = copy.deepcopy(CONDITIONS)
        primitives = dict(PRIMITIVES)
        if self.search_slide_only:
            primitives['search'] = SEARCH_SLIDE_ONLY
        for cond in conditions.values():
            cond['menu'] = [name for name in cond['menu'] if name not in self.remove]
        for condition, name in self.add:
            conditions[condition]['menu'].append(name)
        for condition, name in self.prompt_files:
            conditions[condition]['prompt'] = name
        return conditions, primitives


FREE = TaskPreset('free')
TASKS = {t.task_id: t for t in (
    TaskPreset('blackboard_wipe', 'r40', 'blackboard_wipe.md',
               'Interface check only: the synthetic plant has no physical model of the board, the writing or the '
               'eraser, so wiping cannot be verified'),
    # Screw driving: no lift on the menu (after a release the model kept lifting to look instead of turning back
    # in place and gripping the same spot); wiggle on A's menu to help the thread engage.
    TaskPreset('screw_insert', 'screw1', 'screw_insert.md',
               'Interface check only: the synthetic plant has no physical model of the green screw, the threaded hole '
               'or tightening, so screwing in cannot be verified',
               remove=('lift',), add=(('A', 'wiggle'),), live_args=('--max-actions', '60', '--max-minutes', '60')),
    # Peeling: task-specific system prompts; the control may stop at contact (move_until_contact without force
    # readings); the gripper width checks are off because the peeler is clamped by the operator.
    TaskPreset('peel_vegetable', 'peel2', 'peel_vegetable.md',
               'Interface check only: the synthetic plant has no physical model of the peeler or the cucumber/eggplant, '
               'so peeling cannot be verified',
               add=(('B', 'move_until_contact'),),
               prompt_files=(('A', 'peel_vegetable_force.md'), ('B', 'peel_vegetable_baseline.md')),
               gripper_width_check=False,
               live_args=('--max-actions', '60', '--max-minutes', '60', '--force-max', '20', '--slide-force-default', '10')),
    TaskPreset('usb_insert', 'plug1', 'usb_insert.md',
               "Interface check only: the synthetic plant has no physical model of the USB plug or the power strip's "
               'USB port, so insertion cannot be verified',
               add=(('A', 'wiggle'),), search_slide_only=True, live_args=('--force-max', '20', '--insert-force-max', '20')),
    TaskPreset('can_weight_sort', 'cansort1', 'can_weight_sort.md',
               'Interface check only: the synthetic plant has no physical model of the bottles, the plates or the '
               'water content, so the sorting cannot be verified',
               live_args=('--max-actions', '80', '--max-minutes', '60')),
    TaskPreset('touch_firmness', 'firm1', 'touch_firmness.md',
               'Interface check only: the synthetic plant has no physical model of the soft and hard objects, so '
               'firmness cannot be judged',
               live_args=('--max-actions', '15')),
)}
