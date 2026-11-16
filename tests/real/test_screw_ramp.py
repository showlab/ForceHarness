"""While turning, the screw's push rises in steps from 5 N to `push` (later segments of one call start there) instead
of dropping to 3 N; a full press turns at the full push. Offline only."""
import numpy as np

from forceharness.real import primitives as cp
from test_screw import held_on_surface


def test_a_quick_segment_steps_its_push_up_without_a_stall():
    p, b, prim = held_on_surface()
    rep = prim.call('screw', angle_deg=-30, quick=True)
    b.__exit__(None, None, None)
    assert rep.termination == 'completed' and not b.fault
    assert rep.derived['quick_press'] and rep.derived['turn_push_N'][0] == cp.SCREW_TURN_START_N
    assert rep.derived['turn_push_N'][1] > 12.
    w = np.asarray(rep.wrench_rows)
    turning = w[w[:, 0] >= rep.derived['press_s']]
    assert turning[-5:, 3].mean()-turning[:5, 3].mean() > 4.       # the push really rose while turning


def test_a_full_press_turns_on_at_the_full_push():
    p, b, prim = held_on_surface()
    rep = prim.call('screw', angle_deg=-30)
    b.__exit__(None, None, None)
    assert rep.derived['turn_push_N'] == [cp.SCREW_PUSH_N, cp.SCREW_PUSH_N] and not rep.derived['quick_press']
