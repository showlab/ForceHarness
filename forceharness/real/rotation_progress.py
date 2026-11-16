"""Progress watchdogs of the workstation primitive loop (layer 2): stop a stuck motion with a tool result.

RotationProgress tells a stationary wrist (static friction holding a turn) from a progressing pivot turn;
TrackingProgress runs the robot side's tracking-stall test with less time. Both stop the primitive before the
robot-side guard would fault and end the whole episode; they leave the robot-side limits unchanged.
Use relative rotations within the window, not differences of unsigned angles
from the motion start: a wrist correcting its orientation can cross that start.
"""
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation as R


class RotationProgress:
    def __init__(self, *, axis=None, pivot=None, window_s=3., earliest_s=3.2):
        self.axis = None if axis is None else np.asarray(axis, dtype=float)
        if self.axis is not None:
            self.axis = self.axis / np.linalg.norm(self.axis)
        self.pivot = None if pivot is None else np.asarray(pivot, dtype=float)
        self.window_s, self.earliest_s = window_s, earliest_s
        self.rows = deque()
        self.metrics = {}

    def observe(self, t, pos, quat):
        self.rows.append((float(t), np.asarray(pos, dtype=float).copy(),
                          np.asarray(quat, dtype=float).copy()))
        # Keep the sample immediately before the cutoff, so a full window is
        # available even when the callback timing is not exactly periodic.
        while len(self.rows) > 1 and self.rows[1][0] <= t-self.window_s:
            self.rows.popleft()
        rotations = R.from_quat([row[2] for row in self.rows])
        wrist_span = float(np.degrees((rotations*rotations[0].inv()).magnitude()).max())
        arc_span = 0.
        if self.axis is not None and self.pivot is not None:
            v = np.array([row[1] for row in self.rows])-self.pivot
            v -= (v@self.axis)[:, None]*self.axis
            radius = float(np.linalg.norm(v[0]))
            if radius > 1e-4:
                angles = np.unwrap(np.arctan2(np.cross(v[0], v)@self.axis, v@v[0]))
                arc_span = float(np.ptp(angles)*radius)
        elapsed = float(t-self.rows[0][0])
        self.metrics = dict(window_s=round(elapsed, 3), wrist_span_deg=round(wrist_span, 4),
                            arc_span_mm=round(arc_span*1000, 4),
                            min_wrist_span_deg=.2, min_arc_span_mm=.5)
        return (t >= self.earliest_s and elapsed >= self.window_s-1e-9
                and wrist_span < .2 and arc_span < .0005)


class TrackingProgress:
    """The robot side's tracking-stall test (its target moved >= 0.5 mm within the last 1 s, the hand got
    no 0.5 mm further for 4 s, error > 2 mm) run on the workstation with less time, so the primitive stops
    with a tool result instead of the robot-side fault ending the whole session (for example a sliding
    search pressed so hard that friction stops the hand)."""
    def __init__(self, *, seconds=3.5):
        self.seconds = seconds
        self.targets = deque()
        self.progress_pos = self.progress_at = None
        self.metrics = {}

    def observe(self, t, pos, target):
        t, pos, target = float(t), np.asarray(pos, dtype=float), np.asarray(target, dtype=float)
        self.targets.append((t, target.copy()))
        while len(self.targets) > 1 and self.targets[1][0] <= t-1.:
            self.targets.popleft()      # keep the anchor just before the 1 s window, as the robot side does
        recent = np.array([x for _, x in self.targets])
        moving = len(recent) > 1 and float(np.linalg.norm(np.diff(recent, axis=0), axis=1).sum()) >= .0005
        if self.progress_at is None or not moving or np.linalg.norm(pos-self.progress_pos) > .0005:
            self.progress_pos, self.progress_at = pos.copy(), t
        error = float(np.linalg.norm(target-pos))
        if error <= .002:
            self.progress_at = t
        self.metrics = dict(stuck_s=round(t-self.progress_at, 2), target_lead_mm=round(error*1000, 1),
                            limit_s=self.seconds, nuc_limit_s=4.)
        return t-self.progress_at > self.seconds and error > .002
