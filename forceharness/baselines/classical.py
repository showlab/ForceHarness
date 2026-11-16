"""Position-only reference controllers that the expert script (``baselines/expert.py``) cannot provide just by switching
to the pose interface. They never read force and make no API calls.

``PositionOnlyPeg`` is the "Position only" row for peg_in_hole / peg_in_hole_hard (position control plus the arm's own
compliance). It first presses straight down at the noisy hole position to the nominal depth (the "No Force" baseline of
FORGE, "Straight Down" in Schoettler et al. 2020). If the peg does not go in, it probes an Archimedean spiral point by
point (lift -> move across -> shallow press) with the same spiral as the expert's force search (4 mm pitch, radius up to
``search_radius``, i.e. the same designer prior), and decides from the tool height alone whether the peg dropped into the
hole (a kinematic reading, no force). The press target is 5 mm below the hole opening: on a solid surface that is about
15 N (on a hard surface the position interface adds about 3 N per extra millimetre, and deeper targets would trip the
25 N safety stop) with about 1 mm of actual indentation. A tool more than 3 mm below the opening counts as a drop into the
hole (the opening height carries no noise; the scene noise is in x and y only).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

from forceharness.agent.llm import LLMResponse, ToolCall
from forceharness.baselines.expert import _find


class PositionOnlyPeg:
    kind = "scripted"

    def __init__(self, task_name: str, interface: str = "pose", pitch: float = 0.004, press: float = 0.005, drop: float = 0.003):
        from forceharness.hiddenphys.tasks import get_task

        self.task_name = task_name
        self.interface = interface
        self.extra = dict(get_task(task_name).extra)
        self.radius = float(self.extra.get("search_radius", 0.02))
        self.pitch, self.press, self.drop = pitch, press, drop
        self.stage, self.k, self._n = 0, 0, 0
        self.mem: Dict[str, Any] = {}

    def _tc(self, name, **a) -> ToolCall:
        self._n += 1
        return ToolCall(f"p{self._n}", name, a)

    def decide(self, ctx: Dict[str, Any]) -> LLMResponse:
        return LLMResponse(text="", tool_calls=self._policy(ctx), usage_in=0, usage_out=0, latency_s=0.0)

    def _spiral(self, k: int):
        """k-th spiral point as an offset from the hole: radius r = pitch * theta / (2 pi), consecutive points about one
        pitch apart along the arc."""
        th = math.sqrt(4 * math.pi * k)          # arc length ~ pitch * theta^2 / (4 pi); arc length k * pitch at the k-th point gives theta = sqrt(4 pi k)
        r = self.pitch * th / (2 * math.pi)
        return r * math.cos(th), r * math.sin(th), r

    def _policy(self, ctx) -> List[ToolCall]:
        reps = ctx["reports"]
        last = reps[-1] if reps else None
        if self.stage == 0:
            self.stage = 1
            return [self._tc("scene")]
        if self.stage == 1:
            h = np.array(_find(ctx["last_scene"], "hole")["pos"], dtype=float)
            self.mem.update(h=h, z_mouth=float(h[2]) + 0.04, z_in=float(h[2]) + 0.04 - 0.02)
            self.stage = 2
            return [self._tc("move_to", x=float(h[0]), y=float(h[1]), z=self.mem["z_mouth"] + 0.03),
                    self._tc("move_to", x=float(h[0]), y=float(h[1]), z=self.mem["z_mouth"] - self.press)]     # press straight down from above
        if self.stage == 2:
            z = float(last.end_pos[2]) if last is not None else 1.0
            if z < self.mem["z_mouth"] - self.drop:                     # dropped into the hole (tool more than 3 mm below the opening) -> push to the insertion depth and check
                self.stage = 3
                xy = last.end_pos[:2]
                return [self._tc("move_to", x=float(xy[0]), y=float(xy[1]), z=self.mem["z_in"])]
            return self._next_point()
        if self.stage == 3:
            z = float(last.end_pos[2]) if last is not None else 1.0
            if z < self.mem["z_mouth"] - 0.010:                        # peg more than 1 cm in (the position interface cannot push the full 2 cm: friction in the hole, compliant arm) -> release
                self.stage = 9
                return [self._tc("release"), self._tc("move_by", dx=0, dy=0, dz=0.08),
                        self._tc("done", success=True, note="position-only")]
            self.stage = 2                                              # stuck on the rim: lift and continue the search
            return [self._tc("move_by", dx=0, dy=0, dz=0.008)] + self._next_point()
        return [self._tc("done", success=False)]

    def _next_point(self) -> List[ToolCall]:
        self.k += 1
        dx, dy, r = self._spiral(self.k)
        if r > self.radius:
            self.stage = 9
            return [self._tc("release"), self._tc("move_by", dx=0, dy=0, dz=0.08),
                    self._tc("done", success=False, note="position-only: spiral exhausted")]
        h = self.mem["h"]
        x, y = float(h[0] + dx), float(h[1] + dy)
        return [self._tc("move_to", x=x, y=y, z=self.mem["z_mouth"] + 0.004),          # move across 4 mm above the opening
                self._tc("move_to", x=x, y=y, z=self.mem["z_mouth"] - self.press)]     # shallow probing press
