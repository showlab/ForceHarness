"""Split a tool result (the text the model reads back after an action) into the part the motion layer
reports, the part the force harness added (the physics translation, layer 3) and the scene notes.

The phrases are not copied here: they are read from the runner's own constants in
``forceharness/real/receipt_text.py`` (loaded by file path, so the monitor still needs only the standard library), so the
monitor always matches exactly what the runner writes. In addition:
- forceharness/translation writes "contact events ...", "contact state at end ...", "measurements (...)" and, in the raw
  ablation form, "wrist force/torque ...": the force part, in the force condition only;
- the motion lines (end position, displacement, termination) are the same in both conditions.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Dict, List

_RECEIPT_TEXT = Path(__file__).resolve().parents[2] / "forceharness" / "real" / "receipt_text.py"


def _runner_phrases():
    spec = importlib.util.spec_from_file_location("_fh_receipt_text", _RECEIPT_TEXT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # the module only defines string constants (it imports re)
    return mod


_RT = _runner_phrases()

# A force head ending in ":" opens a block; its indented lines (events, measurements, raw samples) belong to the force
# part as well.
FORCE_HEADS = tuple(dict.fromkeys(("contact events", "contact state at end", "measurements (", "wrist force")
                                  + tuple(h.strip() for h in _RT.FORCE_HEADS)))
SCENE_HEADS = tuple(dict.fromkeys(("[image",) + tuple(h.strip() for h in _RT.SCENE_HEADS)))   # "[image": relay image markers
OBJECT_LINE = _RT.OBJECT_LINE                         # one object of the list, "#1 center [x, y], ..."

# The last section of every relay observation; it ends with the example action JSON that the manual editor and the
# mock model start from.
OBS_MOVE_HEAD = _RT.YOUR_MOVE.lstrip("# ")            # "## YOUR MOVE"
# The tool menu in obs_0 lists done with a success argument (done(success, note)); the mock model then reports
# success=false when it finishes.
DONE_TAKES_SUCCESS = "done(success"
# Rows of the price table (forceharness/eval/prices.yaml) whose trailing comment contains this marker are unverified:
# the model picker shows no price for them.
PRICE_PLACEHOLDER = "placeholder"


def split_lines(text: str) -> List[Dict[str, str]]:
    """[{kind: motion|force|scene, text}] in the original order."""
    rows: List[Dict[str, str]] = []
    in_block = False
    for line in (text or "").split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(SCENE_HEADS) or OBJECT_LINE.match(stripped):
            in_block = False
            rows.append(dict(kind="scene", text=stripped))
        elif stripped.startswith(FORCE_HEADS):
            in_block = stripped.endswith(":")
            rows.append(dict(kind="force", text=line.rstrip()))
        elif in_block and line.startswith("  "):
            rows.append(dict(kind="force", text=line.rstrip()))
        else:
            in_block = False
            rows.append(dict(kind="motion", text=line.rstrip()))
    return rows


def translation(text: str) -> Dict[str, object]:
    rows = split_lines(text)
    force = [r["text"] for r in rows if r["kind"] == "force"]
    return dict(rows=rows, has_force=bool(force), force_text="\n".join(force))
