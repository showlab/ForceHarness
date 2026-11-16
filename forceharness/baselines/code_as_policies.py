"""Code-as-Policies baseline: the model writes one complete Python program that drives the arm through our primitives
via a ``robot`` object, and the program then runs to the end without further model calls.

It reuses the self-terminating force primitives (layer 2) and, on the force interface, the physics translation layer
(layer 3); only the decision loop differs from ForceHarness, which decides between primitives step by step.

``ProgramEpisode`` (harness name ``"cap"``) makes one model call per episode. The system prompt
(``prompts/hiddenphys/code_as_policies.md``) holds the program interface, the robot API of the episode's interface
(signatures and descriptions generated from the tool schemas in ``agent/tools.py``) and the fields that every call
returns; the user message holds the task instruction, the initial scene (the same noisy scene the tool-calling
``Episode`` gives) and the safety limit. The Python code block of the reply runs in a persistent in-process namespace
(modelled on the CaP-X sandbox of Inspect Robots). Conditions are named ``cap|pose`` and ``cap|force``: the ``fmt`` slot
carries the harness name, so grouping by ``fmt|interface`` keeps them apart from the tool-calling rows.

What a program sees is the return value of each ``robot.*`` call (a read-only dict), filtered by interface:

- pose: kinematics only. The fields are exactly those that ``translation/base.py::kinematic_summary(rep,
  hide_force=True)`` writes (the termination is reworded by the same rules, by parsing its output, so both stay in step):
  tool position, commanded / actual displacement, how far a move stopped short of its target, grasp / release results,
  ... This matches a ``none|pose`` tool result item by item; there is no force, contact event or physical quantity.
- force: additionally the force fields: the raw termination, events, peak forces, ``final_wrench(_net)``, all derived
  signals, ``quantities`` (physical quantities from the translation layer) and ``wrench_rows``.
- ``ContactReport.truth`` and ``sim.task_state`` never reach a return value.

Budgets are checked in the wrapper (``max_tool_calls``, ``max_sim_time``, ``max_wall_time``); exceeding one raises
``BudgetExceeded`` and ends the program. It and the stop signal of ``robot.done()`` derive from ``BaseException``, so an
``except Exception`` in the program does not catch them, and any ``robot.*`` call after catching one raises again.
An exception of the program itself ends the episode as a failure (``stop_reason="program_error"``, traceback in the
record). As in ``Episode``, the record's ``error`` field is reserved for infrastructure errors (model API failure,
simulator could not be built), which are rerun rather than scored; a crashing program is a task failure, so it is not
written there.

The executor is not a security sandbox (as the CaP-X sandbox also states); every episode already runs in its own process.
It applies basic hygiene so that a program cannot reach simulator internals (ground truth): imports only from an allow
list, no ``open`` / ``eval`` / ``exec`` / ``globals`` / ``vars`` or similar builtins, and no access to dunder attributes or
to private attributes of objects other than ``self``.
"""
from __future__ import annotations

import ast
import builtins
import contextlib
import dataclasses
import hashlib
import io
import json
import linecache
import math
import re
import signal
import string
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from forceharness.primitives.report import ContactReport
from forceharness.translation import make_injector
from forceharness.translation.base import _short_of_target, kinematic_summary
from forceharness.agent.llm import LLMResponse
from forceharness.agent.loop import PROMPT_DIR, Episode, EpisodeConfig, LLMAgent, _json_safe

HARNESSES = ("cap",)
ROOT = Path(__file__).resolve().parents[2]

# modules a program may import (by top-level package name)
ALLOWED_MODULES = frozenset({"math", "cmath", "numpy", "json", "itertools", "collections", "statistics", "functools", "operator",
                             "random", "re", "copy", "dataclasses", "typing", "enum", "heapq", "bisect", "decimal", "fractions",
                             "string", "textwrap", "time"})
_BANNED_BUILTINS = {"open", "eval", "exec", "compile", "input", "breakpoint", "globals", "locals", "vars", "help", "exit", "quit",
                    "copyright", "credits", "license", "__loader__", "__spec__"}

# Keys that name evaluation ground truth (report.truth, the tasks' task_state, truth_fn outputs): anything handed back to
# the program is scrubbed of them and every hit is recorded (normally there is none).
TRUTH_KEYS = frozenset({
    "truth", "task_state", "verdict_correct", "contact_fraction", "touching_end", "peak_object_contact_force",
    "peak_object_contact_force_N", "tared", "label", "mass", "hole_xy", "clearance_m", "theta_u_deg", "theta_stop_deg", "locked",
    "unlatched_ever", "damaged", "peak_handle_force_N", "f_damage", "lever_max_deg", "com_offset_m", "cubeB_z_error", "broke_free",
    "breakaway_N", "handle_radius_m", "hold_N_max", "pregrasped", "good_wiped", "light_wipes", "peak_normal_N", "f_min", "f_max",
    "peg_minus_hole_xy_mm", "depth_mm", "bar_tilt_deg", "com_lever_xy_mm", "bar_minus_eef_mm", "lever_deg", "door_deg",
    "hinge_damping", "hinge_friction", "table_top",
})


# ============================================================================= stop signals and errors
class ProgramStop(BaseException):
    """Stop signal inside the executor. It derives from BaseException, so ``except Exception`` in a program does not catch it."""
    reason = "stop"


class ProgramDone(ProgramStop):
    reason = "done"


class BudgetExceeded(ProgramStop):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class RobotError(Exception):
    """A robot.* call failed (bad arguments, not available on this interface, simulator error). The program may catch it;
    if it does not, the episode ends."""


class ProgramRejected(Exception):
    """The static check failed (import outside the allow list, access to a private or dunder attribute)."""


# ============================================================================= data conversion
def _plain(o):
    """Convert to plain Python (dict / list / float / int / str / bool / None), removing numpy types and dataclasses."""
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return _plain(dataclasses.asdict(o))
    if o is None or isinstance(o, (str, int, float, bool)):
        return o
    return str(o)


def _plain_arg(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, tuple):
        return [_plain_arg(x) for x in v]
    if isinstance(v, list):
        return [_plain_arg(x) for x in v]
    return v


class ReadOnlyDict(dict):
    """Return value of robot.* calls: read-only (copy it with dict(...) to modify). json.dumps works as usual; it pickles as a
    plain dict."""
    __slots__ = ()

    def _readonly(self, *a, **k):
        raise TypeError("robot results are read-only; make a copy with dict(...) first")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _readonly

    def copy(self):
        return dict(self)

    def __reduce__(self):
        return (dict, (dict(self),))


def _ro(o):
    if isinstance(o, dict):
        return ReadOnlyDict({k: _ro(v) for k, v in o.items()})
    if isinstance(o, list):
        return [_ro(v) for v in o]
    return o


def _tail(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else "...(truncated)...\n" + s[-n:]


def program_sha1(code: Optional[str]) -> Optional[str]:
    return hashlib.sha1(code.encode("utf-8")).hexdigest()[:12] if code else None


def program_key(code: Optional[str]) -> Optional[str]:
    """Identity of a program: the same AST counts as the same program (comments, blank lines and formatting are ignored)."""
    if not code:
        return None
    try:
        return hashlib.sha1(ast.dump(ast.parse(code)).encode("utf-8")).hexdigest()[:12]
    except SyntaxError:
        return hashlib.sha1(code.strip().encode("utf-8")).hexdigest()[:12]


# ============================================================================= results: the fields a program sees
_KEY_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=")


def receipt(rep: ContactReport, interface: str, proprio: str = "minimal") -> Dict[str, Any]:
    """Fields returned to the program for one primitive call. pose / guard: only what kinematic_summary writes; force: also
    the force fields."""
    hide = interface != "force"
    lines = kinematic_summary(rep, hide_force=hide, proprio=proprio).splitlines()
    result = rep.termination
    for ln in lines:
        if ln.startswith("result: "):
            result = ln[len("result: "):].split("; duration")[0]
            break
    out: Dict[str, Any] = dict(primitive=rep.primitive, params=_plain(rep.params), result=result,
                               start_pos=_plain(rep.start_pos), end_pos=_plain(rep.end_pos),
                               commanded_disp=_plain(rep.commanded_disp), actual_disp=_plain(rep.actual_disp))
    if proprio == "full":
        out["duration_s"] = round(float(rep.duration), 3)
        if rep.gripper_opening is not None:
            out["gripper_opening_m"] = round(float(rep.gripper_opening), 4)
    if any(ln.startswith("NOTE: the target was NOT reached") for ln in lines):
        short = _short_of_target(rep)
        if short is not None:
            out["short_of_target_m"] = round(float(short), 4)
    if rep.yaw_deg:
        out["yaw_change_deg"] = _plain(rep.yaw_deg)
    if rep.error:
        out["error"] = str(rep.error)
    for ln in lines:                       # derived keys that kinematic_summary writes (its allow list and hide_force rules apply)
        m = _KEY_LINE.match(ln)
        if m and m.group(1) in rep.derived and m.group(1) not in out:
            out[m.group(1)] = _plain(rep.derived[m.group(1)])
    if not hide:
        out.update(events=[_plain(e) for e in rep.events], peak_force_N=_plain(rep.peak_force), peak_torque_Nm=_plain(rep.peak_torque),
                   mean_force_in_contact_N=_plain(rep.mean_force_in_contact), final_wrench=_plain(rep.final_wrench),
                   final_wrench_net=_plain(rep.final_wrench_net), derived=_plain(rep.derived), quantities=_plain(rep.quantities),
                   wrench_rows=_plain(rep.wrench_rows), pos_rows=_plain(rep.pos_rows))
    return out


def scrub_truth(obj, extra_keys: Sequence[str] = (), path: str = "") -> List[str]:
    """Delete in place every entry of a nested dict whose key names ground truth; return the removed paths (normally none)."""
    keys = TRUTH_KEYS | set(extra_keys)
    removed: List[str] = []
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            if k in keys:
                removed.append(f"{path}/{k}")
                del obj[k]
            else:
                removed += scrub_truth(obj[k], extra_keys, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            removed += scrub_truth(v, extra_keys, f"{path}[{i}]")
    return removed


def _downsample(rows: List[List[float]], k: int) -> List[List[float]]:
    if not rows:
        return []
    idx = range(len(rows)) if len(rows) <= k else np.linspace(0, len(rows) - 1, k).round().astype(int)
    return [[round(float(v), 2) for v in rows[i]] for i in idx]


# ============================================================================= code-block extraction and static checks
_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+.-]*)[ \t]*\r?\n(.*?)```", re.S)
_OPEN_FENCE = re.compile(r"```[ \t]*(?:python|py|python3)?[ \t]*\r?\n(.*)$", re.S | re.I)


def extract_program(text: str) -> Tuple[Optional[str], int]:
    """Extract the program from a model reply. Prefer ```python blocks (the longest one if there are several: a revision
    reply may first quote a snippet of old code), then blocks without a language tag. Without any fence, the whole reply
    counts as code if it parses; with an opening fence but no closing one (truncated output), take everything after the
    fence. Returns (program or None, number of code blocks)."""
    text = text or ""
    blocks = [(lang.lower(), body) for lang, body in _FENCE.findall(text)]
    py = [b for lang, b in blocks if lang in ("python", "py", "python3")]
    cands = py or [b for lang, b in blocks if lang == ""]
    if cands:
        return max(cands, key=len).strip("\n"), len(blocks)
    if not blocks:
        m = _OPEN_FENCE.search(text)
        if m and m.group(1).strip():
            return m.group(1).strip("\n"), 1
        s = text.strip()
        if s and "\n" in s:
            try:
                ast.parse(s)
                return s, 0
            except SyntaxError:
                pass
    return None, len(blocks)


def check_program(code: str) -> ast.Module:
    """Syntax and hygiene check. Raises SyntaxError unchanged and ProgramRejected for disallowed code."""
    tree = ast.parse(code, filename="<program>")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            a = node.attr
            if a.startswith("__") or (a.startswith("_") and not (isinstance(node.value, ast.Name) and node.value.id in ("self", "cls"))):
                raise ProgramRejected(f"line {node.lineno}: access to attribute '{a}' is not allowed in robot programs "
                                      "(no dunder attributes, no private attributes of other objects)")
        elif isinstance(node, ast.Import):
            for al in node.names:
                if al.name.split(".")[0] not in ALLOWED_MODULES:
                    raise ProgramRejected(f"line {node.lineno}: import of '{al.name}' is not allowed; allowed modules: "
                                          + ", ".join(sorted(ALLOWED_MODULES)))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level or mod.split(".")[0] not in ALLOWED_MODULES:
                raise ProgramRejected(f"line {node.lineno}: import from '{mod}' is not allowed; allowed modules: "
                                      + ", ".join(sorted(ALLOWED_MODULES)))
    return tree


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in ALLOWED_MODULES:
        raise ImportError(f"import of '{name}' is not allowed in robot programs")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _guard_name(name):
    if not isinstance(name, str) or name.startswith("_"):
        raise AttributeError(f"access to attribute {name!r} is not allowed in robot programs")
    return name


def _safe_getattr(obj, name, *default):
    return getattr(obj, _guard_name(name), *default)


def _safe_hasattr(obj, name):
    return hasattr(obj, _guard_name(name))


def _safe_setattr(obj, name, value):
    return setattr(obj, _guard_name(name), value)


def _safe_delattr(obj, name):
    return delattr(obj, _guard_name(name))


def _exit(*a):
    raise SystemExit(*a)


def safe_builtins() -> Dict[str, Any]:
    b = {k: v for k, v in builtins.__dict__.items() if k not in _BANNED_BUILTINS}
    b.update(__import__=_safe_import, getattr=_safe_getattr, hasattr=_safe_hasattr, setattr=_safe_setattr,
             delattr=_safe_delattr, exit=_exit, quit=_exit)
    return b


def _program_traceback(e: BaseException) -> str:
    """Keep only the program's own frames (file name <program>) plus the exception line; harness internals are not exposed."""
    frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == "<program>"]
    lines = ["Traceback (only the lines of your program):"]
    for f in frames:
        lines.append(f"  line {f.lineno}, in {f.name}")
        if f.line:
            lines.append(f"    {f.line.strip()}")
    lines.append(f"{type(e).__name__}: {e}")
    return "\n".join(lines)


# ============================================================================= API documentation written into the prompt
_TYPES = {"number": "float", "boolean": "bool", "string": "str", "integer": "int"}


def _ptype(spec: Dict[str, Any]) -> str:
    if "enum" in spec:
        return " | ".join(repr(e) for e in spec["enum"])
    t = spec.get("type")
    if t == "array":
        return "[x, y, z]" if spec.get("minItems") == 3 and spec.get("maxItems") == 3 else "list"
    return _TYPES.get(t, t or "any")


def param_order(tool: Dict[str, Any]) -> List[str]:
    """Order of positional arguments: the schema's required parameters first, then the others in the order of its properties."""
    props = list((tool.get("parameters") or {}).get("properties", {}))
    req = list((tool.get("parameters") or {}).get("required", []))
    return [k for k in props if k in req] + [k for k in props if k not in req]


def _wrap(s: str, indent: str = "    ", width: int = 118) -> str:
    import textwrap
    return textwrap.fill(s, width=width, initial_indent=indent, subsequent_indent=indent)


def api_doc(tools: List[Dict[str, Any]]) -> str:
    """List the robot methods a program can call on this interface; signatures and descriptions come from the schemas in
    agent/tools.py (word for word the same as the tool-calling interface)."""
    out = []
    for t in tools:
        name = t["name"]
        if name in ("scene", "done"):
            continue
        props = (t.get("parameters") or {}).get("properties", {})
        req = set((t.get("parameters") or {}).get("required", []))
        order = param_order(t)
        sig = ", ".join(k if k in req else f"{k}=None" for k in order)
        out.append(f"robot.{name}({sig})")
        out.append(_wrap(t["description"]))
        if order:
            args = []
            for k in order:
                v = props[k]
                a = f"{k}: {_ptype(v)}"
                if v.get("description"):
                    a += f" ({v['description']})"
                args.append(a)
            out.append(_wrap("arguments: " + "; ".join(args) + ". Omitted / None arguments take the primitive's default."))
    scene = next((t for t in tools if t["name"] == "scene"), None)
    if scene is not None:
        out.append("robot.scene()")
        out.append(_wrap(scene["description"] + " Returns a list of dicts (name, kind, pos, ...) in the same format as the scene "
                         "below; counts as one robot call."))
    out.append("robot.state()")
    out.append(_wrap("Current tool position and gripper state as a dict (same as the 'state' field of every result). Free: not "
                     "counted as a robot call."))
    done = next((t for t in tools if t["name"] == "done"), None)
    if done is not None:
        out.append('robot.done(success, verdict="", note="")')
        out.append(_wrap(done["description"] + " Calling it ends the program."))
    return "\n".join(out)


_RETURNS_POSE = """\
  primitive, params        the primitive that ran and its parameters (after argument clean-up)
  result                   how it ended: "completed", "stopped early (motion blocked)", "stopped early (motion blocked, safety stop)",
                           "stopped early (rotation blocked, safety stop)", "stopped early (not moving)", "stopped early", "timeout", "error"
  start_pos, end_pos       tool centre point [x, y, z] (m) before / after the primitive
  commanded_disp, actual_disp   commanded and actual displacement [dx, dy, dz] (m)
  short_of_target_m        only when a move ended more than 5 mm short of its target
  yaw_change_deg           only when the tool turned about the vertical axis
  opening_m, closed_on_something, fully_closed   (grasp / release) final finger opening, and whether something is between the fingers
  rotated_deg, wrist_limit_clipped, requested_deg, executed_deg   (rotate_z / point_tool) how far the tool actually turned
  note, arg_notes, error   remarks (e.g. renamed or ignored arguments)
  state                    {"tool_pos": [x, y, z], "gripper": "closed" | "open command", "gripper_opening_mm": ... (only while open)}
There is no force, torque or contact information of any kind."""

_RETURNS_FORCE = """\
  primitive, params        the primitive that ran and its parameters (after argument clean-up)
  result                   how it ended: "completed", "contact" (stopped by the contact guard), "force_cap" / "torque_cap" (safety
                           stop), "stall", "resistance_drop", "no_contact_within_max_dist" (move_until_contact), "dropped" / "stuck" /
                           "no_surface" / "out_of_range" / "lost_contact" (search), "seated" / "jammed" / "stalled" (insert), "timeout", "error"
  start_pos, end_pos       tool centre point [x, y, z] (m) before / after the primitive
  commanded_disp, actual_disp   commanded and actual displacement [dx, dy, dz] (m)
  short_of_target_m        only when a move ended more than 5 mm short of its target
  yaw_change_deg           only when the tool turned about the vertical axis
  opening_m, closed_on_something, fully_closed   (grasp / release) final finger opening, and whether something is between the fingers
  contact_found, contact_distance_m, contact_pos (probe); rotated_deg, wrist_limit_clipped (rotate_z / point_tool); note, arg_notes, error
  state                    {"tool_pos": [x, y, z], "gripper": "closed" | "open command", "gripper_opening_mm": ... (only while open)}
  events                   list of {t, kind, force, detail, pos}: kind = contact_start / contact_end / force_cap / torque_cap / stall /
                           slip / resistance_drop / ...; t = seconds since the primitive started; force = |F| (N) at that moment
  peak_force_N, peak_torque_Nm, mean_force_in_contact_N
  final_wrench, final_wrench_net   [Fx, Fy, Fz, Tx, Ty, Tz] at the end (N, Nm, world frame; positive = the environment pushes the
                           gripper that way, e.g. pressing down on the table gives Fz > 0; _net = with the weight of what the gripper
                           holds subtracted)
  derived                  primitive-specific signals, e.g. lift: payload_estimate_kg, hold_force_z_N, hold_torque_xy_Nm,
                           hold_torque_xy_mag_Nm; push: force_along_push_mean_N, regulation_error_N (click_peak_N); slide:
                           normal_force_mean_N, tangential_force_mean_N, friction_ratio, lost_contact_fraction; search: dropped,
                           advance_along_normal_m, radius_searched_m; insert: seated, jammed, stalled, advance_along_normal_m,
                           lateral_force_end_N; rotate_z: torque_z_peak_Nm, torque_z_final_Nm; turn: turned_deg,
                           resisting_moment_peak_Nm, click_detected, click_at_deg; pull: path_length_m, turned_deg, pull_force_mean_N,
                           pivot_xy, radius_m
  quantities               physical quantities estimated from the force signal: list of {name, value, unit, sigma, n, window, method,
                           flags, extra}; names include payload_mass (kg), com_offset (cm), contact_height (m), contact_point,
                           contact_normal, impact_peak (N), stiffness (N/mm), normal_force (N), friction_coefficient,
                           insertion_depth, pull_force, path_turn, path_radius, free_direction
  wrench_rows              up to 160 samples [t, Fx, Fy, Fz, Tx, Ty, Tz] over the primitive (same sign convention, not net of the load)
  pos_rows                 tool position [x, y, z] at each of those samples"""

SENSING = {
    "pose": ("only motion results: where the tool ended up, how far it moved, whether a motion stopped early, and what the gripper "
             "closed on. No force or torque information."),
    "force": ("motion results plus the wrist force/torque measured during every primitive: contact events, peak and final forces, "
              "derived signals, physical quantities estimated from the force signal, and the raw samples."),
}


def returns_doc(interface: str) -> str:
    return _RETURNS_FORCE if interface == "force" else _RETURNS_POSE


# ============================================================================= wall-clock budget
# SIGALRM works only in the main thread; every episode runs in the main thread of its own process.
@contextlib.contextmanager
def _wall_alarm(seconds: float, ep: "ProgramEpisode"):
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
        yield
        return

    def handler(signum, frame):
        if ep._exec_active:
            ep._halt("max_wall_time")

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, max(0.1, float(seconds)), 5.0)    # then every 5 s, so a program that swallows the exception still stops
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


# ============================================================================= one episode
class ProgramEpisode(Episode):
    """One Code-as-Policies episode. With program=None the model is called once for the program; with a given program
    the program runs directly without a model call (e.g. to evaluate a frozen program)."""

    def __init__(self, sim, client, cfg: EpisodeConfig, harness: Optional[str] = None, program: Optional[str] = None,
                 phase: str = "eval", prims=None):
        harness = harness or cfg.fmt
        if harness not in HARNESSES:
            raise ValueError(f"unknown program harness {harness!r}; choose from {HARNESSES}")
        # No injector is used here (receipt() filters the fields per interface); "none" only keeps
        # make_injector(cfg.fmt="cap") from failing.
        super().__init__(sim, client, cfg, injector=make_injector("none"), prims=prims)
        self.harness = harness
        # LLMAgent -> its LLMClient; an LLMClient or FakeProgramLLM is used as is. LLMClient itself has a .client (the
        # OpenAI SDK object) that must not be unwrapped again: getattr(client, "client", client) would reach the SDK object
        # and fail with "'Chat' object is not callable".
        self.client = client.client if isinstance(client, LLMAgent) else client
        self.program = program
        self.phase = phase
        self.prim_tools = {t["name"]: t for t in self.tools if t["name"] not in ("scene", "done")}
        self.param_orders = {n: param_order(t) for n, t in self.prim_tools.items()}
        self.visible: List[Dict[str, Any]] = []                  # all return values the program has seen (plain dicts)
        self.claimed: Optional[bool] = None
        self.verdict, self.note = "", ""
        self._halt_reason: Optional[str] = None
        self._exec_active = False
        self._prepared = False
        self.pregrasp = None
        self.t_wall0 = time.time()
        self.stdout_text = ""
        self.program_error: Optional[str] = None
        self.traceback_full: Optional[str] = None
        self.traceback_program: Optional[str] = None
        self.final_scene: Any = None
        self.n_code_blocks: Optional[int] = None
        self.llm_latency = 0.0
        self.record: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ prompt
    def system_prompt(self) -> str:
        tpl = string.Template((PROMPT_DIR / "hiddenphys" / "code_as_policies.md").read_text(encoding="utf-8"))
        txt = tpl.substitute(interface=self.cfg.interface, sensing_note=SENSING["force" if self.cfg.interface == "force" else "pose"],
                             api_doc=api_doc(self.tools), returns_doc=returns_doc(self.cfg.interface), f_cap=f"{self.cfg.f_cap:g}",
                             max_tool_calls=str(self.cfg.max_tool_calls), max_sim_time=f"{self.cfg.max_sim_time:g}")
        if self.cfg.playbook:
            txt += "\n\n" + (PROMPT_DIR / self.cfg.playbook).read_text(encoding="utf-8").strip()
        return txt

    def prepare(self) -> None:
        """Set-up motions before the episode (e.g. lock_or_stiff grasps the handle): as in Episode.run, not counted as calls
        and not shown to the model."""
        if self._prepared:
            return
        self._prepared = True
        if getattr(self.task, "pregrasp_hook", None) is not None:
            try:
                self.pregrasp = self.task.pregrasp_hook(self.sim, self.prims)
            except Exception as e:
                self.pregrasp = dict(error=f"{type(e).__name__}: {e}")

    def intro_text(self, example: bool = False) -> str:
        """User message: task instruction + initial scene (Episode.scene_text, the same noisy scene) + safety limit. With
        example=True (refinement loop) it says that the scene is only an example from the first development run."""
        self.prepare()
        s = f"TASK: {self.task.instruction}\n\n"
        if example:
            s += ("Example of the scene (from the first development run; every run is a new randomized episode, so the program "
                  "must read robot.scene() itself):\n")
        s += self.scene_text()
        s += (f"\n\nSafety: the controller aborts any motion whose wrist force exceeds {self.cfg.f_cap:g} N.\n"
              + ("Write the first version of the program now" if example else "Write the program now")
              + ": reply with exactly one ```python code block containing the complete program.")
        return s

    def state_dict(self) -> Dict[str, Any]:
        """Dict version of state_text (same proprioception rules: with proprio=minimal the finger opening is hidden while the
        gripper is closed)."""
        p = self.sim.eef_pos
        g = self.sim.gripper_opening
        s: Dict[str, Any] = {"tool_pos": [round(float(x), 4) for x in p]}
        if g is not None:
            if self.cfg.proprio != "full" and self.prims.grip_closed:
                s["gripper"] = "closed"
            else:
                s["gripper"] = "closed" if self.prims.grip_closed else "open command"
                s["gripper_opening_mm"] = round(float(g) * 1000, 1)
        if self.cfg.proprio == "full":
            s["sim_time_s"] = round(float(self.sim.sim_time), 1)
        return s

    # ------------------------------------------------------------------ robot API (robot.* in the program)
    def _halt(self, reason: str):
        if self._halt_reason is None:
            self._halt_reason = reason
        if self._halt_reason == "done":
            raise ProgramDone()
        raise BudgetExceeded(self._halt_reason)

    def _check_budget(self) -> None:
        if self._halt_reason is not None:
            self._halt(self._halt_reason)
        if self.n_tool_calls >= self.cfg.max_tool_calls:
            self._halt("max_tool_calls")
        if self.sim.sim_time > self.cfg.max_sim_time:
            self._halt("max_sim_time")
        if time.time() - self.t_wall0 > self.cfg.max_wall_time:
            self._halt("max_wall_time")

    def _robot_call(self, name: str, args: tuple, kwargs: Dict[str, Any]):
        self._check_budget()
        order = self.param_orders[name]
        if len(args) > len(order):
            raise RobotError(f"robot.{name}() takes at most {len(order)} positional arguments ({len(args)} given)")
        kw = dict(zip(order, args))
        for k, v in kwargs.items():
            if k in kw:
                raise RobotError(f"robot.{name}() got multiple values for argument '{k}'")
            kw[k] = v
        kw = {k: _plain_arg(v) for k, v in kw.items() if v is not None}       # None = use the primitive's default
        self.n_tool_calls += 1
        try:
            rep = self.prims.call(name, **kw)
        except ProgramStop:
            raise
        except TypeError as e:
            msg = f"robot.{name}: bad arguments: {e}"
            self.visible.append(dict(call=name, args=_plain(kw), error=msg))
            raise RobotError(msg) from None
        except Exception as e:       # ValueError (zero direction, not on this interface) / simulator exception
            msg = f"robot.{name}: {e}" if isinstance(e, ValueError) else f"robot.{name}: {type(e).__name__}: {e}"
            self.visible.append(dict(call=name, args=_plain(kw), error=msg))
            raise RobotError(msg) from None
        self.reports.append(rep)
        for q in rep.quantities:
            self.mlog.append(dict(turn=len(self.reports), primitive=rep.primitive, **{k: q.get(k) for k in ("name", "value", "unit", "sigma", "flags", "extra")}))
        d = receipt(rep, self.cfg.interface, self.cfg.proprio)
        d["state"] = self.state_dict()
        self.visible.append(dict(call=name, **d))
        return _ro(d)

    def _robot_scene(self):
        self._check_budget()
        self.n_tool_calls += 1
        sc = _plain(self.sim.scene())
        self.last_scene = sc
        self.visible.append(dict(call="scene", scene=sc))
        return [_ro(o) for o in sc]

    def _robot_state(self):
        if self._halt_reason is not None:
            self._halt(self._halt_reason)
        return _ro(self.state_dict())

    def _robot_done(self, success=False, verdict="", note=""):
        if self._halt_reason is not None:
            self._halt(self._halt_reason)
        self.claimed = bool(success)
        self.verdict = "" if verdict is None else str(verdict)
        self.note = "" if note is None else str(note)
        self.visible.append(dict(call="done", success=self.claimed, verdict=self.verdict, note=self.note))
        self._halt("done")

    def make_robot(self):
        ep = self
        ns: Dict[str, Any] = {"__slots__": (), "__doc__": "The robot API of this episode (see the system prompt)."}

        def _method(name):
            def method(self, *args, **kwargs):
                return ep._robot_call(name, args, kwargs)
            method.__name__ = name
            method.__qualname__ = f"Robot.{name}"
            method.__doc__ = ep.prim_tools[name]["description"]
            return method

        for n in self.prim_tools:
            ns[n] = _method(n)
        ns["scene"] = lambda self: ep._robot_scene()
        ns["state"] = lambda self: ep._robot_state()
        ns["done"] = lambda self, success=False, verdict="", note="": ep._robot_done(success, verdict, note)
        names = ", ".join(list(self.prim_tools) + ["scene", "state", "done"])
        ns["__repr__"] = lambda self: f"<robot ({ep.cfg.interface} interface): {names}>"

        def _missing(self, name):
            raise AttributeError(f"robot.{name} is not available in this episode ({ep.cfg.interface} interface); available: {names}")

        ns["__getattr__"] = _missing
        return type("Robot", (), ns)()

    # ------------------------------------------------------------------ execution
    def _execute(self, code: str) -> str:
        try:
            tree = check_program(code)
        except SyntaxError as e:
            self.program_error = f"SyntaxError: {e.msg} (line {e.lineno})"
            self.traceback_program = self.program_error
            return "program_error"
        except ProgramRejected as e:
            self.program_error = f"rejected: {e}"
            self.traceback_program = self.program_error
            return "program_error"
        ns = {"__builtins__": safe_builtins(), "__name__": "__main__", "robot": self.make_robot(), "np": np, "numpy": np, "math": math}
        code_obj = compile(tree, "<program>", "exec")
        linecache.cache["<program>"] = (len(code), None, code.splitlines(True), "<program>")
        buf = io.StringIO()
        status = "program_end"
        remaining = self.cfg.max_wall_time - (time.time() - self.t_wall0)
        with _wall_alarm(remaining, self), contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                try:
                    self._exec_active = True
                    exec(code_obj, ns)
                finally:
                    self._exec_active = False
            except ProgramDone:
                status = "done"
            except BudgetExceeded as e:
                status = e.reason
            except SystemExit:
                status = "program_end"
            except KeyboardInterrupt:
                raise
            except BaseException as e:     # the program's own exception (including RobotError): the episode ends as a failure
                status = "program_error"
                self.program_error = f"{type(e).__name__}: {e}"
                self.traceback_full = traceback.format_exc()
                self.traceback_program = _program_traceback(e)
        if self._halt_reason is not None:            # a swallowed stop signal still counts
            status = self._halt_reason
        self.stdout_text = buf.getvalue()
        return status

    def run(self) -> Dict[str, Any]:
        cfg = self.cfg
        self.t_wall0 = time.time()
        stop_reason, error = None, None
        try:
            self.messages = [{"role": "system", "text": self.system_prompt()},
                             {"role": "user", "parts": [{"type": "text", "text": self.intro_text()}]}]
        except Exception as e:
            stop_reason, error = "exception", f"prompt error: {type(e).__name__}: {e}"
        code = self.program
        if stop_reason is None and code is None:
            if self.client is None:
                stop_reason, error = "agent_error", "agent error: no program and no model client"
            else:
                try:
                    resp: LLMResponse = self.client.chat(self.messages, [])
                except Exception as e:
                    stop_reason, error = "agent_error", f"agent error: {e}"
                else:
                    self.n_calls += 1
                    self.usage_in += int(resp.usage_in or 0)
                    self.usage_out += int(resp.usage_out or 0)
                    self.llm_latency = float(resp.latency_s or 0.0)
                    self.messages.append({"role": "assistant", "text": resp.text, "tool_calls": []})
                    code, self.n_code_blocks = extract_program(resp.text)
                    if code is None:
                        stop_reason = "no_program"
                        self.program_error = "no python code block found in the model reply"
                        self.traceback_program = self.program_error
        self.program = code
        if stop_reason is None:
            stop_reason = self._execute(code)
        try:
            self.final_scene = _plain(self.sim.scene())
        except Exception as e:
            self.final_scene = dict(error=f"{type(e).__name__}: {e}")
        success = False
        try:
            success = bool(self.sim.success())
        except Exception as e:
            error = (error or "") + f" success_check_error: {e}"
        truth = dict(self.sim.task_state)
        verdict_correct = None
        if "label" in truth and self.verdict:
            verdict_correct = self.verdict.strip().lower().startswith(truth["label"])
        cost = self.usage_in / 1e6 * cfg.price_in_per_m + self.usage_out / 1e6 * cfg.price_out_per_m
        rec = dict(
            episode_id=cfg.episode_id, condition=cfg.condition, task=cfg.task, model=cfg.model, fmt=cfg.fmt, vision=cfg.vision,
            interface=cfg.interface, seed=cfg.seed, agent="program", harness=self.harness, phase=self.phase,
            success=success, claimed_success=self.claimed, verdict=self.verdict, verdict_correct=verdict_correct, truth=truth,
            n_model_calls=self.n_calls, n_tool_calls=self.n_tool_calls, tokens_in=self.usage_in, tokens_out=self.usage_out,
            tokens_cache_read=self.cache_read, tokens_cache_write=self.cache_write,
            cost_usd=round(cost, 4), sim_time_s=round(self.sim.sim_time, 2), wall_time_s=round(time.time() - self.t_wall0, 1),
            peak_wrench_N=round(float(self.sim.peak_wrench), 2), peak_object_contact_force_N=round(float(self.sim.peak_contact_force), 2),
            n_events=sum(len(r.events) for r in self.reports), n_force_cap=sum(r.termination in ("force_cap", "torque_cap") for r in self.reports),
            qa_asked=0, qa_correct=0, stop_reason=stop_reason, error=error, note=self.note, pregrasp=self.pregrasp,
            program=code, program_sha1=program_sha1(code), program_error=self.program_error, traceback=self.traceback_full,
            program_stdout=_tail(self.stdout_text, 4000), n_code_blocks=self.n_code_blocks, llm_latency_s=round(self.llm_latency, 2),
            cfg=dataclasses.asdict(cfg),
        )
        self.record = rec
        self._save(rec)
        return rec

    def _save(self, rec: Dict[str, Any]) -> None:
        super()._save(rec)
        if self.program:
            (self.out / "program.py").write_text(self.program, encoding="utf-8")
        (self.out / "program_stdout.txt").write_text(self.stdout_text or "", encoding="utf-8")
        (self.out / "visible.json").write_text(json.dumps(self.visible, default=_json_safe), encoding="utf-8")


# ============================================================================= fake model (for tests: no network, no key)
class FakeProgramLLM:
    """Return a prewritten program for (task, interface) from a directory, wrapped in a python code block. Call k looks in
    each directory for <task>__<interface>__v<k>.py, then <task>__<interface>.py, then <task>.py; a given program_file is
    always used. The last user message received is kept in inbox."""
    kind = "fake"

    def __init__(self, task: str, interface: str, program_dirs=None, program_file: Optional[str] = None):
        self.task, self.interface = task, interface
        dirs = [program_dirs] if isinstance(program_dirs, (str, Path)) else list(program_dirs or [])
        self.dirs = [self._resolve(d) for d in dirs]
        self.program_file = self._resolve(program_file) if program_file else None
        self.n_calls = 0
        self.inbox: List[str] = []

    @staticmethod
    def _resolve(p) -> Path:
        p = Path(p)
        return p if p.is_absolute() else ROOT / p

    def _pick(self) -> Tuple[str, Path]:
        if self.program_file is not None:
            return self.program_file.read_text(encoding="utf-8"), self.program_file
        for d in self.dirs:
            for name in (f"{self.task}__{self.interface}__v{self.n_calls}.py", f"{self.task}__{self.interface}.py", f"{self.task}.py"):
                f = d / name
                if f.exists():
                    return f.read_text(encoding="utf-8"), f
        raise FileNotFoundError(f"fake program for {self.task}/{self.interface} (call {self.n_calls}) not found in {self.dirs}")

    def chat(self, messages: List[Dict[str, Any]], tools) -> LLMResponse:
        self.n_calls += 1
        last_user = next((m for m in reversed(messages) if m["role"] == "user"), {"parts": []})
        self.inbox.append("\n".join(p.get("text", "") for p in last_user.get("parts", []) if p.get("type") == "text"))
        code, path = self._pick()
        reply = f"Program ({path.name}):\n```python\n{code.rstrip()}\n```"
        n_in = 0
        for m in messages:
            n_in += len(m.get("text") or "")
            n_in += sum(len(p.get("text", "")) for p in m.get("parts", []) if p.get("type") == "text")
        return LLMResponse(text=reply, tool_calls=[], usage_in=n_in // 4, usage_out=len(reply) // 4, latency_s=0.0)


def make_program_client(model: Dict[str, Any], cfg: EpisodeConfig):
    """provider "fake" -> FakeProgramLLM (a directory or list of directories in model.fake_programs, or the file
    model.fake_program); otherwise an LLMClient built as in runner._make_agent."""
    if model.get("provider") == "fake":
        return FakeProgramLLM(cfg.task, cfg.interface, program_dirs=model.get("fake_programs"), program_file=model.get("fake_program"))
    from forceharness.eval.runner import _make_agent

    return _make_agent("llm", cfg, model).client


def run_program_episode(sim, cfg: EpisodeConfig, model: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point of eval/runner.py::run_one when model.harness == "cap" (one model call per episode)."""
    harness = model.get("harness")
    if harness != "cap":
        raise ValueError(f"unknown harness {harness!r}; only 'cap' (Code-as-Policies) is supported")
    return ProgramEpisode(sim, make_program_client(model, cfg), cfg, harness=harness).run()
