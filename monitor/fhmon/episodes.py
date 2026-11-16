"""Episode adapters: the only module that knows the harness's run-folder formats.

An adapter turns one run folder into the monitor's common episode dict (see
`FreeAdapter.load`). Supported record version:
- free_task/1, written by the real-robot runner scripts/real/free_task.py and the fixed tasks built on it
  (scripts/real/<task>.py). The family is the record's task_id: "free" for the free task (the task is the
  operator's instruction), else the fixed task's id from monitor/tasks.json. Every turn photo comes with
  the objects the wrist depth camera detected.
A new fixed task on the same runner needs one FixedTaskAdapter in ADAPTERS and an entry in tasks.json;
a runner with another record format needs its own Adapter subclass.

Turn n always means: obs_n.md (what the model saw) -> cmd_n.json (its decision)
-> actions[n] in episode.json (+ the primitive report action_nn.json) -> receipt (tool result),
which the model reads at the top of obs_{n+1}.md.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import receipts, streams
from .util import iso_to_epoch, mtime, read_json, read_jsonl, read_text, signature

OBS_RE = re.compile(r"obs_(\d+)\.md$")
IMG_RE = re.compile(r"obs_(\d+)_(.+)\.(png|jpg|jpeg)$")
CMD_RE = re.compile(r"cmd_(\d+)\.json$")
STALE_S = 15 * 60


def arm_of(condition: Optional[str]) -> str:
    return "baseline" if str(condition or "").startswith("none") else "harness"


def parse_cmd_text(raw: str) -> Optional[dict]:
    """Same leniency as the runners' relay client: whole JSON, else the outer {...}."""
    raw = (raw or "").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        i, j = raw.find("{"), raw.rfind("}")
        if i < 0 or j <= i:
            return None
        try:
            data = json.loads(raw[i:j + 1])
        except json.JSONDecodeError:
            return None
    if isinstance(data, list):
        return dict(calls=data, note=None)
    if not isinstance(data, dict):
        return None
    calls = data.get("calls")
    if calls is None and "name" in data:
        calls = [dict(name=data["name"], args=data.get("args") or {})]
    return dict(calls=calls if isinstance(calls, list) else [], note=data.get("note"))


def read_relay(relay_dir: Optional[Path]) -> dict:
    out = dict(dir=relay_dir, exists=False, turns={}, pending=None, done=None)
    if relay_dir is None or not relay_dir.is_dir():
        return out
    out["exists"] = True
    turns: Dict[int, dict] = {}
    try:
        names = [p.name for p in relay_dir.iterdir()]
    except OSError:
        names = []
    for name in names:
        m = OBS_RE.match(name)
        if m:
            turns.setdefault(int(m.group(1)), {})["obs"] = relay_dir / name
            continue
        m = IMG_RE.match(name)
        if m:
            turns.setdefault(int(m.group(1)), {}).setdefault("images", []).append(relay_dir / name)
            continue
        m = CMD_RE.match(name)
        if m:
            turns.setdefault(int(m.group(1)), {})["cmd"] = relay_dir / name
    for t in turns.values():
        # wrist first: the prompts describe the wrist overlay before the exterior view
        t["images"] = sorted(t.get("images", []), key=lambda p: ("wrist" not in p.name, p.name))
    out["turns"] = turns
    out["pending"] = read_json(relay_dir / "pending.json")
    out["done"] = read_json(relay_dir / "done.json")
    return out


class Paths:
    """Turns absolute paths into the /api/file URLs the browser may load."""

    def __init__(self, roots: List[Path]):
        self.roots = [r.resolve() for r in roots]

    def rel(self, path: Optional[Path]) -> Optional[str]:
        if path is None:
            return None
        p = Path(path).resolve()
        for i, root in enumerate(self.roots):
            try:
                return f"{i}:{p.relative_to(root).as_posix()}"
            except ValueError:
                continue
        return None

    def url(self, path: Optional[Path]) -> Optional[str]:
        rel = self.rel(path)
        if rel is None:
            return None
        from urllib.parse import quote
        return "/api/file?f=" + quote(rel, safe="")

    def resolve(self, token: str) -> Optional[Path]:
        root_index, _, rel = token.partition(":")
        if not root_index.isdigit() or int(root_index) >= len(self.roots):
            return None
        root = self.roots[int(root_index)]
        p = (root / rel).resolve()
        try:
            p.relative_to(root)
        except ValueError:
            return None
        return p


_report_cache: "OrderedDict[tuple, dict]" = OrderedDict()
_report_lock = threading.Lock()
_relay_cache: Dict[Path, Path] = {}


def _load_report(path: Path) -> Optional[dict]:
    sig = signature(path)
    if sig is None:
        return None
    key = (str(path), sig)
    with _report_lock:
        if key in _report_cache:
            _report_cache.move_to_end(key)
            return _report_cache[key]
    data = read_json(path)
    if not isinstance(data, dict):
        return None
    with _report_lock:
        _report_cache[key] = data
        while len(_report_cache) > 64:
            _report_cache.popitem(last=False)
    return data


def _jsonable_small(value, depth=0):
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, float)):
        return None if isinstance(value, float) and not math.isfinite(value) else value
    if isinstance(value, (list, tuple)):
        if len(value) > 12 or depth > 2:
            return None
        return [_jsonable_small(v, depth + 1) for v in value]
    if isinstance(value, dict) and depth < 2:
        return {str(k): _jsonable_small(v, depth + 1) for k, v in list(value.items())[:24]}
    return None


def report_summary(path: Path) -> Optional[dict]:
    d = _load_report(path)
    if d is None:
        return None
    rows = d.get("wrench_rows") or []
    spark = []
    if rows:
        step = max(1, math.ceil(len(rows) / 48))
        for i in range(0, len(rows), step):
            chunk = rows[i:i + step]
            spark.append(round(max(math.sqrt(r[1] ** 2 + r[2] ** 2 + r[3] ** 2) for r in chunk), 3))
    derived = {k: _jsonable_small(v) for k, v in (d.get("derived") or {}).items()}
    t0, t1 = d.get("t_start"), d.get("t_end")
    return dict(
        primitive=d.get("primitive"), params=_jsonable_small(d.get("params") or {}),
        termination=d.get("termination"),
        duration_s=(t1 - t0) if isinstance(t0, (int, float)) and isinstance(t1, (int, float)) else None,
        start_pos=d.get("start_pos"), end_pos=d.get("end_pos"),
        commanded_disp=d.get("commanded_disp"), actual_disp=d.get("actual_disp"),
        events=d.get("events") or [], peak_force=d.get("peak_force"), peak_torque=d.get("peak_torque"),
        mean_force_in_contact=d.get("mean_force_in_contact"), n_steps=d.get("n_steps"),
        gripper_opening=d.get("gripper_opening"), yaw_deg=d.get("yaw_deg"),
        final_wrench=d.get("final_wrench_net") or d.get("final_wrench"),
        derived={k: v for k, v in derived.items() if v is not None},
        quantities=[q for q in (d.get("quantities") or []) if q.get("name") != "estimator_error"],
        error=d.get("error"), n_rows=len(rows), spark=spark)


def report_rows(path: Path) -> Optional[dict]:
    d = _load_report(path)
    if d is None:
        return None
    rows = d.get("wrench_rows") or []
    pts = []
    for r in rows:
        if len(r) < 7:
            continue
        t, fx, fy, fz, tx, ty, tz = (float(v) for v in r[:7])
        pts.append([round(t, 3), round(fx, 3), round(fy, 3), round(fz, 3),
                    round(math.sqrt(fx * fx + fy * fy + fz * fz), 3),
                    round(math.sqrt(tx * tx + ty * ty + tz * tz), 4)])
    return dict(primitive=d.get("primitive"), termination=d.get("termination"), points=pts,
                pos=d.get("pos_rows") or [], events=d.get("events") or [],
                peak_force=d.get("peak_force"), peak_torque=d.get("peak_torque"))


def read_session(ep_dir: Path) -> Optional[dict]:
    """Monitor-launched episodes live in <data>/<sid>/episode with session.json next to it."""
    meta = read_json(ep_dir.parent / "session.json")
    if isinstance(meta, dict) and meta.get("kind") == "fhmon_session":
        return meta
    return None


def read_bridge(path: Optional[Path]) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    if path is None:
        return out
    for row in read_jsonl(path):
        if isinstance(row, dict) and isinstance(row.get("turn"), int):
            out[row["turn"]] = row
    return out


def pending_confirmations(ep_dir: Path) -> List[dict]:
    folder = ep_dir / "confirmations"
    if not folder.is_dir():
        return []
    out = []
    for req in sorted(folder.glob("*.request.json")):
        stem = req.name[:-len(".request.json")]
        if (folder / f"{stem}.result.json").exists():
            continue
        data = read_json(req) or {}
        ctx = data.get("context") or {}
        out.append(dict(kind=data.get("kind"), expected=data.get("expected"),
                        question=ctx.get("question") or ctx.get("action"), request_id=data.get("request_id")))
    return out


class Adapter:
    family = "?"
    versions: tuple = ()

    def matches(self, record: dict) -> bool:
        return str(record.get("version", "")) in self.versions

    # -- shared pieces -------------------------------------------------------
    def relay_dir(self, ep_dir: Path, record: dict, session: Optional[dict]) -> Optional[Path]:
        if session and session.get("relay_dir"):
            return Path(session["relay_dir"])
        sibling = ep_dir.with_name(ep_dir.name + "_relay")
        return sibling if sibling.is_dir() else None

    def obs(self, rt: dict, paths: Paths, with_text: bool) -> Optional[dict]:
        if not rt:
            return None
        obs_path = rt.get("obs")
        text = read_text(obs_path, 200_000) if (obs_path and with_text) else None
        return dict(name=obs_path.name if obs_path else None, url=paths.url(obs_path), text=text,
                    mtime=mtime(obs_path) if obs_path else None,
                    images=[dict(name=p.name, url=paths.url(p),
                                 label="wrist" if "wrist" in p.name else ("exterior" if "exterior" in p.name else p.stem))
                            for p in rt.get("images", [])])

    def decision(self, rt: dict) -> Optional[dict]:
        cmd_path = rt.get("cmd") if rt else None
        if not cmd_path:
            return None
        parsed = parse_cmd_text(read_text(cmd_path, 200_000) or "")
        if parsed is None:
            return dict(calls=[], note=None, unparsed=True, mtime=mtime(cmd_path))
        parsed["mtime"] = mtime(cmd_path)
        return parsed

    def model_driven(self, ep_dir: Path, record: dict, session: Optional[dict]) -> bool:
        """A scripted demo writes cmd_n before obs_n; a model (runner --demo-model) answers after."""
        relay = read_relay(self.relay_dir(ep_dir, record, session))
        for t in relay["turns"].values():
            if t.get("cmd") and t.get("obs") and (mtime(t["cmd"]) or 0) - (mtime(t["obs"]) or 0) > 0.5:
                return True
        pending = relay.get("pending") or {}
        return bool(pending.get("waiting")) and not record.get("ended")

    def fingerprint(self, ep_dir: Path) -> tuple:
        """Cheap change detector, polled every 0.3 s while a browser watches the episode."""
        session = read_session(ep_dir)
        relay = _relay_cache.get(ep_dir)
        if relay is None:
            relay = self.relay_dir(ep_dir, read_json(ep_dir / "episode.json") or {}, session)
            if relay is not None and relay.is_dir():
                _relay_cache[ep_dir] = relay
        sdir = ep_dir.parent
        return (signature(ep_dir / "episode.json"), mtime(ep_dir), mtime(relay) if relay else None,
                signature(sdir / "bridge.jsonl") if session else None,
                signature(sdir / "session.json") if session else None)

    def base(self, ep_dir: Path, key: str, record: dict, session: Optional[dict]) -> dict:
        started = record.get("started")
        model = record.get("model")
        if session and session.get("model"):
            model_label = session["model"].get("label")
            model_source = session["model"].get("source")
        elif model == "relay":
            model_label, model_source = (record.get("model_name") + " (relay)" if record.get("model_name") else "External relay"), "relay"
        elif model == "synthetic" and self.model_driven(ep_dir, record, session):
            model_label, model_source = "External relay (demo scene)", "relay"
        elif model == "synthetic":
            model_label, model_source = "Scripted demo", "scripted"
        else:
            model_label, model_source = model, "harness"
        return dict(
            key=key, family=self.family, version=record.get("version"), mode=record.get("mode"),
            condition=record.get("condition"), arm=arm_of(record.get("condition")),
            interface=record.get("interface"), model=model_label, model_source=model_source,
            started=started, started_epoch=iso_to_epoch(started), name=(session or {}).get("id") or ep_dir.name,
            session=_public_session(session), config=record.get("config") or {},
            confirmations_pending=pending_confirmations(ep_dir))


def _public_session(session: Optional[dict]) -> Optional[dict]:
    if not session:
        return None
    keep = ("id", "task", "condition", "mode", "instruction", "pair_id", "pair_role", "status",
            "exit_code", "started", "ended", "command", "stopped_by", "model", "name")
    return {k: session.get(k) for k in keep if k in session}


class FreeAdapter(Adapter):
    """Free task (scripts/real/free_task.py, record version free_task/1) and the fixed tasks that run on the
    same runner: relay obs/cmd per turn, actions in episode.json, one primitive report action_nn.json per
    executed action, the force stream, the operator's task text and the objects detected each turn."""
    family = "free"
    versions = ("free_task/1",)
    task_ids = (None, "free")             # fixed tasks set task_id to their own id (FixedTaskAdapter)
    success_outcome = "success"

    def matches(self, record):
        return str(record.get("version", "")) in self.versions and record.get("task_id") in self.task_ids

    def load(self, ep_dir: Path, key: str, paths: Paths, with_text: bool = True,
             bridge_log: Optional[Path] = None) -> dict:
        record = read_json(ep_dir / "episode.json") or {}
        session = read_session(ep_dir)
        relay_dir = self.relay_dir(ep_dir, record, session)
        relay = read_relay(relay_dir)
        bridge = read_bridge(ep_dir.parent / "bridge.jsonl" if session else bridge_log)
        out = self.base(ep_dir, key, record, session)
        started = out["started_epoch"]
        actions = record.get("actions") or []
        ended = bool(record.get("ended"))
        scenes = {u.get("turn"): u for u in record.get("scene_updates") or [] if isinstance(u, dict)}
        n = max(len(actions), (max(relay["turns"]) + 1) if relay["turns"] else 0)
        turns = []
        for i in range(n):
            a = actions[i] if i < len(actions) else None
            rt = relay["turns"].get(i, {})
            rep_path = ep_dir / f"action_{i:02d}.json"
            rep = report_summary(rep_path) if rep_path.exists() else None
            receipt = (a or {}).get("receipt")
            decision = self.decision(rt)
            x0 = (decision["mtime"] - started) if (decision and decision.get("mtime") and started) else None
            x1 = (mtime(rep_path) - started) if (rep and started) else None
            if x0 is not None and x1 is not None and rep.get("duration_s"):
                x0 = max(x0, x1 - rep["duration_s"])
            turn = dict(
                n=i, obs=self.obs(rt, paths, with_text), decision=decision, model=bridge.get(i),
                action=_action(a, ended),
                receipt=receipt,
                translation=receipts.translation(receipt.get("text", "")) if receipt else None,
                report=rep, report_url=f"/api/episode/action?key={_q(key)}&n={i}" if rep else None,
                x0=x0, x1=x1)
            u = scenes.get(i)
            if u:
                turn["scene"] = dict(objects=[{k: o.get(k) for k in ("id", "centre", "top", "height", "length", "width",
                                                                      "axis_deg", "partial", "under_hand", "label")}
                                              for o in u.get("objects") or []],
                                     table_z=u.get("table_z"), synthetic=bool(u.get("synthetic")), error=u.get("error"))
            turns.append(turn)
        status = _status(ended, relay, actions, turns, [ep_dir / "episode.json", relay_dir,
                                                        streams.find_stream(ep_dir)[0]])
        scored = bool(record.get("scored"))
        executed = sum(1 for a in actions if "termination" in a)
        out.update(
            status=status, ended=ended, finished=bool(record.get("finished")),
            fault=record.get("fault") or record.get("backend_fault"),
            valid=record.get("valid_episode"),
            result=dict(scored=scored, outcome=record.get("outcome"),
                        success=(record.get("outcome") == self.success_outcome) if scored else None,
                        model_claim=record.get("model_success"), model_note=record.get("model_note"),
                        budget_stop=record.get("budget_stop"), score_note=record.get("score_note")),
            metrics=dict(actions=len(actions), executed=executed,
                         invalid=sum(1 for a in actions if a.get("invalid")),
                         duration_s=record.get("duration_s") or ((time.time() - started) if (started and not ended) else None),
                         peak_force_N=record.get("peak_force_N"), peak_raw_force_N=record.get("peak_raw_force_N"),
                         decision_latency_s=_mean([e.get("latency_s") for e in record.get("relay_events") or []]
                                                  or [b.get("latency_s") for b in bridge.values()])),
            relay=dict(dir=paths.rel(relay_dir), exists=relay["exists"], pending=relay["pending"],
                       done=relay["done"] is not None),
            turns=turns,
            stream=_stream_info(ep_dir, paths),
            task_text=record.get("instruction"), menu=record.get("menu"),
            limits=dict(max_actions=record.get("max_actions"), max_minutes=record.get("max_minutes")))
        return out


class FixedTaskAdapter(FreeAdapter):
    """A fixed task on the free-task runner: the same record, with task_id naming the task (monitor/tasks.json)."""

    def __init__(self, family: str):
        self.family = family
        self.task_ids = (family,)


ADAPTERS = [FixedTaskAdapter(t) for t in ("blackboard_wipe", "screw_insert", "peel_vegetable", "usb_insert",
                                          "can_weight_sort", "touch_firmness")] + [FreeAdapter()]


def adapter_for(record: Any) -> Optional[Adapter]:
    if not isinstance(record, dict):
        return None
    for a in ADAPTERS:
        if a.matches(record):
            return a
    return None


# -- helpers -----------------------------------------------------------------
def _q(text: str) -> str:
    from urllib.parse import quote
    return quote(text, safe="")


def _mean(values) -> Optional[float]:
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 2) if vals else None


def _action(a: Optional[dict], ended: bool) -> Optional[dict]:
    if not a:
        return None
    if a.get("invalid"):
        status = "invalid"
    elif a.get("declined") is not None:
        status = "declined"
    elif a.get("name") == "done":
        status = "done"
    elif "termination" not in a and "receipt" not in a:
        status = "executing" if not ended else "interrupted"
    else:
        status = "ok"
    return dict(name=a.get("name"), args=a.get("args"), termination=a.get("termination"), status=status,
                invalid=a.get("invalid"), declined=a.get("declined"), ignored_args=a.get("ignored_args"))


def _status(ended: bool, relay: dict, actions: list, turns: list, watch: list) -> str:
    if ended:
        return "ended"
    newest = max([m for m in (mtime(p) for p in watch if p) if m] or [0])
    if newest and time.time() - newest > STALE_S:
        return "stale"
    pending = relay.get("pending") or {}
    turn = pending.get("turn")
    if pending.get("waiting") and isinstance(turn, int) and not (relay["turns"].get(turn) or {}).get("cmd"):
        return "waiting_model"
    if turns and (turns[-1].get("action") or {}).get("status") == "executing":
        return "executing"
    if not relay.get("exists") and not actions:
        return "starting"
    return "running"


def _stream_info(ep_dir: Path, paths: Paths) -> dict:
    path, kind = streams.find_stream(ep_dir)
    return dict(kind=kind, file=paths.rel(path), size=(path.stat().st_size if path else 0))
