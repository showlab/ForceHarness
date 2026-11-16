"""Episode index, per-episode watchers, the event hub behind the browser's live stream, and the
harness-vs-baseline comparison (success rate with a 95% Wilson interval)."""
from __future__ import annotations

import math
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

from . import episodes, streams
from .util import read_json, signature


class Hub:
    """Fan-out of events per episode key; slow browsers drop events instead of blocking."""

    def __init__(self):
        self.lock = threading.Lock()
        self.subs: Dict[str, set] = {}

    def subscribe(self, key: str) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=4000)
        with self.lock:
            self.subs.setdefault(key, set()).add(q)
        return q

    def unsubscribe(self, key: str, q: queue.Queue) -> None:
        with self.lock:
            self.subs.get(key, set()).discard(q)

    def count(self, key: str) -> int:
        with self.lock:
            return len(self.subs.get(key, ()))

    def publish(self, key: str, event: dict) -> None:
        with self.lock:
            targets = list(self.subs.get(key, ()))
        for q in targets:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass


class Store:
    """Finds episodes under the roots and loads them through the adapters, with caching."""

    PATTERNS = ("*/episode.json", "*/*/episode.json", "*/*/*/episode.json")

    def __init__(self, roots: List[Path]):
        self.paths = episodes.Paths(roots)
        self.lock = threading.Lock()
        self._summary: Dict[str, tuple] = {}       # key -> (fingerprint, summary)
        self._version: Dict[str, tuple] = {}       # episode.json path -> (signature, adapter or None)
        self.bridge_log = None                     # key -> bridge.jsonl of an attached bridge (set by the server)
        self.labels = None                         # operator success flag + note (labels.Labels, set by the server)

    def ep_dir(self, key: str) -> Optional[Path]:
        p = self.paths.resolve(key)
        return p if p and (p / "episode.json").is_file() else None

    def key_of(self, ep_dir: Path) -> Optional[str]:
        return self.paths.rel(ep_dir)

    def adapter(self, ep_dir: Path) -> Optional[episodes.Adapter]:
        path = ep_dir / "episode.json"
        sig = signature(path)
        cached = self._version.get(str(path))
        if cached and cached[0] == sig:
            return cached[1]
        adapter = episodes.adapter_for(read_json(path))
        self._version[str(path)] = (sig, adapter)
        return adapter

    def load(self, key: str, with_text: bool = True) -> Optional[dict]:
        ep_dir = self.ep_dir(key)
        if not ep_dir:
            return None
        adapter = self.adapter(ep_dir)
        if not adapter:
            return None
        ep = adapter.load(ep_dir, key, self.paths, with_text,
                          bridge_log=self.bridge_log(key) if self.bridge_log else None)
        return self.labels.apply(ep) if self.labels else ep

    def fingerprint(self, key: str) -> Optional[tuple]:
        ep_dir = self.ep_dir(key)
        adapter = self.adapter(ep_dir) if ep_dir else None
        if not adapter:
            return None
        return adapter.fingerprint(ep_dir) + ((self.labels.stamp(key),) if self.labels else ())

    def scan(self) -> List[dict]:
        out = []
        seen = set()
        for root in self.paths.roots:
            for pattern in self.PATTERNS:
                for path in root.glob(pattern):
                    ep_dir = path.parent
                    if ep_dir in seen:
                        continue
                    seen.add(ep_dir)
                    key = self.key_of(ep_dir)
                    if not key or not self.adapter(ep_dir):
                        continue
                    summary = self.summary(key)
                    if summary:
                        out.append(summary)
        out.sort(key=lambda s: s.get("started_epoch") or 0, reverse=True)
        return out

    def summary(self, key: str) -> Optional[dict]:
        fp = self.fingerprint(key)
        with self.lock:
            cached = self._summary.get(key)
        if cached and cached[0] == fp and cached[1].get("status") == "ended":
            return cached[1]
        ep = self.load(key, with_text=False)
        if ep is None:
            return None
        s = {k: v for k, v in ep.items() if k not in ("turns", "plan", "config")}
        s["turn_count"] = len(ep.get("turns") or [])
        s["peak_force"] = (ep["metrics"].get("peak_raw_force_N") if ep["metrics"].get("peak_raw_force_N") is not None
                           else ep["metrics"].get("peak_force_N"))
        s["note"] = (ep.get("result") or {}).get("model_note")
        with self.lock:
            self._summary[key] = (fp, s)
        return s


class Watcher:
    """Follows one episode: publishes a snapshot when its files change and the force stream as it grows."""

    def __init__(self, store: Store, hub: Hub, key: str):
        self.store, self.hub, self.key = store, hub, key
        self.recent: deque = deque(maxlen=25 * 180)          # ~3 min of points for late joiners
        self.tail: Optional[streams.StreamTail] = None
        self.last_fp = None
        self.snapshot: Optional[dict] = None
        self.idle_since: Optional[float] = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"watch:{key}", daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _ensure_tail(self, ep: dict) -> None:
        # finished episodes are drawn from the whole-file overview instead
        if self.tail is not None or not ep or ep.get("status") in ("ended", "stale"):
            return
        stream = ep.get("stream") or {}
        if not stream.get("file"):
            return
        path = self.store.paths.resolve(stream["file"])
        if path:
            self.tail = streams.StreamTail(path, stream["kind"], ep.get("started_epoch"))

    def _run(self):
        while not self.stopped.is_set():
            try:
                fp = self.store.fingerprint(self.key)
                if fp != self.last_fp:
                    self.last_fp = fp
                    ep = self.store.load(self.key)
                    if ep is not None:
                        self.snapshot = ep
                        self.hub.publish(self.key, dict(type="snapshot", episode=ep))
                if self.snapshot:
                    self._ensure_tail(self.snapshot)
                if self.tail is not None:
                    pts = self.tail.poll()
                    if pts:
                        self.recent.extend(pts)
                        event = dict(type="force", points=pts, clock=self.tail.clock_info())
                        if self.tail.phases_changed:
                            self.tail.phases_changed = False
                            event["phases"] = self.tail.phases[-200:]
                        self.hub.publish(self.key, event)
            except Exception as exc:  # keep watching; report once per distinct error
                self.hub.publish(self.key, dict(type="warning", message=f"watcher: {type(exc).__name__}: {exc}"))
            if self.hub.count(self.key) == 0:
                self.idle_since = self.idle_since or time.time()
                if time.time() - self.idle_since > 30:
                    self.stopped.set()
                    break
            else:
                self.idle_since = None
            self.stopped.wait(0.3)

    def initial_events(self) -> List[dict]:
        events = []
        if self.snapshot is None:
            ep = self.store.load(self.key)
            if ep is not None:
                self.snapshot = ep
        if self.snapshot is not None:
            events.append(dict(type="snapshot", episode=self.snapshot))
        if self.recent:
            events.append(dict(type="force", points=list(self.recent), reset=True,
                               clock=self.tail.clock_info() if self.tail else None,
                               phases=self.tail.phases[-200:] if self.tail else None))
        return events


class Watchers:
    def __init__(self, store: Store, hub: Hub):
        self.store, self.hub = store, hub
        self.lock = threading.Lock()
        self.items: Dict[str, Watcher] = {}

    def ensure(self, key: str) -> Watcher:
        with self.lock:
            w = self.items.get(key)
            if w is None or w.stopped.is_set():
                w = Watcher(self.store, self.hub, key).start()
                self.items[key] = w
            return w


def wilson(k: int, n: int, z: float = 1.96):
    if n <= 0:
        return None, None, None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def compare(summaries: List[dict], family: str, mode: str = "all", instruction: Optional[str] = None) -> dict:
    rows = [s for s in summaries if s.get("family") == family and (mode == "all" or s.get("mode") == mode)]
    # free task: A and B are only comparable on the same task text, so the page picks one
    texts: Dict[str, int] = {}
    for s in rows:
        if s.get("task_text"):
            texts[s["task_text"]] = texts.get(s["task_text"], 0) + 1
    if instruction:
        rows = [s for s in rows if s.get("task_text") == instruction]
    arms = {}
    for arm in ("harness", "baseline"):
        group = [s for s in rows if s.get("arm") == arm]
        # a hand-set success flag counts even when the harness marked the episode faulty or invalid
        ended = [s for s in group if s.get("status") == "ended" or (s.get("result") or {}).get("manual")]
        valid = [s for s in ended if (s.get("valid") is not False and not s.get("fault"))
                 or (s.get("result") or {}).get("manual")]
        scored = [s for s in valid if (s.get("result") or {}).get("scored")]
        wins = [s for s in scored if (s.get("result") or {}).get("success")]
        rate, lo, hi = wilson(len(wins), len(scored))
        acts = [s["metrics"]["actions"] for s in valid if s.get("metrics")]
        durs = [s["metrics"]["duration_s"] for s in valid if (s.get("metrics") or {}).get("duration_s")]
        peaks = sorted(s["peak_force"] for s in valid if isinstance(s.get("peak_force"), (int, float)))
        invalid_actions = sum((s.get("metrics") or {}).get("invalid") or 0 for s in valid)
        total_actions = sum(acts)
        claims = [s for s in valid if (s.get("result") or {}).get("model_claim") is not None]
        arms[arm] = dict(
            n=len(group), n_ended=len(ended), n_valid=len(valid), n_scored=len(scored), n_success=len(wins),
            success_rate=rate, ci_low=lo, ci_high=hi,
            mean_actions=_mean(acts), mean_duration_s=_mean(durs),
            median_peak_force=peaks[len(peaks) // 2] if peaks else None,
            invalid_action_rate=(invalid_actions / total_actions) if total_actions else None,
            model_claim_rate=(sum(1 for s in claims if s["result"]["model_claim"]) / len(claims)) if claims else None)
    return dict(family=family, mode=mode, arms=arms, episodes=rows, instruction=instruction or None,
                instructions=[dict(text=k, n=v) for k, v in sorted(texts.items(), key=lambda kv: -kv[1])])


def _mean(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None
