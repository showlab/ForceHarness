"""Operator labels: a hand-set success flag and a note per episode.

Kept by the monitor in <data_dir>/labels.json, never in episode.json (the harness score command
refuses to overwrite). A label's success, when set, wins over the harness score everywhere the
monitor shows or counts results; clearing it falls back to the harness score.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from .util import now_iso, read_json, write_json

NOTE_MAX = 4000


class Labels:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.Lock()
        data = read_json(self.path)
        self.items: dict = data if isinstance(data, dict) else {}
        self.revs: dict = {}        # key -> save count; part of the episode fingerprint so caches refresh

    def get(self, key: str) -> Optional[dict]:
        with self.lock:
            lab = self.items.get(key)
            return dict(lab) if lab else None

    def stamp(self, key: str) -> int:
        with self.lock:
            return self.revs.get(key, 0)

    def set(self, key: str, success, note: str) -> Optional[dict]:
        if not (success is None or isinstance(success, bool)):
            raise ValueError("success must be true, false or null")
        note = (note or "").strip()
        if len(note) > NOTE_MAX:
            raise ValueError(f"Note is too long (at most {NOTE_MAX} characters)")
        with self.lock:
            self.revs[key] = self.revs.get(key, 0) + 1
            if success is None and not note:
                self.items.pop(key, None)
                lab = None
            else:
                lab = dict(success=success, note=note, updated=now_iso())
                self.items[key] = lab
            write_json(self.path, self.items)
            return dict(lab) if lab else None

    def apply(self, ep: dict) -> dict:
        lab = self.get(ep.get("key"))
        r = ep.get("result")
        if r is None:
            r = ep["result"] = {}
        r["label"] = lab
        if lab and lab.get("success") is not None:
            r["harness"] = dict(scored=r.get("scored"), success=r.get("success"), outcome=r.get("outcome"))
            r.update(scored=True, success=lab["success"], manual=True)
        return ep
