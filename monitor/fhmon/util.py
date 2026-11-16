"""File helpers shared by the adapters, the bridge and the server."""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def read_json(path, retries: int = 3) -> Optional[Any]:
    """Read JSON another process may be rewriting; None when missing or still torn."""
    for attempt in range(retries):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            if attempt == retries - 1:
                return None
            time.sleep(0.05)
    return None


def write_json(path, data) -> None:
    """Atomic write: a reader sees the old file or the new one, never half of it."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path, row) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path) -> list:
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        pass
    return rows


def read_text(path, limit: int = 4_000_000) -> Optional[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read(limit)
    except OSError:
        return None


def signature(path) -> Optional[tuple]:
    """(mtime_ns, size): changes whenever another process rewrites the file."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def mtime(path) -> Optional[float]:
    try:
        return os.stat(path).st_mtime
    except OSError:
        return None


def iso_to_epoch(text) -> Optional[float]:
    if not isinstance(text, str):
        return None
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


_SLUG = re.compile(r"[^A-Za-z0-9_.-]+")


def slug(text: str, limit: int = 60) -> str:
    return _SLUG.sub("-", text or "").strip("-.")[:limit]


def is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def finite(value) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if v == v and v not in (float("inf"), float("-inf")) else None
