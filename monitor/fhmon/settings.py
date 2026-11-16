"""Model API profiles.

Profiles (no secrets) are saved in <data>/settings.json. A key either lives only in this
process's memory (typed in the browser, gone on restart) or comes from an environment
variable the profile names, e.g. FH_API_KEY, which `python -m fhmon --env-file .env` (and
monitor/run.sh) load from the repository's .env -- the one place where keys may live on disk.
"""
from __future__ import annotations

import copy
import json
import os
import re
import threading
from pathlib import Path
from typing import Optional

from . import codex_cli, llm
from .receipts import PRICE_PLACEHOLDER
from .util import read_json, read_text, slug, write_json

EFFORTS = ("", "none", "minimal", "low", "medium", "high", "xhigh")
KEY_ENV_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}(KEY|TOKEN)$")
# The harness price table, relative to the repository root ($ per 1M tokens).
PRICE_TABLE = Path("forceharness", "eval", "prices.yaml")
# Its lines look like:  gpt-6-astra:    {in: 10.0, out: 50.0}   # comment
PRICE_LINE = re.compile(r"^([A-Za-z0-9][\w.\-]*)\s*:\s*\{\s*in\s*:\s*([\d.]+)\s*,\s*out\s*:\s*([\d.]+)\s*\}(.*)$")
CLAUDE_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-1")


def load_prices(path: Path) -> dict:
    """{name: (in, out, placeholder)} from the harness price table ($ per 1M tokens, longest prefix wins)."""
    prices = {}
    for line in (read_text(path) or "").splitlines():
        m = PRICE_LINE.match(line.strip())
        if m:
            prices[m.group(1)] = (float(m.group(2)), float(m.group(3)), PRICE_PLACEHOLDER in m.group(4).lower())
    return prices


def price_for(model: str, prices: dict):
    """(in, out) by longest prefix; None when unknown or marked as a placeholder in the table."""
    best = max((k for k in prices if model.startswith(k)), key=len, default=None)
    return prices[best][:2] if best and not prices[best][2] else None


def presets(provider: str, prices: dict) -> list:
    """Common model names for the dropdown: Claude ids for Anthropic, the price table's own names otherwise."""
    if provider == "anthropic":
        names = list(CLAUDE_MODELS)
    else:
        names = [k for k, v in prices.items() if not k.startswith("claude") and not v[2]]
    out = []
    for name in names:
        pr = price_for(name, prices)
        out.append(dict(id=name, label=name, price_in=pr[0] if pr else None, price_out=pr[1] if pr else None))
    return out


def default_profiles() -> list:
    return [
        dict(id="openai", name="OpenAI compatible", provider="openai",
             base_url=os.environ.get("FH_API_BASE") or llm.DEFAULT_BASE["openai"], model="",
             key_env="FH_API_KEY", max_tokens=4096, temperature=None, reasoning_effort="",
             image_turns=2, image_detail="auto", price_in_per_m=None, price_out_per_m=None, extra_body={}),
        dict(id="anthropic", name="Anthropic", provider="anthropic", base_url=llm.DEFAULT_BASE["anthropic"],
             model="claude-opus-5-5", key_env="ANTHROPIC_API_KEY", max_tokens=4096, temperature=None,
             reasoning_effort="", image_turns=2, image_detail="auto", price_in_per_m=None, price_out_per_m=None,
             extra_body={}),
        codex_profile(),
    ]


def codex_profile() -> dict:
    """The local Codex CLI on the ChatGPT plan (no key, no per-token price)."""
    return dict(id="codex", name="Codex (ChatGPT subscription)", provider="codex-cli", base_url=llm.DEFAULT_BASE["codex-cli"],
                model="gpt-6-astra", key_env="", max_tokens=4096, temperature=None, reasoning_effort="",
                image_turns=2, image_detail="auto", price_in_per_m=None, price_out_per_m=None, extra_body={})


class SettingsError(ValueError):
    pass


class Settings:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.keys: dict = {}                       # profile id -> key, memory only
        data = read_json(self.path)
        profiles = data.get("profiles") if isinstance(data, dict) else None
        self.profiles = profiles if isinstance(profiles, list) and profiles else default_profiles()
        if not any(isinstance(p, dict) and p.get("provider") == "codex-cli" for p in self.profiles):
            self.profiles.append(codex_profile())      # settings saved before the Codex option existed

    # -- read ----------------------------------------------------------------
    def get(self, pid: str) -> Optional[dict]:
        with self.lock:
            for p in self.profiles:
                if p.get("id") == pid:
                    return copy.deepcopy(p)
        return None

    def key_for(self, profile: dict) -> Optional[str]:
        if profile.get("provider") in llm.KEYLESS:
            return ""
        return self.keys.get(profile.get("id")) or (os.environ.get(profile.get("key_env") or "") or None)

    def key_source(self, profile: dict) -> Optional[str]:
        if profile.get("provider") in llm.KEYLESS:
            return "codex-login" if codex_cli.available() else None
        if self.keys.get(profile.get("id")):
            return "memory"
        env = profile.get("key_env") or ""
        return f"env:{env}" if env and os.environ.get(env) else None

    def public(self) -> list:
        with self.lock:
            out = []
            for p in self.profiles:
                item = copy.deepcopy(p)
                item["key_source"] = self.key_source(p)
                item["has_key"] = item["key_source"] is not None
                out.append(item)
            return out

    # -- write ---------------------------------------------------------------
    def upsert(self, data: dict, key: Optional[str] = None, clear_key: bool = False) -> dict:
        """Save a profile. Returns it plus key_dropped: a key kept in memory is forgotten when the
        interface or Base URL changes, so it is never sent to a host it was not typed for."""
        is_new = not str((data or {}).get("id") or "").strip()
        profile = self._validate(data)
        dropped = False
        with self.lock:
            if is_new:                      # a new profile never replaces an existing one
                taken = {p.get("id") for p in self.profiles}
                base, n = profile["id"], 2
                while profile["id"] in taken:
                    profile["id"] = f"{base}-{n}"
                    n += 1
            for i, p in enumerate(self.profiles):
                if p.get("id") == profile["id"]:
                    moved = (p.get("provider") != profile["provider"]
                             or (p.get("base_url") or "").rstrip("/") != profile["base_url"].rstrip("/"))
                    if moved and not key and profile["id"] in self.keys:
                        self.keys.pop(profile["id"])
                        dropped = True
                    self.profiles[i] = profile
                    break
            else:
                self.profiles.append(profile)
            if clear_key:
                self.keys.pop(profile["id"], None)
            if key:
                self.keys[profile["id"]] = key.strip()
            self._save()
        return dict(profile, key_dropped=dropped)

    def key_for_host(self, stored: Optional[dict], provider: str, base_url: str,
                     typed: str = "", key_env: str = "") -> tuple:
        """(key, source) for an unsaved form: a key typed now, else the env var the form names,
        else the stored memory key -- the latter only if the form still points at the same host."""
        if typed.strip():
            return typed.strip(), "typed"
        env = key_env.strip()
        if env and KEY_ENV_RE.match(env) and os.environ.get(env):
            return os.environ[env], f"env:{env}"
        if stored and stored.get("provider") == provider and \
                (stored.get("base_url") or "").rstrip("/") == base_url.rstrip("/") and self.keys.get(stored.get("id")):
            return self.keys[stored["id"]], "memory"
        return None, None

    def delete(self, pid: str) -> None:
        with self.lock:
            self.profiles = [p for p in self.profiles if p.get("id") != pid]
            self.keys.pop(pid, None)
            self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.path, {"profiles": self.profiles, "_note": "no API keys are stored here"})

    @staticmethod
    def _validate(d: dict) -> dict:
        if not isinstance(d, dict):
            raise SettingsError("profile must be an object")
        name = str(d.get("name") or "").strip()[:60] or "Profile"
        pid = slug(str(d.get("id") or "")) or slug(name.lower()) or "profile"
        provider = d.get("provider") or "openai"
        if provider not in llm.PROVIDERS:
            raise SettingsError(f"provider must be one of {', '.join(llm.PROVIDERS)}")
        base = str(d.get("base_url") or "").strip() or llm.DEFAULT_BASE[provider]
        if not re.match(r"^https?://[^\s]+$", base):
            raise SettingsError("base_url must start with http:// or https://")
        key_env = str(d.get("key_env") or "").strip()
        if key_env and not KEY_ENV_RE.match(key_env):
            raise SettingsError("The environment variable name must be uppercase and end in KEY or TOKEN (for example FH_API_KEY)")
        effort = str(d.get("reasoning_effort") or "")
        if effort not in EFFORTS:
            raise SettingsError(f"reasoning_effort must be one of {', '.join(e or '(empty)' for e in EFFORTS)}")
        extra = d.get("extra_body") or {}
        if isinstance(extra, str):
            try:
                extra = json.loads(extra) if extra.strip() else {}
            except json.JSONDecodeError as exc:
                raise SettingsError(f"extra_body is not valid JSON: {exc}") from None
        if not isinstance(extra, dict) or len(json.dumps(extra)) > 4000:
            raise SettingsError("extra_body must be a JSON object (4 KB at most)")
        return dict(
            id=pid, name=name, provider=provider, base_url=base, model=str(d.get("model") or "").strip()[:200],
            key_env=key_env,
            max_tokens=_num(d.get("max_tokens"), int, 1, 200_000, 4096),
            temperature=_num(d.get("temperature"), float, 0, 2, None),
            reasoning_effort=effort,
            image_turns=_num(d.get("image_turns"), int, 0, 20, 2),
            image_detail=d.get("image_detail") if d.get("image_detail") in ("auto", "low", "high") else "auto",
            price_in_per_m=_num(d.get("price_in_per_m"), float, 0, 10_000, None),
            price_out_per_m=_num(d.get("price_out_per_m"), float, 0, 10_000, None),
            extra_body=extra)


def _num(value, kind, lo, hi, default):
    if value in (None, ""):
        return default
    try:
        v = kind(value)
    except (TypeError, ValueError):
        raise SettingsError(f"Invalid number: {value!r}") from None
    if not lo <= v <= hi:
        raise SettingsError(f"Number out of range [{lo}, {hi}]: {value!r}")
    return v
