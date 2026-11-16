"""Streaming clients for the model APIs the harness already supports
(forceharness/agent/llm.py: OpenAI-compatible chat, OpenAI Responses, Anthropic),
re-implemented on urllib so the monitor needs no SDK and imports nothing from the harness.

Provider-neutral turns:
  [{"role": "user" | "assistant",
    "parts": [{"type": "text", "text": str} | {"type": "image", "mime": str, "b64": str, "name": str}]}]
A key is passed in per call; it is never logged, stored or put into an error message.
"""
from __future__ import annotations

import json
import re
import socket
import time
import urllib.error
import urllib.request
from typing import Callable, Dict, List, Optional

from . import codex_cli

PROVIDERS = {
    "openai": "OpenAI compatible (/chat/completions)",
    "openai-responses": "OpenAI Responses (/responses)",
    "anthropic": "Anthropic Messages (/v1/messages)",
    "codex-cli": "Codex CLI (ChatGPT subscription quota, local login, no key)",
}
DEFAULT_BASE = {
    "openai": "https://api.openai.com/v1",
    "openai-responses": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "codex-cli": "https://chatgpt.com",          # not used: codex talks to its own backend with its login
}
KEYLESS = {"codex-cli"}


class LLMError(Exception):
    pass


class Cancelled(Exception):
    pass


Delta = Optional[Callable[[str, str], None]]


def call(profile: dict, key: str, system: str, turns: List[dict], on_delta: Delta = None,
         cancel=None, timeout: float = 300.0, trace=None) -> Dict[str, object]:
    """One model request. Returns text, thinking, usage_in, usage_out, latency_s, stop_reason.
    `trace`: a file for the codex-cli event stream (other providers ignore it)."""
    provider = profile.get("provider") or "openai"
    if provider not in PROVIDERS:
        raise LLMError(f"Unknown provider: {provider}")
    if not key and provider not in KEYLESS:
        raise LLMError("This profile has no API key (enter it in Settings or provide it in an environment variable)")
    if not profile.get("model"):
        raise LLMError("This profile has no model name")
    if provider == "codex-cli":
        t0 = time.time()
        try:
            result = codex_cli.call(profile, system, turns, on_delta or (lambda kind, text: None), cancel, trace=trace)
        except codex_cli.CodexCancelled:
            raise Cancelled() from None
        except codex_cli.CodexError as exc:
            raise LLMError(str(exc)) from None
        result["latency_s"] = round(time.time() - t0, 2)
        return result
    fn = {"openai": _openai_chat, "openai-responses": _openai_responses, "anthropic": _anthropic}[provider]
    t0 = time.time()
    try:
        result = fn(profile, key, system, turns, on_delta or (lambda kind, text: None), cancel, timeout)
    except LLMError as exc:
        raise LLMError(str(exc).replace(key, "***")) from None
    result["latency_s"] = round(time.time() - t0, 2)
    return result


# -- transport ---------------------------------------------------------------
def _error_message(body: str) -> str:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return body.strip()[:400] or "(empty body)"
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        return str(err.get("message") or err.get("type") or err)[:400]
    if isinstance(err, str):
        return err[:400]
    return str(data)[:400]


def _post(url: str, headers: dict, body: dict, timeout: float):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json", "Accept": "text/event-stream, application/json",
                                          **headers})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read(4000).decode("utf-8", "replace")
        raise LLMError(f"HTTP {exc.code}: {_error_message(detail)}") from None
    except urllib.error.URLError as exc:
        raise LLMError(f"Connection failed: {exc.reason}") from None
    except (socket.timeout, TimeoutError):
        raise LLMError("Connection timed out") from None


def _sse(resp, cancel):
    """Yield (event, data) pairs from a text/event-stream response."""
    event, data = None, []
    try:
        for raw in resp:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if not line:
                if data:
                    yield event, "\n".join(data)
                event, data = None, []
                continue
            if line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "event":
                event = value
            elif field == "data":
                data.append(value)
    except (socket.timeout, TimeoutError):
        raise LLMError("Model stream interrupted (read timed out)") from None
    if data:
        yield event, "\n".join(data)


def _is_stream(resp) -> bool:
    return "text/event-stream" in (resp.headers.get("Content-Type") or "")


def _text_of(turn: dict) -> str:
    return "\n".join(p["text"] for p in turn.get("parts", []) if p.get("type") == "text")


def _base(profile: dict) -> str:
    provider = profile.get("provider") or "openai"
    return (profile.get("base_url") or DEFAULT_BASE[provider]).rstrip("/")


def _apply_common(profile: dict, body: dict) -> dict:
    if profile.get("temperature") not in (None, ""):
        body["temperature"] = float(profile["temperature"])
    extra = profile.get("extra_body") or {}
    if isinstance(extra, dict):
        body.update(extra)
    return body


# -- OpenAI-compatible chat/completions ----------------------------------------
def _openai_chat(profile, key, system, turns, on_delta, cancel, timeout):
    base = _base(profile)
    messages = [{"role": "system", "content": system}] if system else []
    for t in turns:
        if t["role"] == "assistant":
            messages.append({"role": "assistant", "content": _text_of(t) or "(no content)"})
            continue
        content = []
        for p in t["parts"]:
            if p["type"] == "text":
                content.append({"type": "text", "text": p["text"]})
            else:
                content.append({"type": "image_url", "image_url": {
                    "url": f"data:{p['mime']};base64,{p['b64']}", "detail": profile.get("image_detail") or "auto"}})
        messages.append({"role": "user", "content": content})
    body = {"model": profile["model"], "messages": messages, "stream": True, "stream_options": {"include_usage": True}}
    if profile.get("max_tokens"):
        # same rule as the harness client: api.openai.com wants max_completion_tokens
        body["max_completion_tokens" if "api.openai.com" in base else "max_tokens"] = int(profile["max_tokens"])
    if profile.get("reasoning_effort"):
        body["reasoning_effort"] = profile["reasoning_effort"]
    _apply_common(profile, body)
    headers = {"Authorization": f"Bearer {key}"}
    try:
        resp = _post(base + "/chat/completions", headers, body, timeout)
    except LLMError as exc:
        if "stream_options" not in str(exc):
            raise
        body.pop("stream_options", None)
        resp = _post(base + "/chat/completions", headers, body, timeout)
    text, thinking, usage, stop = [], [], {}, None
    with resp:
        if not _is_stream(resp):
            data = json.loads(resp.read().decode("utf-8", "replace") or "{}")
            msg = ((data.get("choices") or [{}])[0].get("message") or {})
            text.append(msg.get("content") or "")
            thinking.append(msg.get("reasoning_content") or "")
            usage = data.get("usage") or {}
            on_delta("text", text[0])
        else:
            for _event, payload in _sse(resp, cancel):
                if payload.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if obj.get("error"):
                    raise LLMError(_error_message(json.dumps(obj)))
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    if isinstance(piece, str) and piece:
                        text.append(piece)
                        on_delta("text", piece)
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if isinstance(reasoning, str) and reasoning:
                        thinking.append(reasoning)
                        on_delta("thinking", reasoning)
                    if choice.get("finish_reason"):
                        stop = choice["finish_reason"]
                if obj.get("usage"):
                    usage = obj["usage"]
    return dict(text="".join(text), thinking="".join(thinking), stop_reason=stop,
                usage_in=usage.get("prompt_tokens"), usage_out=usage.get("completion_tokens"))


# -- OpenAI Responses ---------------------------------------------------------
def _openai_responses(profile, key, system, turns, on_delta, cancel, timeout):
    base = _base(profile)
    items = []
    for t in turns:
        if t["role"] == "assistant":
            items.append({"role": "assistant", "content": [{"type": "output_text", "text": _text_of(t) or "(no content)"}]})
            continue
        content = []
        for p in t["parts"]:
            if p["type"] == "text":
                content.append({"type": "input_text", "text": p["text"]})
            else:
                content.append({"type": "input_image", "image_url": f"data:{p['mime']};base64,{p['b64']}"})
        items.append({"role": "user", "content": content})
    body = {"model": profile["model"], "input": items, "stream": True, "store": False}
    if system:
        body["instructions"] = system
    if profile.get("max_tokens"):
        body["max_output_tokens"] = int(profile["max_tokens"])
    if profile.get("reasoning_effort"):
        body["reasoning"] = {"effort": profile["reasoning_effort"], "summary": "auto"}
    _apply_common(profile, body)
    resp = _post(base + "/responses", {"Authorization": f"Bearer {key}"}, body, timeout)
    text, thinking, usage, stop = [], [], {}, None
    with resp:
        if not _is_stream(resp):
            data = json.loads(resp.read().decode("utf-8", "replace") or "{}")
            for item in data.get("output") or []:
                for c in item.get("content") or []:
                    if c.get("type") == "output_text":
                        text.append(c.get("text") or "")
            usage = data.get("usage") or {}
            on_delta("text", "".join(text))
        else:
            for _event, payload in _sse(resp, cancel):
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                kind = obj.get("type", "")
                if kind == "response.output_text.delta":
                    text.append(obj.get("delta") or "")
                    on_delta("text", obj.get("delta") or "")
                elif kind in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
                    thinking.append(obj.get("delta") or "")
                    on_delta("thinking", obj.get("delta") or "")
                elif kind == "response.completed":
                    response = obj.get("response") or {}
                    usage = response.get("usage") or {}
                    stop = response.get("status")
                elif kind in ("response.failed", "response.incomplete"):
                    response = obj.get("response") or {}
                    err = response.get("error") or response.get("incomplete_details") or {}
                    if kind == "response.failed":
                        raise LLMError(str(err.get("message") or err or "response failed"))
                    stop = "incomplete"
                elif kind == "error":
                    raise LLMError(str(obj.get("message") or obj))
    return dict(text="".join(text), thinking="".join(thinking), stop_reason=stop,
                usage_in=usage.get("input_tokens"), usage_out=usage.get("output_tokens"))


# -- Anthropic Messages ---------------------------------------------------------
def _anthropic(profile, key, system, turns, on_delta, cancel, timeout):
    base = _base(profile)
    url = base + ("/messages" if base.endswith("/v1") else "/v1/messages")
    messages = []
    for t in turns:
        if t["role"] == "assistant":
            messages.append({"role": "assistant", "content": [{"type": "text", "text": _text_of(t) or "(no content)"}]})
            continue
        blocks = []
        for p in t["parts"]:
            if p["type"] == "text":
                blocks.append({"type": "text", "text": p["text"]})
            else:
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": p["mime"], "data": p["b64"]}})
        messages.append({"role": "user", "content": blocks})
    body = {"model": profile["model"], "max_tokens": int(profile.get("max_tokens") or 4096),
            "messages": messages, "stream": True}
    if system:
        body["system"] = system
    _apply_common(profile, body)
    resp = _post(url, {"x-api-key": key, "anthropic-version": "2023-06-01"}, body, timeout)
    text, thinking, usage_in, usage_out, stop = [], [], None, None, None
    with resp:
        if not _is_stream(resp):
            data = json.loads(resp.read().decode("utf-8", "replace") or "{}")
            for block in data.get("content") or []:
                if block.get("type") == "text":
                    text.append(block.get("text") or "")
                elif block.get("type") == "thinking":
                    thinking.append(block.get("thinking") or "")
            usage = data.get("usage") or {}
            usage_in, usage_out, stop = usage.get("input_tokens"), usage.get("output_tokens"), data.get("stop_reason")
            on_delta("text", "".join(text))
        else:
            for _event, payload in _sse(resp, cancel):
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                kind = obj.get("type")
                if kind == "message_start":
                    usage_in = ((obj.get("message") or {}).get("usage") or {}).get("input_tokens")
                elif kind == "content_block_delta":
                    delta = obj.get("delta") or {}
                    if delta.get("type") == "text_delta":
                        text.append(delta.get("text") or "")
                        on_delta("text", delta.get("text") or "")
                    elif delta.get("type") == "thinking_delta":
                        thinking.append(delta.get("thinking") or "")
                        on_delta("thinking", delta.get("thinking") or "")
                elif kind == "message_delta":
                    usage_out = (obj.get("usage") or {}).get("output_tokens", usage_out)
                    stop = (obj.get("delta") or {}).get("stop_reason", stop)
                elif kind == "error":
                    err = obj.get("error") or {}
                    raise LLMError(str(err.get("message") or err))
    return dict(text="".join(text), thinking="".join(thinking), stop_reason=stop,
                usage_in=usage_in, usage_out=usage_out)


# -- model lists ----------------------------------------------------------------------
# Not chat models: embeddings, speech, images, moderation, legacy completions.
NOT_CHAT = re.compile(r"embed|tts|whisper|dall-e|moderation|davinci|babbage|transcrib|audio|realtime|image|sora", re.I)


def list_models(profile: dict, key: str, timeout: float = 20.0) -> List[dict]:
    """Models this key can use, newest first: [{id, label, created}]. Listing costs no tokens."""
    provider = profile.get("provider") or "openai"
    if provider not in PROVIDERS:
        raise LLMError(f"Unknown provider: {provider}")
    if provider == "codex-cli":
        return codex_cli.models()
    if not key:
        raise LLMError("No API key")
    base = _base(profile)
    if provider == "anthropic":
        url = base + ("/models" if base.endswith("/v1") else "/v1/models") + "?limit=1000"
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    else:
        url, headers = base + "/models", {"Authorization": f"Bearer {key}"}
    req = urllib.request.Request(url, headers={"Accept": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace") or "{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read(4000).decode("utf-8", "replace")
        raise LLMError(f"HTTP {exc.code}: {_error_message(detail)}".replace(key, "***")) from None
    except urllib.error.URLError as exc:
        raise LLMError(f"Connection failed: {exc.reason}") from None
    except (socket.timeout, TimeoutError):
        raise LLMError("Connection timed out") from None
    except json.JSONDecodeError:
        raise LLMError("The model list is not JSON") from None
    if isinstance(data, dict):
        items = data.get("data") or data.get("models") or []
    else:
        items = data if isinstance(data, list) else []
    out = []
    for m in items:
        if not isinstance(m, dict):
            continue
        mid = str(m.get("id") or m.get("name") or "").removeprefix("models/")
        if not mid or (provider != "anthropic" and NOT_CHAT.search(mid)):
            continue
        created = m.get("created")
        if created is None and isinstance(m.get("created_at"), str):
            from datetime import datetime
            try:
                created = datetime.fromisoformat(m["created_at"].replace("Z", "+00:00")).timestamp()
            except ValueError:
                created = None
        label = m.get("display_name")
        out.append(dict(id=mid, label=f"{mid} ({label})" if label and label != mid else mid,
                        created=created if isinstance(created, (int, float)) else None))
    out.sort(key=lambda m: (-(m["created"] or 0), m["id"]))
    return out
