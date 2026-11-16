"""Model API clients: OpenAI-compatible endpoints (Chat Completions, or the Responses API) and the native Anthropic API.

The frozen model is reached only through these clients.  Messages use one neutral format and are converted to each
provider's format just before sending:
  {"role": "system", "text": str}
  {"role": "user", "parts": [Part, ...]}
  {"role": "assistant", "text": str, "tool_calls": [ToolCall, ...], "provider_content": raw Anthropic content blocks (optional)}
  {"role": "tool", "tool_call_id": str, "name": str, "parts": [Part, ...]}
Part = {"type": "text", "text": str} | {"type": "image", "png": bytes, "label": str}

OpenAI tool messages cannot carry images: images are moved into a user message that immediately follows
("[images attached to the tool results above]").

Anthropic models with always-on thinking (Fable 5.x / Mythos 5.x / Opus 5.x / Sonnet 5.x / Haiku 5.x):
  - thinking cannot be switched off and temperature must not be sent; its depth is set with output_config.effort
    (from --reasoning-effort); thinking is displayed "summarized" and the summaries go into the transcript.
  - thinking blocks must be sent back unchanged and the history may only be appended to (the API rejects a request
    whose earlier turns were modified with HTTP 400): assistant messages keep provider_content (the raw content
    blocks), which is sent back verbatim; the episode loop only ever appends.
  - every request puts cache_control on the last block of the last message (moving that marker does not count as
    modifying the history); usage records cache reads and writes separately.
  - requests are streamed (max_tokens defaults to 64000, thinking included).  stop_reason == "refusal" ends the
    episode with an error, with no retry and no fallback model: a fallback would let a different model answer and
    mix models in the evaluation.

Credentials come from the arguments or the environment: FH_API_KEY / OPENAI_API_KEY and FH_API_BASE (OpenAI-compatible),
ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL (Anthropic).
"""
from __future__ import annotations

import base64
import copy
import dataclasses
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

_ANTHROPIC_THINKING = re.compile(r"^claude-(fable|mythos|opus-5|sonnet-5|haiku-5)")
_EFFORTS = ("low", "medium", "high", "xhigh", "max")


class LLMRefusal(RuntimeError):
    """The model refused (Anthropic stop_reason == "refusal"); never retried."""


def _anthropic_block(b) -> Dict[str, Any]:
    """Response content block -> request block that can be sent back as is.  The thinking / signature of thinking
    blocks are kept byte for byte."""
    t = b.type
    if t == "thinking":
        return {"type": "thinking", "thinking": b.thinking, "signature": b.signature}
    if t == "redacted_thinking":
        return {"type": "redacted_thinking", "data": b.data}
    if t == "text":
        return {"type": "text", "text": b.text}
    if t == "tool_use":
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
    return b.model_dump(exclude_none=True)


@dataclasses.dataclass
class ToolCall:
    id: str
    name: str
    args: Dict[str, Any]


@dataclasses.dataclass
class LLMResponse:
    text: str
    tool_calls: List[ToolCall]
    usage_in: int                      # all input tokens (cache reads and writes included)
    usage_out: int
    latency_s: float
    raw: Any = None
    cache_read: int = 0                # of which billed as cache reads
    cache_write: int = 0               # of which billed as cache writes
    provider_content: Optional[List[Dict[str, Any]]] = None   # Anthropic: raw content blocks (thinking included), sent back unchanged on the next turn
    stop_reason: str = ""


def _b64(png: bytes) -> str:
    return base64.b64encode(png).decode()


class LLMClient:
    """One model endpoint.  provider is "openai" (Chat Completions), "openai-responses" (Responses API) or "anthropic";
    when omitted it is "anthropic" for model names starting with "claude" and "openai" otherwise."""

    def __init__(self, model: str, provider: Optional[str] = None, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                 extra_body: Optional[Dict[str, Any]] = None, retries: int = 3, timeout: float = 600.0,
                 reasoning_effort: Optional[str] = None):
        self.model = model
        self.provider = provider or ("anthropic" if model.lower().startswith("claude") else "openai")
        self.anthropic_thinking = self.provider == "anthropic" and bool(_ANTHROPIC_THINKING.match(model.lower()))
        self.temperature = temperature
        self.max_tokens = max_tokens or (64000 if self.anthropic_thinking else 4096)
        self.extra_body = extra_body or {}
        self.reasoning_effort = reasoning_effort
        self.retries = retries
        self._prev_response_id: Optional[str] = None     # conversation chain of the Responses API
        self._n_sent = 0                                   # number of neutral messages already sent to the Responses API
        if self.provider in ("openai", "openai-responses"):
            from openai import OpenAI

            self.client = OpenAI(base_url=base_url or os.environ.get("FH_API_BASE") or None,
                                 api_key=api_key or os.environ.get("FH_API_KEY") or os.environ.get("OPENAI_API_KEY"),
                                 timeout=timeout)
        elif self.provider == "anthropic":
            import anthropic

            self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
                                              base_url=base_url or os.environ.get("ANTHROPIC_BASE_URL") or None, timeout=timeout)
        else:
            raise ValueError(self.provider)

    # ------------------------------------------------------------------ format conversion
    @staticmethod
    def _to_openai(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        pending_images: List[Dict[str, Any]] = []

        def flush():
            if pending_images:
                out.append({"role": "user", "content": [{"type": "text", "text": "[images attached to the tool results above]"}] + pending_images})
                pending_images.clear()

        for m in messages:
            r = m["role"]
            if r == "system":
                out.append({"role": "system", "content": m["text"]})
            elif r == "user":
                flush()
                content = []
                for p in m["parts"]:
                    if p["type"] == "text":
                        content.append({"type": "text", "text": p["text"]})
                    else:
                        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(p['png'])}", "detail": "auto"}})
                out.append({"role": "user", "content": content})
            elif r == "assistant":
                flush()
                # an empty reply (no text, no tool call): content must not be null, which is only allowed together with tool_calls
                # (otherwise the API returns 400)
                msg: Dict[str, Any] = {"role": "assistant", "content": m.get("text") or (None if m.get("tool_calls") else "(no content)")}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.args)}}
                                         for tc in m["tool_calls"]]
                out.append(msg)
            elif r == "tool":
                texts = [p["text"] for p in m["parts"] if p["type"] == "text"]
                for p in m["parts"]:
                    if p["type"] == "image":
                        pending_images.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(p['png'])}", "detail": "auto"}})
                        texts.append(f"[image '{p.get('label', 'image')}' attached below]")
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": "\n".join(texts) or "(empty)"})
        flush()
        return out

    @staticmethod
    def _to_anthropic(messages: List[Dict[str, Any]]):
        system = "\n".join(m["text"] for m in messages if m["role"] == "system")
        out: List[Dict[str, Any]] = []

        def blocks(parts):
            b = []
            for p in parts:
                if p["type"] == "text":
                    b.append({"type": "text", "text": p["text"]})
                else:
                    b.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(p["png"])}})
            return b

        for m in messages:
            r = m["role"]
            if r == "system":
                continue
            if r == "user":
                out.append({"role": "user", "content": blocks(m["parts"])})
            elif r == "assistant":
                if m.get("provider_content"):            # sent back verbatim (thinking blocks included); never split and rebuilt
                    out.append({"role": "assistant", "content": copy.deepcopy(m["provider_content"])})
                    continue
                content = []
                if m.get("text"):
                    content.append({"type": "text", "text": m["text"]})
                for tc in m.get("tool_calls") or []:
                    content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.args})
                out.append({"role": "assistant", "content": content or [{"type": "text", "text": "(no content)"}]})
            elif r == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": blocks(m["parts"]) or "(empty)"}
                if out and out[-1]["role"] == "user" and out[-1]["content"] and out[-1]["content"][0].get("type") == "tool_result":
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return system, out

    # ------------------------------------------------------------------ calls
    def chat(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> LLMResponse:
        last_err = None
        for attempt in range(self.retries):
            try:
                t0 = time.time()
                if self.provider == "openai":
                    resp = self._chat_openai(messages, tools)
                elif self.provider == "openai-responses":
                    resp = self._chat_responses(messages, tools)
                else:
                    resp = self._chat_anthropic(messages, tools)
                resp.latency_s = time.time() - t0
                return resp
            except LLMRefusal:
                raise
            except Exception as e:  # network / rate limit: exponential backoff (an Anthropic 400 will not succeed when resent, so raise it at once)
                if self.provider == "anthropic" and type(e).__name__ == "BadRequestError":
                    raise
                last_err = e
                time.sleep(2.0 * (2 ** attempt))
        raise RuntimeError(f"LLM call failed after {self.retries} attempts: {last_err}")

    def _chat_openai(self, messages, tools) -> LLMResponse:
        base = str(getattr(self.client, "base_url", "") or "")
        tok_key = "max_completion_tokens" if "api.openai.com" in base else "max_tokens"   # Gemini's OpenAI-compatible layer only accepts max_tokens
        kwargs: Dict[str, Any] = dict(model=self.model, messages=self._to_openai(messages))
        kwargs[tok_key] = self.max_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if tools:
            kwargs["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                                "parameters": t["parameters"]}} for t in tools]
            kwargs["tool_choice"] = "auto"
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        r = self.client.chat.completions.create(**kwargs)
        msg = r.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append(ToolCall(tc.id, tc.function.name, args))
        u = r.usage
        return LLMResponse(text=msg.content or "", tool_calls=calls,
                           usage_in=int(getattr(u, "prompt_tokens", 0) or 0), usage_out=int(getattr(u, "completion_tokens", 0) or 0),
                           latency_s=0.0, raw=r.model_dump() if hasattr(r, "model_dump") else None)

    # ------------------------------------------------------------------ OpenAI Responses API (reasoning models + function tools)
    @staticmethod
    def _parts_to_input_content(parts):
        out = []
        for p in parts:
            if p["type"] == "text":
                out.append({"type": "input_text", "text": p["text"]})
            else:
                out.append({"type": "input_image", "image_url": f"data:image/png;base64,{_b64(p['png'])}", "detail": "auto"})
        return out

    def _chat_responses(self, messages, tools) -> LLMResponse:
        """Chain the turns with previous_response_id: each request sends only the messages added since the previous one
        (tool results / user messages); the system prompt is passed as instructions every time."""
        system = "\n".join(m["text"] for m in messages if m["role"] == "system")
        new_msgs = messages[self._n_sent:] if self._prev_response_id else messages
        items = []
        for m in new_msgs:
            r = m["role"]
            if r == "system":
                continue
            if r == "user":
                items.append({"role": "user", "content": self._parts_to_input_content(m["parts"])})
            elif r == "tool":
                texts = [p["text"] for p in m["parts"] if p["type"] == "text"]
                imgs = [p for p in m["parts"] if p["type"] == "image"]
                if imgs:
                    texts.append("[" + ", ".join(f"image '{p.get('label', 'image')}' attached in the next user message" for p in imgs) + "]")
                items.append({"type": "function_call_output", "call_id": m["tool_call_id"], "output": "\n".join(texts) or "(empty)"})
                if imgs:
                    items.append({"role": "user", "content": [{"type": "input_text", "text": "[images attached to the tool result above]"}] + self._parts_to_input_content(imgs)})
            elif r == "assistant" and not self._prev_response_id:
                # an assistant message already in the history on the first send (should not happen): send its text only
                if m.get("text"):
                    items.append({"role": "assistant", "content": [{"type": "output_text", "text": m["text"]}]})
        kwargs: Dict[str, Any] = dict(model=self.model, input=items, instructions=system or None, max_output_tokens=self.max_tokens, store=True)
        if self._prev_response_id:
            kwargs["previous_response_id"] = self._prev_response_id
        if tools:
            kwargs["tools"] = [{"type": "function", "name": t["name"], "description": t["description"], "parameters": t["parameters"]} for t in tools]
            kwargs["tool_choice"] = "auto"
        if self.reasoning_effort and self.reasoning_effort != "none":
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        r = self.client.responses.create(**kwargs)
        self._prev_response_id = r.id
        self._n_sent = len(messages) + 1          # the loop appends this turn's assistant message next; the next request starts after it
        calls, texts = [], []
        for item in r.output:
            t = getattr(item, "type", "")
            if t == "function_call":
                try:
                    args = json.loads(item.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": item.arguments}
                calls.append(ToolCall(item.call_id, item.name, args))
            elif t == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", "") == "output_text":
                        texts.append(c.text)
        u = r.usage
        return LLMResponse(text="\n".join(texts), tool_calls=calls, usage_in=int(getattr(u, "input_tokens", 0) or 0),
                           usage_out=int(getattr(u, "output_tokens", 0) or 0), latency_s=0.0,
                           raw=r.model_dump() if hasattr(r, "model_dump") else None)

    def _chat_anthropic(self, messages, tools) -> LLMResponse:
        system, msgs = self._to_anthropic(messages)
        kwargs: Dict[str, Any] = dict(model=self.model, max_tokens=self.max_tokens, messages=msgs)
        if self.temperature is not None and not self.anthropic_thinking:   # thinking models reject temperature (400)
            kwargs["temperature"] = self.temperature
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]
        extra = dict(self.extra_body)
        if self.anthropic_thinking:
            extra.setdefault("thinking", {"type": "adaptive", "display": "summarized"})
            if self.reasoning_effort in _EFFORTS:
                extra.setdefault("output_config", {"effort": self.reasoning_effort})
            # prompt caching: mark the last block only in the copy being sent; the stored history is left untouched
            last = msgs[-1]
            if isinstance(last.get("content"), list) and last["content"]:
                last = msgs[-1] = dict(last, content=list(last["content"]))
                last["content"][-1] = dict(last["content"][-1], cache_control={"type": "ephemeral"})
        if extra:
            kwargs["extra_body"] = extra                 # via extra_body: an older SDK may not have typed parameters for these yet
        if self.anthropic_thinking:
            with self.client.messages.stream(**kwargs) as s:
                r = s.get_final_message()
        else:
            r = self.client.messages.create(**kwargs)
        if r.stop_reason == "refusal":
            det = getattr(r, "stop_details", None)
            raise LLMRefusal(f"refusal ({getattr(det, 'category', None)}): {getattr(det, 'explanation', '')}")
        text, calls = [], []
        for b in r.content:
            if b.type == "text":
                text.append(b.text)
            elif b.type == "tool_use":
                calls.append(ToolCall(b.id, b.name, dict(b.input)))
        u = r.usage
        c_read = int(getattr(u, "cache_read_input_tokens", 0) or 0)
        c_write = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        return LLMResponse(text="\n".join(text), tool_calls=calls, usage_in=int(u.input_tokens) + c_read + c_write,
                           usage_out=int(u.output_tokens), latency_s=0.0, raw=r.model_dump() if hasattr(r, "model_dump") else None,
                           cache_read=c_read, cache_write=c_write,
                           provider_content=[_anthropic_block(b) for b in r.content] if self.anthropic_thinking else None,
                           stop_reason=str(r.stop_reason or ""))
