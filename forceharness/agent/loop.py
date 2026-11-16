"""The episode loop: system prompt + task + scene -> the model returns tool calls -> primitives run -> the injector
renders each ContactReport as the tool result -> back to the model.

The model decides only between primitives.  Each episode records success, claimed success, model and tool calls,
tokens, cost, simulated time, wall time, peak force and the number of events, and optionally the accuracy of a contact
question (with qa_every > 0 the model is asked every N turns whether the tool is touching anything, and the answer is
compared with ground truth).
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from forceharness.primitives.library import Primitives
from forceharness.primitives.report import ContactReport
from forceharness.translation import make_injector
from forceharness.agent.llm import LLMClient, LLMResponse, ToolCall
from forceharness.agent.tools import tool_schemas

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


@dataclasses.dataclass
class EpisodeConfig:
    """Settings of one episode: the condition (task, model, force form, vision, interface), the seed, the budgets, the
    simulated force noise and the output location."""
    task: str
    model: str
    fmt: str = "events"
    vision: bool = True
    interface: str = "force"
    seed: int = 0
    max_turns: int = 25
    max_tool_calls: int = 60
    max_sim_time: float = 150.0
    max_wall_time: float = 1200.0
    images_per_turn: bool = True
    image_size: int = 256
    kp: float = 150.0
    scene_noise_pos: float = 0.003
    scene_noise_yaw_deg: float = 2.0
    f_contact: float = 2.0
    f_cap: float = 25.0
    force_sign: float = -1.0
    force_noise_std: float = 0.0     # simulated force noise of a Franka-like arm (N per tick); the HiddenPhys configs use 0.5
    force_bias_std: float = 0.0      # residual bias after calibration (N, constant per episode); the HiddenPhys configs use 0.3
    qa_every: int = 0
    proprio: str = "minimal"        # minimal: no duration / opening while carrying / sim time (closes a proprioceptive side channel); full: report them
    playbook: str = ""              # a playbook file under prompts/; when set it is appended to the system prompt (worked-example ablation)
    run_name: str = "dev"
    out_dir: str = "runs"
    price_in_per_m: float = 0.0
    price_out_per_m: float = 0.0
    note: str = ""

    @property
    def condition(self) -> str:
        return f"{self.task}|{self.model}|{self.fmt}|{'vision' if self.vision else 'novision'}|{self.interface}"

    @property
    def episode_id(self) -> str:
        return f"{self.task}__{self.model.replace('/', '_')}__{self.fmt}__{'v' if self.vision else 'nv'}__{self.interface}__s{self.seed}"


class LLMAgent:
    """Agent backed by an API model: one chat call per decision."""
    kind = "llm"

    def __init__(self, client: LLMClient):
        self.client = client

    def decide(self, ctx: Dict[str, Any]) -> LLMResponse:
        return self.client.chat(ctx["messages"], ctx["tools"])

    def ask(self, messages: List[Dict[str, Any]], question: str) -> str:
        msgs = messages + [{"role": "user", "parts": [{"type": "text", "text": question}]}]
        return self.client.chat(msgs, []).text


def _json_safe(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, bytes):
        return f"<{len(o)} bytes>"
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    return str(o)


class Episode:
    """One episode of a model (or a scripted agent) driving the primitives of one backend; ``run`` returns the record."""

    def __init__(self, sim, agent, cfg: EpisodeConfig, injector=None, prims: Optional[Primitives] = None):
        self.sim = sim
        self.task = sim.task
        self.agent = agent
        self.cfg = cfg
        extra_tools = tuple(self.task.extra.get("extra_tools", ()))
        # pos_tol: the "arrived" tolerance of move_to (default 3 mm); a move_by shorter than it counts as arrived without
        # moving, so a task that needs finer moves sets a smaller pos_tol in task.extra
        self.prims = prims or Primitives(sim, interface=cfg.interface, f_contact=cfg.f_contact, f_cap=cfg.f_cap, extra_tools=extra_tools,
                                         pos_tol=float(self.task.extra.get("pos_tol", 0.003)),
                                         slow_in_contact=bool(self.task.extra.get("slow_in_contact", False)))
        self.injector = injector or make_injector(cfg.fmt)
        self.tools = tool_schemas(cfg.interface, self.task.has_gripper, extra_tools)
        self.messages: List[Dict[str, Any]] = []
        self.reports: List[ContactReport] = []
        self.last_scene: List[Dict[str, Any]] = []
        self.usage_in = self.usage_out = 0
        self.cache_read = self.cache_write = 0     # input tokens billed as cache reads / writes (Anthropic)
        self.n_calls = 0
        self.n_tool_calls = 0
        self.qa = {"asked": 0, "correct": 0}
        self.mlog: List[Dict[str, Any]] = []     # measurements of this episode (references for the L3 rendering)
        self.images_saved = 0
        self.out = Path(cfg.out_dir) / cfg.run_name / "transcripts" / cfg.episode_id
        self.out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ prompt
    def system_prompt(self) -> str:
        tpl = (PROMPT_DIR / "hiddenphys" / "system.md").read_text(encoding="utf-8")
        tool_names = ", ".join(t["name"] for t in self.tools)
        vision_note = ("You also receive camera images (external 'agentview' and wrist camera) once per turn."
                       if self.cfg.vision else "You receive NO camera images; rely on the scene description and the signals you are given.")
        txt = tpl.format(interface=self.cfg.interface, tool_names=tool_names, force_note=self.injector.describe() or
                         "You receive no force information.", vision_note=vision_note, f_cap=self.cfg.f_cap)
        if self.cfg.playbook:
            txt += "\n\n" + (PROMPT_DIR / self.cfg.playbook).read_text(encoding="utf-8").strip()
        return txt

    def state_text(self) -> str:
        p = self.sim.eef_pos
        g = self.sim.gripper_opening
        s = f"robot state: tool at [{p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}] m"
        if g is not None:
            if self.cfg.proprio != "full" and self.prims.grip_closed:
                s += ", gripper closed"
            else:
                s += f", gripper opening {g*1000:.1f} mm ({'closed' if self.prims.grip_closed else 'open command'})"
        if self.cfg.proprio == "full":
            s += f", sim time {self.sim.sim_time:.1f} s"
        return s

    def scene_text(self) -> str:
        self.last_scene = self.sim.scene()
        return "scene:\n" + json.dumps(self.last_scene, default=_json_safe) + "\n" + self.state_text()

    def image_parts(self) -> List[Dict[str, Any]]:
        parts = []
        if not self.cfg.vision:
            return parts
        from PIL import Image
        import io

        for cam, img in self.sim.images().items():
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="PNG")
            png = buf.getvalue()
            parts.append({"type": "image", "png": png, "label": cam})
        return parts

    # ------------------------------------------------------------------ execution
    def run_tool(self, tc: ToolCall) -> List[Dict[str, Any]]:
        name, args = tc.name, dict(tc.args)
        self.n_tool_calls += 1
        if name == "scene":
            return [{"type": "text", "text": self.scene_text()}]
        try:
            rep = self.prims.call(name, **args)
        except TypeError as e:
            return [{"type": "text", "text": f"tool error: bad arguments for {name}: {e}"}]
        except ValueError as e:
            return [{"type": "text", "text": f"tool error: {e}"}]
        except Exception as e:  # a simulator exception must not crash the episode
            return [{"type": "text", "text": f"tool error: {type(e).__name__}: {e}"}]
        self.reports.append(rep)
        parts = self.injector.render(rep, {"state": self.state_text(), "proprio": self.cfg.proprio, "log": self.mlog})
        for q in rep.quantities:
            self.mlog.append(dict(turn=len(self.messages), primitive=rep.primitive, **{k: q.get(k) for k in ("name", "value", "unit", "sigma", "flags", "extra")}))
        parts.append({"type": "text", "text": self.state_text()})
        return parts

    def run(self) -> Dict[str, Any]:
        cfg = self.cfg
        t_wall0 = time.time()
        self.messages = [{"role": "system", "text": self.system_prompt()}]
        self.pregrasp = None
        if getattr(self.task, "pregrasp_hook", None) is not None:     # setup before the episode (e.g. grasping the handle): not counted as a call, result not shown to the model
            try:
                self.pregrasp = self.task.pregrasp_hook(self.sim, self.prims)
            except Exception as e:
                self.pregrasp = dict(error=f"{type(e).__name__}: {e}")
        intro = (f"TASK: {self.task.instruction}\n\n{self.scene_text()}\n\nSafety: the controller aborts any motion whose wrist force exceeds "
                 f"{cfg.f_cap:g} N. Begin.")
        self.messages.append({"role": "user", "parts": [{"type": "text", "text": intro}] + self.image_parts()})
        finished, claimed, verdict, note, stop_reason, error = False, None, "", "", "budget", None
        for turn in range(cfg.max_turns):
            ctx = dict(messages=self.messages, tools=self.tools, reports=self.reports, last_scene=self.last_scene,
                       interface=cfg.interface, task=self.task, turn=turn, sim_state=self.state_text())
            try:
                resp = self.agent.decide(ctx)
            except Exception as e:
                error = f"agent error: {e}"
                stop_reason = "agent_error"
                break
            self.n_calls += 1
            self.usage_in += resp.usage_in
            self.usage_out += resp.usage_out
            self.cache_read += int(getattr(resp, "cache_read", 0) or 0)
            self.cache_write += int(getattr(resp, "cache_write", 0) or 0)
            amsg = {"role": "assistant", "text": resp.text, "tool_calls": resp.tool_calls}
            if getattr(resp, "provider_content", None):
                amsg["provider_content"] = resp.provider_content      # Anthropic thinking blocks must be sent back unchanged
            self.messages.append(amsg)
            if not resp.tool_calls:
                self.messages.append({"role": "user", "parts": [{"type": "text", "text": "Please act by calling a tool (or call done)."}]})
                continue
            for tc in resp.tool_calls:
                if tc.name == "done":
                    claimed = bool(tc.args.get("success", False))
                    verdict = str(tc.args.get("verdict", "") or "")
                    note = str(tc.args.get("note", "") or "")
                    finished = True
                    self.messages.append({"role": "tool", "tool_call_id": tc.id, "name": "done", "parts": [{"type": "text", "text": "episode ended"}]})
                    break
                parts = self.run_tool(tc)
                self.messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "parts": parts})
                if self.n_tool_calls >= cfg.max_tool_calls:
                    break
            if finished:
                stop_reason = "done"
                break
            if cfg.vision and cfg.images_per_turn:
                self.messages.append({"role": "user", "parts": [{"type": "text", "text": "Current camera views:"}] + self.image_parts()})
            if cfg.qa_every and hasattr(self.agent, "ask") and (turn + 1) % cfg.qa_every == 0:
                self._contact_qa()
            if self.n_tool_calls >= cfg.max_tool_calls:
                stop_reason = "max_tool_calls"; break
            if self.sim.sim_time > cfg.max_sim_time:
                stop_reason = "max_sim_time"; break
            if time.time() - t_wall0 > cfg.max_wall_time:
                stop_reason = "max_wall_time"; break
        success = False
        try:
            success = bool(self.sim.success())
        except Exception as e:
            error = (error or "") + f" success_check_error: {e}"
        truth = dict(self.sim.task_state)
        verdict_correct = None
        if "label" in truth and verdict:
            verdict_correct = verdict.strip().lower().startswith(truth["label"])
        cost = self.cost_usd()
        rec = dict(
            episode_id=cfg.episode_id, condition=cfg.condition, task=cfg.task, model=cfg.model, fmt=cfg.fmt, vision=cfg.vision,
            interface=cfg.interface, seed=cfg.seed, agent=getattr(self.agent, "kind", "llm"),
            success=success, claimed_success=claimed, verdict=verdict, verdict_correct=verdict_correct, truth=truth,
            n_model_calls=self.n_calls, n_tool_calls=self.n_tool_calls, tokens_in=self.usage_in, tokens_out=self.usage_out,
            tokens_cache_read=self.cache_read, tokens_cache_write=self.cache_write,
            cost_usd=round(cost, 4), sim_time_s=round(self.sim.sim_time, 2), wall_time_s=round(time.time() - t_wall0, 1),
            peak_wrench_N=round(float(self.sim.peak_wrench), 2), peak_object_contact_force_N=round(float(self.sim.peak_contact_force), 2),
            n_events=sum(len(r.events) for r in self.reports), n_force_cap=sum(r.termination in ("force_cap", "torque_cap") for r in self.reports),
            qa_asked=self.qa["asked"], qa_correct=self.qa["correct"], stop_reason=stop_reason, error=error, note=note,
            pregrasp=self.pregrasp,
            cfg=dataclasses.asdict(cfg),
        )
        self._save(rec)
        return rec

    def cost_usd(self) -> float:
        """Cost in USD.  Input tokens are billed at the list price; cache reads / writes (Anthropic) are billed at the
        cache prices when the price table has them.  OpenAI cache usage is not recorded, so OpenAI input is billed at the
        list price."""
        from forceharness.eval.pricing import cache_prices_for    # imported here because the eval package imports this module

        cfg = self.cfg
        cr, cw = cache_prices_for(cfg.model)
        cr = cfg.price_in_per_m if cr is None else cr
        cw = cfg.price_in_per_m if cw is None else cw
        plain = self.usage_in - self.cache_read - self.cache_write
        return (plain * cfg.price_in_per_m + self.cache_read * cr + self.cache_write * cw + self.usage_out * cfg.price_out_per_m) / 1e6

    def _contact_qa(self) -> None:
        truth = bool(self.sim.gripper_contact_state()["in_contact"])
        q = "Question (answer with a single word, yes or no): is the tool currently in physical contact with anything?"
        try:
            ans = self.agent.ask(self.messages, q).strip().lower()
        except Exception:
            return
        self.qa["asked"] += 1
        pred = ans.startswith("yes")
        self.qa["correct"] += int(pred == truth)
        self.n_calls += 1

    def _save(self, rec: Dict[str, Any]) -> None:
        msgs = []
        for m in self.messages:
            mm = dict(m)
            if "parts" in mm:
                parts = []
                for p in mm["parts"]:
                    if p["type"] == "image":
                        fn = self.out / f"img_{self.images_saved:03d}_{p.get('label', 'img')}.png"
                        fn.write_bytes(p["png"])
                        self.images_saved += 1
                        parts.append({"type": "image", "path": fn.name, "label": p.get("label")})
                    else:
                        parts.append(p)
                mm["parts"] = parts
            if "tool_calls" in mm:
                mm["tool_calls"] = [dataclasses.asdict(tc) for tc in mm["tool_calls"]]
            msgs.append(mm)
        (self.out / "transcript.json").write_text(json.dumps(msgs, indent=1, default=_json_safe), encoding="utf-8")
        (self.out / "reports.json").write_text(json.dumps([r.to_dict() for r in self.reports], default=_json_safe), encoding="utf-8")
        (self.out / "record.json").write_text(json.dumps(rec, indent=1, default=_json_safe), encoding="utf-8")
