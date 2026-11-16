"""One ForceHarness episode on ForceBench (the agent loop on top of the ForceBench backend).

It is the Episode of agent/loop.py with: the ForceBench system prompt (15 Hz, ForceBench's frames and table height); the
ForceBench-only tools (locate, locate_object, press_turn, move_turn); gridded camera images for RGB-only perception; a guard
against blank renders; and saving of transcript / reports / record also when the evaluator ends the episode early (success
criterion met, step limit, connection closed)."""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from forceharness.agent.loop import PROMPT_DIR, Episode
from forceharness.backends.forcebench.session import EvaluatorClosed


class FBEpisode(Episode):
    system_template = "forcebench/system_privileged.md"
    # Backend constant for the ForceBench arm (the primitive code is unchanged; only an instance attribute is overridden):
    # at full reach the arm's z stiffness is about 0.5 N/mm (1.2 N/mm close in), so the 6 mm contact-seeking lead used on
    # MuJoCo cannot build the 3 N contact threshold and the contact search waits until it times out; 15 mm can.
    prim_overrides = dict(max_lead_contact=0.015)

    def __init__(self, sim, agent, cfg, injector=None, prims=None, prim_overrides=None):
        from forceharness.backends.forcebench.prims import LOCATE_OBJECT_SCHEMA, LOCATE_SCHEMA, MOVE_TURN_SCHEMA, FBPrimitives, PRESS_TURN_SCHEMA

        task = sim.task
        # ForceBench-only tools are listed in task.extra["fb_extra_tools"]; the base loop only knows the names in agent/tools.py,
        # so they are attached here. locate / locate_object (pixel -> world position) and move_turn do not read force and are
        # given to every interface; press_turn needs the force interface.
        listed = tuple(task.extra.get("fb_extra_tools", ()))
        fb_extra = tuple(t for t in listed if t in ("locate", "locate_object", "move_turn") or cfg.interface == "force")
        if cfg.interface != "force" and "press_turn" in listed and "move_turn" not in fb_extra:
            fb_extra += ("move_turn",)            # press_turn closes a loop on force; without force it becomes the position-only move_turn
        self.perception = str(task.extra.get("perception", "privileged"))
        self.loc_log = []
        if prims is None:
            prims = FBPrimitives(sim, interface=cfg.interface, f_contact=cfg.f_contact, f_cap=cfg.f_cap,
                                 extra_tools=tuple(task.extra.get("extra_tools", ())) + tuple(t for t in fb_extra if t not in ("locate", "locate_object")),
                                 pos_tol=float(task.extra.get("pos_tol", 0.003)),
                                 slow_in_contact=bool(task.extra.get("slow_in_contact", False)))
        super().__init__(sim, agent, cfg, injector=injector, prims=prims)
        self.prims.turn_sides = bool(task.extra.get("turn_sides", False))
        objects = sorted(task.extra.get("object_models") or {})
        locate_object = dict(LOCATE_OBJECT_SCHEMA, parameters={**LOCATE_OBJECT_SCHEMA["parameters"], "properties": {
            **LOCATE_OBJECT_SCHEMA["parameters"]["properties"], "object": {"type": "string", "enum": objects or [""]}}})
        schemas = {"press_turn": PRESS_TURN_SCHEMA, "move_turn": MOVE_TURN_SCHEMA, "locate": LOCATE_SCHEMA, "locate_object": locate_object}
        at = next((i for i, t in enumerate(self.tools) if t["name"] == "done"), len(self.tools))
        for name in fb_extra:
            self.tools.insert(at, dict(schemas[name]))
            at += 1
        for k, v in dict(self.prim_overrides if prim_overrides is None else prim_overrides).items():
            if not hasattr(self.prims, k):
                raise AttributeError(f"Primitives has no attribute {k!r}")
            setattr(self.prims, k, v)

    def system_prompt(self) -> str:
        rgb = getattr(self, "perception", "privileged") == "rgb"
        tpl = (PROMPT_DIR / ("forcebench/system_rgb.md" if rgb else self.system_template)).read_text(encoding="utf-8")
        tool_names = ", ".join(t["name"] for t in self.tools)
        if rgb:
            vision_note = ("You receive two camera images at the start and after every turn: 'main' and 'wrist' (224 x 224 pixels, shown "
                           "enlarged with a yellow pixel grid: faint lines every 16 px, labelled every 32 px, in original pixel units).")
        else:
            vision_note = ("You also receive camera images (external 'agentview' and wrist camera) once per turn."
                           if self.cfg.vision else "You receive NO camera images; rely on the scene description and the signals you are given.")
        txt = tpl.format(interface=self.cfg.interface, tool_names=tool_names,
                         force_note=self.injector.describe() or "You receive no force information.", vision_note=vision_note,
                         f_cap=self.cfg.f_cap, table_z=float(self.task.extra.get("table_top_z", 0.0)),
                         budget_s=float(self.sim.max_steps) / 15.0)
        if self.cfg.playbook:
            txt += "\n\n" + (PROMPT_DIR / self.cfg.playbook).read_text(encoding="utf-8").strip()
        return txt

    # ---------------------------------------------------------------- RGB only: gridded images, locate, locate_object
    def image_parts(self):
        if getattr(self, "perception", "privileged") != "rgb":
            return super().image_parts()
        from forceharness.backends.forcebench.vision import grid_png

        parts = []
        for cam in ("main", "wrist"):
            img = self.sim.images_raw.get(cam)
            if img is not None:
                parts.append({"type": "image", "png": grid_png(img), "label": cam})
        return parts

    def _locate_object(self, a):
        from forceharness.backends.forcebench.vision import locate_known_object

        name = str(a.get("object", ""))
        models = self.task.extra.get("object_models") or {}
        if name not in models:
            return [{"type": "text", "text": f"tool error: locate_object: object must be one of {sorted(models)}"}]
        try:
            cam, u, v = str(a.get("camera", "main")), float(a["u"]), float(a["v"])
            res = locate_known_object(self.sim, cam, u, v, models[name])
        except Exception as e:
            return [{"type": "text", "text": f"tool error: locate_object: {e}"}]
        if res is None:
            return [{"type": "text", "text": f"locate_object {name}: nothing found around {cam} ({u:.0f}, {v:.0f}); click on the object"}]
        x, y = float(res["xy"][0]), float(res["xy"][1])
        self.loc_log.append(dict(step=self.sim.step_index, tool="locate_object", object=name, camera=cam, u=u, v=v, score=round(res["score"], 3),
                                 p=[round(x, 4), round(y, 4), 0.0], rough=np.round(res["rough"], 4).tolist(),
                                 truth=getattr(self.sim, "diag", {}).get("entities")))
        return [{"type": "text", "text": (f"locate_object {name} from {cam} ({u:.0f}, {v:.0f}) -> axis at world x = {x:+.4f}, y = {y:+.4f} m; "
                                          f"fit score {res['score']:.2f}")},
                {"type": "text", "text": self.state_text()}]

    def run_tool(self, tc):
        if tc.name == "locate_object":
            self.n_tool_calls += 1
            return self._locate_object(dict(tc.args))
        if tc.name != "locate":
            return super().run_tool(tc)
        self.n_tool_calls += 1
        a = dict(tc.args)
        try:
            cam = str(a.get("camera", "main"))
            u, v = float(a["u"]), float(a["v"])
            # Function tools of the OpenAI Responses API are strict by default, so the model fills in every field (unused
            # ones with 0). The plane is therefore named by `plane`, not inferred from which coordinate was given: a version
            # that inferred it rejected every locate call the model made.
            key = str(a.get("plane") or "z").strip().lower()
            if key not in ("x", "y", "z"):
                return [{"type": "text", "text": f"tool error: locate: plane must be 'z', 'x' or 'y' (got {key!r})"}]
            if key == "z":
                val = float(a["z"]) if a.get("z") is not None else float(self.task.extra.get("table_top_z", 0.0) or 0.0)
            else:
                raw = a.get("value") if a.get("value") is not None else a.get(key)
                if raw is None:
                    return [{"type": "text", "text": f"tool error: locate: plane '{key}' needs value = the {key} coordinate of the plane"}]
                val = float(raw)
            p = self.sim.locate_on_plane(cam, u, v, val, axis="xyz".index(key))
        except Exception as e:
            return [{"type": "text", "text": f"tool error: locate: {e}"}]
        if p is None:
            return [{"type": "text", "text": f"locate: that pixel's line of sight does not reach the plane {key} = {val:.3f} m"}]
        mpp = self.sim.metres_per_pixel(cam, p)
        self.loc_log.append(dict(step=self.sim.step_index, camera=cam, u=u, v=v, plane=key, value=val, p=np.round(p, 4).tolist(),
                                 truth=getattr(self.sim, "diag", {}).get("entities")))
        return [{"type": "text", "text": (f"locate {cam} ({u:.0f}, {v:.0f}) on {key} = {val:.3f} m -> world [{p[0]:+.3f}, {p[1]:+.3f}, "
                                          f"{p[2]:+.3f}] m (one pixel there is about {mpp * 1000:.1f} mm)")},
                {"type": "text", "text": self.state_text()}]

    def blank_images(self) -> list:
        """Describe each start image that is a uniform colour (empty list = images are fine).

        When the GPU runs short of memory, Isaac Sim can render all-white images; a model then only reports that it sees
        nothing and gives up, so such an episode is recorded as a rendering failure instead (see run_safe)."""
        if getattr(self, "perception", "privileged") != "rgb" and not self.cfg.vision:
            return []
        bad = []
        for cam, img in (self.sim.images_raw or {}).items():
            a = np.asarray(img, dtype=float)
            if a.size and float(a.std()) < 3.0:
                bad.append(f"{cam} (mean {a.mean():.0f}, std {a.std():.1f})")
        return bad

    def run_safe(self) -> Dict[str, Any]:
        t0 = time.time()
        bad = self.blank_images()
        if bad:                                   # broken render: record the episode as blank_images without calling (or paying for) the model
            rec = self._partial(t0, "harness")
            rec.update(stop_reason="blank_images", error="camera images are blank: " + ", ".join(bad))
            self._save(rec)
            self._save_trace()
            return rec
        try:
            rec = self.run()
            rec["ended_by"] = "harness"
        except EvaluatorClosed:
            rec = self._partial(t0, "evaluator")
            self._save(rec)
        self._save_trace()
        return rec

    def _partial(self, t0: float, ended_by: str) -> Dict[str, Any]:
        cfg = self.cfg
        cost = self.cost_usd()
        return dict(
            episode_id=cfg.episode_id, condition=cfg.condition, task=cfg.task, model=cfg.model, fmt=cfg.fmt, vision=cfg.vision,
            interface=cfg.interface, seed=cfg.seed, agent=getattr(self.agent, "kind", "llm"),
            success=None, claimed_success=None, verdict="", verdict_correct=None, truth={},
            n_model_calls=self.n_calls, n_tool_calls=self.n_tool_calls, tokens_in=self.usage_in, tokens_out=self.usage_out,
            tokens_cache_read=self.cache_read, tokens_cache_write=self.cache_write,
            cost_usd=round(cost, 4), sim_time_s=round(self.sim.sim_time, 2), wall_time_s=round(time.time() - t0, 1),
            peak_wrench_N=round(float(self.sim.peak_wrench), 2), peak_object_contact_force_N=0.0,
            n_events=sum(len(r.events) for r in self.reports),
            n_force_cap=sum(r.termination in ("force_cap", "torque_cap") for r in self.reports),
            qa_asked=0, qa_correct=0, stop_reason=f"ended_by_{ended_by}", error=None, note="", pregrasp=getattr(self, "pregrasp", None),
            ended_by=ended_by, cfg=dataclasses.asdict(cfg),
        )

    def _save_trace(self) -> None:
        tr = getattr(self.sim, "trace", None)
        if tr:
            np.save(self.out / "fb_trace.npy", np.asarray(tr, dtype=np.float32))   # step, action (7), state before the action (7), raw wrench (6), sag compensation
        pt = getattr(self.sim, "priv_trace", None)
        if pt:
            np.save(self.out / "fb_priv_trace.npy", np.asarray(pt, dtype=np.float32))   # diagnostics: step, true position (3) + rpy (3) of the first scene entity
        (self.out / "fb_privileged_last.json").write_text(json.dumps(getattr(self.sim, "privileged", {}) or getattr(self.sim, "diag", {}),
                                                                    default=str), encoding="utf-8")
        logs = list(getattr(self, "loc_log", []) or []) + list(getattr(self.agent, "loc_log", []) or [])
        if logs:
            (self.out / "fb_locate_log.json").write_text(json.dumps(logs, default=str), encoding="utf-8")
        dbg = list(getattr(self.agent, "dbg_imgs", []) or [])
        if dbg:                                   # images the scripted agent saw at each perception step (RGB only), for checking its detections offline
            arrs = {f"k{d['k']}_{c}": im for d in dbg for c, im in d["images"].items()}
            np.savez_compressed(self.out / "fb_perception_imgs.npz", **arrs)
            (self.out / "fb_perception_meta.json").write_text(json.dumps(
                [{k: v for k, v in d.items() if k != "images"} for d in dbg], default=lambda o: np.asarray(o).tolist()), encoding="utf-8")
