# Architecture

ForceHarness sits between a frozen model and a robot. The model is reached only through an API, so its weights never change
and each decision takes seconds; contact forces change in milliseconds. The harness therefore keeps force in two places:
**inside** each tool, where a primitive closes the loop on force at the controller's rate, and **in the tool result**, where
fixed estimators turn the force trace into physical quantities that inform the next decision.

## Three layers

| Layer | What it does | Code |
|---|---|---|
| 1. Force sensing and calibration | Turns the robot's raw signals into the force that the environment applies to the hand. A wrist F/T sensor gives it directly; arms without one estimate the flange wrench from external joint torques, `w = (Jᵀ)⁺ τ_ext`, `F = −(w_F − b)`. Three run-time rules handle slow bias: **tare at rest** (re-zero at rest before each empty-hand primitive), **settle, then read** (relax the controller onto the measured pose, then average a still window), **report changes** (relative to the empty hand or the moment of the grasp). | `backends/mujoco/env.py` (sensor + noise model), `backends/forcebench/fbsim.py`, `robot_side/franka_polymetis/wrench.py` + `forceharness/real/backend.py` (real FR3) |
| 2. Self-terminating force primitives | Each tool is a contact skill with a few parameters. All primitives share one loop: at every tick a small control law reads the tared force and pose and returns an increment of the target pose; the loop clamps the lead of the target over the measured pose, `F ≈ K (p* − p)` with `‖p* − p‖ ≤ L`, sends the target, detects contact events, and decides whether to stop. Force regulation is admittance: `v = clip(k_f (F_d − F_u), ±v_max)`. A primitive ends on an event (contact, stall, a drop in resistance, its own completion condition), with force caps and a time limit as fallbacks, then unloads the spring and holds still while the model thinks. | `primitives/library.py` (`Primitives`), `primitives/events.py`, `primitives/report.py` (`ContactReport`); real-robot subclass in `forceharness/real/primitives.py` |
| 3. Physics translation | After each primitive, fixed estimators (means, least-squares fits, error propagation) turn the trace into quantities with uncertainty: payload mass, centre-of-mass offset, friction, contact height corrected for detector lag, stiffness, insertion advance with an outcome flag. No task thresholds: the layer says how heavy something is, never whether that is enough. Rendering levels: L1 = name, value, unit; **L2 (default, `phys2`)** = ± σ, the window and status flags. | `translation/estimators.py`, `translation/quantities.py` |

The **guard** sits under all of it: limits on force, speed, lead and workspace that the model cannot change
(simulation: `f_cap` in the primitives; real robot: `robot_side/franka_polymetis/guard.py` plus the runner's validation).

## One turn

```
observation (instruction, tool menu, text history, latest images)  ──►  model  ──►  one tool call
                                                                                          │
     validation (tool on the menu, parameters in range, path inside the workspace) ◄─────┘
                         │ refused: the robot does not move, the tool result says why
                         ▼
     primitive loop at ~20 Hz:  read tared force & pose → control law → clamp lead → send target → events → stop?
                         │
                         ▼
     settle + still reading → estimators → tool result (quantities ± σ, window, flags) → appended to the history
```

## Action interfaces and the no-force control

`primitives/library.py` defines three menus (`INTERFACES`):

- `force`: position tools plus the force primitives (`move_until_contact`, `push`, `slide`, `search`, `insert`, `lift`, `probe`,
  and task-specific extras such as `pull`, `turn`, `point_tool`; the real robot adds `screw`, `wiggle`, `locate`; ForceBench adds
  `press_turn`, `locate`). This is ForceHarness.
- `pose`: position-only tools; motions run to completion unless a safety limit stops them, and tool results carry no force.
  With the `none` rendering this is the **no-force control** of the paper (`none|pose`).
- `guard`: position tools that stop on contact (kept for analysis).

A condition is written `fmt|interface`, e.g. `phys2|force` (ForceHarness) or `none|pose` (no-force control). Other `fmt` values
are the ablations of how force is shown to the model: `raw` (recent samples), `stats`, `events`, `signals` (events plus derived
signals), `plot` (a PNG of the trace), `phys1` (L1), `phys3`, `phys4` (L2 plus reference lines / everyday anchors); they can be
combined with `+`.

## Backends

All backends implement the contract in `backends/base.py` (`eef_pos`, `eef_quat`, `step`, `hold`, `wrench`,
`reaction_force`, ...). `python -m forceharness.doctor --backend module:Class` checks a class against it.

| Backend | Where | Layer 1 | Model access |
|---|---|---|---|
| MuJoCo (HiddenPhys) | `backends/mujoco/` | wrist force sensor + Franka-level noise (0.5 N per tick) and per-episode residual bias (0.3 N) | `agent/loop.py`, object poses as noisy text |
| ForceBench | `backends/forcebench/` (policy server for ForceBench's evaluator) | the simulator's wrist wrench (no noise, rigid links), tare rules | same loop; RGB images + camera calibration, `locate` from pixels |
| Real Franka FR3 | `forceharness/real/` (workstation) + `robot_side/franka_polymetis/` (robot PC) | `(Jᵀ)⁺ τ_ext` on the robot PC, offset calibration, tare / settle / report changes | runner + web monitor relay, wrist RGB-D for `locate` |
| Fake | `backends/fake.py` | ideal | offline tests |

## What an episode writes

`runs/<run_name>/transcripts/<episode_id>/`: `transcript.json` (every message), `reports.json` (every primitive: parameters,
termination, events, force trace, derived quantities, simulator ground truth), `record.json` (the episode summary), and images
the model saw. Batch runs append one line per episode to `runs/<run_name>/results.jsonl`; reruns skip finished episodes.

## Where the model-facing text lives

- System prompts and task instructions: `forceharness/prompts/` and the `instruction` of each task in `hiddenphys/tasks.py`
- Tool descriptions: `agent/tools.py` (simulators), `forceharness/real/` (real robot menu)
- Tool results: `translation/` (renderers) and the kinematic summary in `translation/base.py`

Changing any of these changes what the model reads, and therefore results.
