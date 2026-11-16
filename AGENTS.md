# Instructions for coding agents

This repository is ForceHarness: a harness that lets a frozen frontier model (reached only through an API) use wrist force
for contact-rich manipulation, the HiddenPhys MuJoCo benchmark, a ForceBench adapter, and a real-robot reference
implementation for a Franka FR3 under Polymetis. Start with README.md for the overview.

## If you are asked to install, set up, deploy, reproduce or try ForceHarness

Follow the setup skill: **[skills/forceharness-setup/SKILL.md](skills/forceharness-setup/SKILL.md)**. It covers the
simulators and real robots, including how to check whether an arm can sense force at all, and it defines gates you must
report to the human. Read it completely before running anything that installs packages, spends API credit or talks to a robot.

## Rules that always apply

- **Never move a robot** (arm or gripper) without the human's explicit go-ahead in the conversation for that step, with a
  person at the emergency stop. Read-only checks come first.
- **Never loosen safety limits** (force caps, speeds, lead, workspace box, validation). Tightening is fine.
- **Ask before paid API calls** and state the expected cost. Never print, log or commit keys; they belong in `.env`.
- **Model-facing text is part of the method.** Changing anything in `forceharness/prompts/`, tool descriptions
  (`forceharness/agent/tools.py`, `forceharness/real/`), or tool-result rendering (`forceharness/translation/`) changes
  results; say so whenever you do it.

## Code map

- `forceharness/primitives/` self-terminating force primitives (shared by all backends)
- `forceharness/translation/` physics translation layer (estimators + rendering) and ablation forms
- `forceharness/agent/` API clients, tool schemas, episode loop
- `forceharness/backends/` backend contract (`base.py`), MuJoCo, ForceBench, offline fake
- `forceharness/real/` and `robot_side/` real robot (workstation side, robot PC side); protocol in `docs/robot_protocol.md`
- `forceharness/hiddenphys/` benchmark tasks; `forceharness/baselines/` reference controllers and learned-policy code
- `configs/` one file per paper result; `scripts/` entry points; `monitor/` web GUI; `docs/` documentation

Conventions: SI units (m, rad, N, s), world frame = robot base frame with z up, quaternions `[x, y, z, w]`, reaction force =
force of the environment on the hand (pressing down on a table gives +z).

## Checks before you hand work back

```bash
python -m pytest -q                         # offline, no simulator / robot / key needed
python -m pytest -q monitor/tests           # web monitor
python -m forceharness.doctor --target sim  # environment summary (add --target forcebench / real as needed)
```
