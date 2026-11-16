---
name: forceharness-setup
description: Install, verify and run ForceHarness on a new machine, simulator or robot arm. Use when asked to set up, deploy, reproduce or try ForceHarness; to connect it to a real robot; to check whether a robot can sense force well enough; to port it to another arm; or to run a first episode on HiddenPhys (MuJoCo), ForceBench (Isaac Lab) or a real robot.
---

# ForceHarness setup

You are taking a fresh clone of this repository to a **verified first episode** on the human's target: the HiddenPhys
benchmark in MuJoCo, ForceBench in Isaac Lab, or a real robot arm. Work in stages. Every stage ends at a **gate** with a
verdict (PASS, WARN, FAIL) and evidence (command, numbers, file). Report each gate to the human before moving on, and keep
a running log in `runs/setup_report.md` using [references/report-template.md](references/report-template.md).

What ForceHarness needs, in one paragraph: a frozen model calls one tool per turn; each tool is a force primitive that runs
a closed loop at the controller side (about 15 to 20 Hz) by moving a target pose ahead of the measured pose, so that a stock
compliant controller turns that lead into force; it stops itself on contact events; fixed estimators turn its force trace into
physical quantities returned as the tool result. So a target needs (1) a compliant pose-target controller, (2) a force estimate
at the hand, (3) a safe way to stop. Read [docs/architecture.md](../../docs/architecture.md) if you need more.

## If you only have the link

If the working directory is not a ForceHarness clone (no `pyproject.toml` with `name = "forceharness"`), clone it first:
`git clone https://github.com/showlab/ForceHarness.git && cd ForceHarness`, then read `AGENTS.md` and continue here. Links in
this skill are relative to `skills/forceharness-setup/`. An agent that can only fetch files finds them under
`https://raw.githubusercontent.com/showlab/ForceHarness/main/` (this file is `skills/forceharness-setup/SKILL.md`), but it
must not run anything until the repository is cloned on the machine that will run it.

## Ground rules: never break these

1. **No robot motion without the human's explicit go-ahead in this conversation** for that stage, after the human has
   confirmed that a person is at the emergency stop and the workspace is clear. Approval does not carry over to the next stage.
   Gripper motion counts as motion.
2. **Read-only first.** Every real-robot stage starts with checks that only read state.
3. **Never loosen a safety limit** (force caps, speed, lead, workspace box, envelope, validation). You may tighten them.
   Never edit `forceharness/primitives/`, `forceharness/translation/` or `robot_side/*/guard.py` to make a check pass;
   report the failure instead. (A port to a new arm gets its own copy of the guard in which only arm- and cell-specific
   constants change; see references/porting.md.)
4. **API money**: ask before the first paid model call and give the expected cost (see the path sections). Never print,
   log or commit API keys; they live in `.env`.
5. **Do not guess hardware facts.** If the arm model, controller, sensor or network is unclear, ask the human; then measure.
   The measurement decides, not a table.
6. **Stop on FAIL.** Explain what failed, why it matters for ForceHarness, and what would fix it. Do not work around it silently.

## Step 0: target and environment

1. If the human did not say, ask which target: **A** HiddenPhys in MuJoCo (any Linux or macOS machine; easiest),
   **B** ForceBench (Linux with an RTX GPU), **C** a real robot (any arm; the reference implementation is a Franka FR3 under
   Polymetis). They can pick several; do A first, it validates the harness itself without hardware.
2. Check that `python3 --version` is 3.10 or newer (a stock macOS `python3` is 3.9: use `uv venv -p 3.11 .venv`, or a
   Homebrew / conda Python). Create an environment, install the core package, then run the doctor:
   ```bash
   python3 -m venv .venv && . .venv/bin/activate
   pip install -e ".[dev]"
   python -m forceharness.doctor --json          # core + all targets; read it, then summarize PASS/WARN/FAIL per target
   python -m pytest -q                           # offline tests: no simulator, robot or key needed
   python scripts/run_episode.py --task weight_sort --agent scripted --backend fake   # one offline episode
   ```
   `doctor --json` exits with code 1 at this point because the simulator rows FAIL until Path A installs MuJoCo; judge
   Gate 0 on the core rows only.
   **Gate 0**: Python >= 3.10, core imports PASS, offline tests pass (skips are fine). FAIL: fix the install first.

## Path A: HiddenPhys in MuJoCo

1. `pip install -e ".[sim]"` (pins robosuite 1.5.1 and **mujoco 3.2.7**; newer MuJoCo breaks robosuite 1.5.1).
   Rendering is only needed for camera images (the paper's HiddenPhys episodes use none). The default is `MUJOCO_GL=egl` on
   Linux (pick a free GPU with `MUJOCO_EGL_DEVICE_ID`, whose numbering may differ from `nvidia-smi`) and `cgl` on macOS; on a
   Linux machine without a GPU use `MUJOCO_GL=osmesa`, or skip images with `check_sim.py --no-render`.
2. `python -m forceharness.doctor --target sim` must show mujoco 3.2.7 and robosuite 1.5.x as PASS.
3. `python scripts/check_sim.py --tasks weight_sort`, then without `--tasks` (five tasks). Add `--no-render` if rendering is
   not set up.
   **Gate A1**: every task PASS (force-sign check: contact with the table gives reaction Fz > 0; the expert episode ran without
   error). The expert should also succeed on most seeds; a single failure is not a FAIL.
4. Mechanism tasks: run each expert script on one seed:
   `python scripts/hiddenphys/door_latch_expert.py F --seeds 9100` and
   `python scripts/hiddenphys/lock_or_stiff_expert.py F --seeds 9100` (strategy F = expert; without `--seeds` they run all 50
   test seeds). **Gate A2**: they run and report success or a clean failure.
5. First model episode (needs an API key in `.env` and the human's OK; HiddenPhys episodes without images cost roughly
   US$0.2 to 0.6 each with GPT-6 Astra):
   ```bash
   python scripts/run_episode.py --task weight_sort --model gpt-6-astra --provider openai-responses \
       --reasoning-effort low --paper-defaults --seed 9100
   ```
   Read `runs/<run>/transcripts/<episode_id>/transcript.json` and show the human one tool call and its tool result.
   **Gate A3**: the episode ends with `done` or a budget stop and no `error`.
6. To reproduce a paper table, follow [docs/reproduce.md](../../docs/reproduce.md); always run `--dry-run` first and give
   the human the episode count and cost estimate before launching. Details and pitfalls: [references/sim.md](references/sim.md).

## Path B: ForceBench in Isaac Lab

Requirements (checked by `python -m forceharness.doctor --target forcebench`): Linux x86_64 with glibc >= 2.35
(Ubuntu 22.04), an NVIDIA GPU **with RT cores** (RTX / A6000 / L40 class; H100, H200, A100 cannot render), driver >= 550.90.07
(older drivers need an opt-in workaround with a warning), about 40 GB of disk. The human must read and accept the NVIDIA
Omniverse EULA themselves; never accept it on their behalf. Then follow [references/forcebench.md](references/forcebench.md)
and [docs/forcebench.md](../../docs/forcebench.md): install, one scripted episode (no API), one model episode (about US$0.2
with GPT-6.1 Sol, about US$1 with GPT-6 Astra, RGB images included), batch runs.
**Gate B**: the evaluator prints `FORCEBENCH_VECTOR_RESULT=` (Isaac's exit code is always 0, so do not trust it).

## Path C: a real robot

Read [references/real-robot.md](references/real-robot.md) before starting; it has the commands for the reference
Franka FR3 under Polymetis. The gates, in this order:

**C1 Identify the embodiment (no motion).** Ask and record: arm make and model, controller software and version
(e.g. Polymetis, libfranka/franky, franka_ros2, ur_rtde, Kortex, a vendor SDK, ROS 2 control), real-time PC vs workstation,
gripper, cameras (the reference runner needs a wrist RGB-D camera, used to locate objects, and an exterior RGB camera
that the model sees but nothing is measured from), and how the workstation reaches the robot PC (or whether the SDK runs on
the workstation itself).

**C2 Force source, from the specification (no motion).** Look the arm up in
[references/force-sources.md](references/force-sources.md) and confirm with the human or the vendor documentation which
force channel exists. Decide:
- **Expected OK**: a wrist 6-axis F/T sensor, or torque sensors in every joint with an external-wrench estimate. Continue.
- **Expected DEGRADED**: an industrial arm whose force comes from motor-current estimates (no torque sensors, no wrist
  sensor). Tell the human that contact detection may work but weighing, gentle placing and insertion search will likely be
  unreliable, and continue only if they agree. Before investing in a port, measure M1 to M3 of
  [references/force-check.md](references/force-check.md) read-only with a short script on the vendor SDK (see "Measuring before
  a port" there); if any test FAILs, it is NO_FORCE.
- **NO_FORCE: stop here and warn.** No wrist sensor and no joint torque sensors, and at most the coarse load or current
  readings of hobby / smart servos (Feetech, Dynamixel XL-class and similar), the motor-current or effort readings of
  low-cost arms (AgileX Piper class), or no feedback at all. Say plainly that
  ForceHarness's force primitives and physics translation cannot work on this arm; only the no-force control (`none|pose`)
  could run, and that is not the method (and even that needs a new backend). Offer: run the simulators, add a 6-axis wrist
  F/T sensor, or use an arm with joint torque sensors. A wrist sensor fixes only the force channel: C3 still applies
  (hobby-servo and low-cost arms stay stiff position-controlled), and the arm must carry the sensor's mass.

**C3 Control gate (no motion).** The arm must accept pose or joint targets streamed at >= 20 Hz, report its pose at >= 20 Hz,
and stop by holding or releasing the target. Then:
- a **compliant** controller (impedance, operational-space, admittance): PASS;
- a **stiff** position controller *and* a good force signal from C2 (e.g. a UR e-Series with its flange sensor): WARN. The
  lead of the target must be turned into compliant motion by an admittance loop on the robot side or by the vendor's
  force/compliance mode ([references/porting.md](references/porting.md)); continue only with the human's agreement;
- a stiff position controller without a usable force signal, or no streaming interface: FAIL, stop and explain.

**C4 Safety gate (no motion).** Confirm the emergency stop with the human, measure the cell, and clear the table except a soft
test object. Limits are code, not configuration. On the reference stack they are `Limits` in
`robot_side/franka_polymetis/guard.py` (8 mm/s, 25 N raw force cap, 3 N·m torque cap, 8 mm lead, workspace box x 0.20-0.65 m,
|y| < 0.30 m, z 0.12-0.55 m) and the ceilings in `forceharness/real/config.py`, which must match at the handshake. If the cell
needs a smaller box or lower values, tighten both identically and redeploy. Never loosen an existing limit to make a check
pass. For a ported arm, set the box from that arm's cell (it is a property of the cell, not of the method) and keep the force
and torque caps, speed and lead at or below the reference values.

**C5 Backend (no motion).** Franka FR3 or Panda with Polymetis: copy `configs/real/robot.example.json` to
`configs/real/robot.json`, fill in the site values (robot-PC ssh alias, Polymetis paths and ports, camera serials, hand-eye
file), deploy `robot_side/`, and run its self-test and compile check, which send nothing to the arm. Any other arm: port it as
described in [references/porting.md](references/porting.md) (the port brings its own configuration; `robot.json` is specific
to the reference stack). Route 1 (a workstation backend): check it with `python -m forceharness.doctor --backend
<module:Class>` and offline tests on a fake plant. Route 2 (a new robot-side worker): its self-test and offline tests that
serve the worker over a pipe, as in `tests/real/`. Agree on the route with the human before writing code. Route 2 is the
recommended route for real tasks: Route 1 gives only the simulator-style episode loop, and `scripts/run_episode.py --backend`
accepts only `robosuite|fake`, so a Route 1 port also needs its own short runner (porting.md lists what the loop reads).
**Hand-eye (no motion)**: the reference runner locates objects with the wrist camera. Put an object where you can measure its
position in the base frame and check that the located position is within 5 mm; otherwise redo the hand-eye calibration
([docs/real_robot.md](../../docs/real_robot.md)) before C7.

**C6 Force gate, measured (no arm motion; a gripper close only after a go-ahead).** Measure with
[references/force-check.md](references/force-check.md): static noise and drift with the arm still, the sign by a human pushing
the fingertips up from below, a known mass. Verdict:
- **FORCE_OK**: write the calibration file and continue.
- **FORCE_DEGRADED**: warn the human in plain words which ForceHarness behaviours will be unreliable and continue only if they
  agree.
- **NO_FORCE** (the stream cannot be read at >= 20 Hz, or any FAIL): **stop and warn**, as in C2.

**C7 Staged bring-up**, each stage after its own go-ahead: S2 tiny free motions (1 cm at 8 mm/s); S3 first contact onto a soft
object (`move_until_contact`); S4 primitive checks against references (press 5 N on a kitchen scale, lift a known mass);
S5 one task episode with the model through the web monitor, the human watching.
**Gate C**: each stage PASS with numbers in the report.

## When you finish

Give the human the gate table from `runs/setup_report.md`: what passed, what is only warned, what was not run and why, and
the next command they can run themselves. Common problems and fixes: [references/troubleshooting.md](references/troubleshooting.md).
