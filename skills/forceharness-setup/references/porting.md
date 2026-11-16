# Porting ForceHarness to another arm

Two routes. Pick by where the arm's SDK has to run.

## Route 1: one Python backend on the workstation

Use this when the SDK runs on the workstation and gives you a compliant controller and a force reading at >= 20 Hz
(for example an arm with a vendor Python SDK and a wrist F/T sensor).

1. Implement the contract in `forceharness/backends/base.py`. Required: `dt`, `action_dim`, `eef_pos`, `eef_quat`
   (`[x, y, z, w]`), `gripper_opening`, `sim_time`, `step(dpos, drot, grip)`, `hold(n, grip)`, `wrench()`,
   `reaction_force(w)`. Strongly recommended: `rezero_bias(steps)` (tare), `set_kp(kp)` and `kp` (stiffness),
   `wrist_joint_rad`.
   - `step` is one control tick: move the **target** pose by `dpos` / `drot` and send it to the compliant controller; do not
     wait for the arm to arrive. The primitives create force by letting the target lead the measured pose (paper Eq. 2), so the
     backend never needs its own force controller.
   - `reaction_force` must return the force of the environment on the hand in the base frame, tared, with pressing down on a
     table giving +z. Check it with the M3 sign test of [force-check.md](force-check.md).
   - Units: metres, radians, newtons, seconds.
2. Put every safety limit in the backend, below the harness, so the model cannot reach it: raw force and torque caps, a
   workspace box, a maximum target speed, a maximum lead of target over measured pose, and a stop that holds or releases
   the target. Copy the values of the reference guard (`robot_side/franka_polymetis/guard.py`) or tighter.
3. Check: `python -m forceharness.doctor --backend yourpkg.module:YourBackend` (static contract check), then write a small
   offline test with a fake plant like `tests/real/` does, then the staged bring-up in [real-robot.md](real-robot.md).
4. Episodes: the simulator episode loop (`forceharness/agent/loop.py`) also reads `task` (a `TaskSpec`; the gripper tools
   check `task.has_gripper`), `scene()`, `success()`, `task_state`, `peak_wrench`, `peak_contact_force` and, for images,
   `images()`. `doctor --backend` does not check these. `scripts/run_episode.py --backend` accepts only `robosuite|fake` and
   uses the HiddenPhys system prompt, so write a short runner for your backend, starting from `scripts/run_episode.py`.
   For a real arm you will also want the perception and validation of `forceharness/real/runner.py`, which is why Route 2
   is recommended for real-robot tasks.

## Route 2: keep the reference workstation side, replace the robot PC side

Use this when the arm needs a real-time computer next to it (like the FR3) or you want the reference runner, monitor,
perception and validation unchanged (recommended for real-robot tasks: Route 1 only gives you the simulator-style episode
loop). If the arm has no separate robot PC (e.g. a UR arm driven by ur_rtde from the workstation), run the worker on the
workstation: see "Arms without a separate robot PC" in the protocol document. The workstation (`forceharness/real/backend.py`, `transport.py`) talks to a worker on
the robot PC through the JSON-lines protocol in [docs/robot_protocol.md](../../../docs/robot_protocol.md). Port the robot side:

1. Write a worker for your SDK that speaks the same protocol (same commands, fields, units, lease and fault semantics). Start
   from `robot_side/franka_polymetis/worker.py`; replace only `driver.py` (SDK calls, IK, controller start/stop) and
   `wrench.py` (your force estimate: a wrist sensor reading rotated into the base frame, or `(J^T)^+ tau_ext`).
2. Copy `guard.py` for your arm and change only the arm- and cell-specific constants (joint count, joint limits, workspace
   box); keep every force, torque, speed, lead, lease and envelope value at the reference value or tighter, and have the
   human review the diff. The handshake makes the workstation refuse a guard whose limits differ from its configuration.
3. Adapt the workstation code that names the FR3 (the checklist in section 9 of the protocol document): the handshake's
   flange link, the site-file schema, the kinematic model used for validation (the runner walks every path before moving),
   wrist range and TCP, gripper widths, and the model-facing text that names the arm. Beyond `driver.py` and `wrench.py`,
   these places are also specific to the FR3 under Polymetis (the protocol checklist does not list all of them yet):
   - `forceharness/real/transport.py`: `ROBOT_SIDE` and `FILES` point at `robot_side/franka_polymetis`, and `bundle_command`
     passes Polymetis flags (hardware config, conda prefix, ports).
   - `robot_side/franka_polymetis/worker.py` also uses `driver.make_policy`, `driver.torch`, `driver.robot` and
     `driver.telemetry` (compile check and state reads), beyond the driver interface in the protocol checklist.
   - `forceharness/real/cameras.py` reads Intel RealSense cameras only.
   - `forceharness/real/kinematics.py` and `posture_bias.py` model the 7-joint FR3; the runner validates every path with
     them, so replace them with your arm's kinematics. Continuous joints (e.g. Kinova) need their own handling in the
     guard's joint box.
   - `forceharness/real/config.py` accepts only Franka Hand gripper values (force <= 70 N, 78 to 80 mm open); another
     gripper needs its own limits there and in the guard.
4. If the arm's controller is stiff (UR e-Series with `servoL`, the Kortex high-level API, xArm position mode), the driver
   must make it compliant, because the primitives create force by letting the target lead the measured pose. Run a virtual
   spring-damper in the driver, at 50 Hz or faster, between the harness target `x*` and the pose `x` you send to the arm:
   `D dx/dt = K (x* - x) + F_env`, where `F_env` is the tared force of the environment on the hand (the `reaction_force`
   convention: pressing down on a table gives +z). Clip `dx/dt` to the guard speed and `x* - x` to the guard lead. In free
   space `x` follows `x*`; against a rigid surface it stops and the hand presses with `K (x* - x)`. Choose `K` near the
   reference pressing stiffness (about 0.75 N/mm on the FR3), report it as `kp` / `set_kp`, and choose `D` so that `D / K`
   is a few control ticks. The alternative is the vendor's force or compliance mode (e.g. ur_rtde `forceMode`). Tare with
   the vendor call (e.g. ur_rtde `zeroFtSensor`) and configure the payload. This recipe has not been run on hardware:
   validate it with stages S2 to S4 of [real-robot.md](real-robot.md) before any task, and say so in your documentation.
5. Check Route 2 with the worker's self-test (no motion) and offline tests that serve your worker over a pipe with a
   synthetic plant (like `tests/real/`), not with `doctor --backend` (that checks Route 1's contract). Then the force gate
   and the staged bring-up.

## What you must not change

The primitives (`forceharness/primitives/library.py`), the estimators and renderers (`forceharness/translation/`) and the
prompts are the method. In the prompts change only what describes your hardware (the arm's name, gripper, cameras), and
record that change. Tune constants that describe your hardware (stiffness, contact thresholds, speeds) in configuration,
never the safety limits upward.
