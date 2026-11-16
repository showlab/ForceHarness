# Robot-side protocol (`fh_continuous/2`) and porting guide

The real-robot backend is split in two processes. The **workstation** runs the harness (`forceharness/real/`): the
model relay, perception, validation and the force primitives, which decide about every 50 ms where the hand's target
should go next. The **robot PC** (the real-time computer next to the arm) runs one **worker** per episode
(`robot_side/franka_polymetis/worker.py`), which owns the arm controller and applies every hard limit itself, whatever
the workstation asks. They talk over the stdin/stdout of one SSH session in JSON lines. This page specifies that
protocol as implemented, then explains how to port the robot side to another arm.

Error messages and the per-episode log file (`nuc.jsonl`) call the robot PC "NUC".

```
workstation                                         robot PC
forceharness/real/transport.py   ssh, JSON lines    robot_side/franka_polymetis/
  SSHPort.request(op, ...)  ───────── stdin ──────►   worker.py   one guarded session, 50 Hz tick
                            ◄──────── stdout ──────     guard.py    limits and checks (no hardware imports)
                                                        driver.py   Polymetis: state, IK, controller start/stop
                                                        wrench.py   w = (J^T)^+ tau_ext at the flange
  RemoteGripper.run(op)     ── one ssh command ──►    gripper.py   one Franka Hand action, one JSON line
```

Units are SI (m, rad, N, N·m, s); positions and rotations are in the robot base frame (z up); quaternions are
`[x, y, z, w]`.

## 1. Starting the worker

The workstation starts the worker from a deployed, content-addressed bundle (section 8) with one command:

```
ssh -o BatchMode=yes -o ConnectTimeout=5 [ssh_options] <ssh_host> <remote command>
```

where the remote command, run by the robot PC's shell, is (one line in practice; every value is shell-quoted)

```
cd <deploy_dir>/robot_side_<hash> && sha256sum --status -c SHA256SUMS && export CONDA_PREFIX=<conda_prefix> &&
exec <python> -u worker.py --hardware-config <franka_hardware.yaml>
  --limits-json '{"speed": 0.008, "f_cap": 25.0, "t_cap": 3.0}' --robot-port 50051 --gripper-port 50052 [--enable-motion]
```

| Flag | Meaning |
|---|---|
| `--enable-motion` | Without it the worker is **read-only**: it answers `state` and `close` and treats every other request as a fault. |
| `--hardware-config` | Polymetis robot-client configuration; its `executable_cfg.limits.joint_pos_lower/upper` are intersected with the worker's own joint box. |
| `--limits-json` | Tighter limits for this session (section 6). A looser value makes the worker exit before the handshake. |
| `--robot-port`, `--gripper-port` | Polymetis robot and gripper servers on the robot PC itself (`127.0.0.1`). |
| `--compile-only` | Connect, read one state, compile and run the controller once offline, print `{"type":"compile_check",...}` and exit. Never sends a controller. |

One worker per deploy folder at a time (an exclusive `flock` on `<deploy_dir>/.worker.lock`). Closing stdin ends the
session (section 7).

**Handshake.** The first line on stdout is

```json
{"type": "ready", "version": "fh_continuous/2", "motion_enabled": true, "metadata": {...}, "limits": {...}}
```

The workstation waits up to 20 s for it and refuses the session unless `version` is `fh_continuous/2`,
`motion_enabled` equals what it asked for, `metadata.frame` is `"world"` and `metadata.reference` is `"panda_link8"`;
then the backend refuses it if `limits.speed`, `limits.f_cap` or `limits.t_cap` is larger than its own configuration.
`limits` is the full set in force (section 6). `metadata` must contain `default_Kx` and `default_Kxd` (6 positive
values: the controller's Cartesian gains at kp = 150), `joint_low` and `joint_high` (7 values each, the joint box in
force), `frame` and `reference`; the Polymetis driver adds descriptive fields (`hz`, `default_Kq`, `default_Kqd`,
`controller`, `use_grav_comp`, `frame_errors`, ...), which are only logged.

## 2. Requests and replies

A request is one JSON object per line, at most 16384 characters:

```json
{"id": 7, "op": "step", "dpos": [0.0, 0.0, -0.0004], "drot": [0.0, 0.0, 0.0],
 "tcp_translation": [0.0, 0.0, 0.1185], "tcp_quaternion": [0.0, 0.0, -0.38268343, 0.92387953]}
```

`id` is an integer that increases with every request; only one request may be outstanding. A repeated or smaller id,
or a request sent while a `step` reply is pending, is a fault. A successful request is answered with

```json
{"type": "reply", "id": 7, "ok": true, "session": {"mode": "active", "controller_starts": 1, "holds": []}, "state": {...}}
```

`session.mode` is `idle` (no own controller), `active` (a primitive runs) or `holding` (the own controller holds the
last target between primitives); `holds` lists every hold of the session with its anchor, target, largest displacement,
target error and force change. There are no error replies: a failed request is a **fault** (section 7).

| `op` | Fields | Precondition | Effect | Reply | Workstation timeout |
|---|---|---|---|---|---|
| `state` | | none (also read-only) | one checked state read | `state` | 1 s |
| `close` | | none (also read-only) | `end`, then the worker exits | no `state` | 4.5 s |
| `set_kp` | `kp`: number or 6 numbers in (0, 150] | no controller running, or the same gains | gains of the next controller: Kx = default_Kx·kp/150, Kxd = default_Kxd·sqrt(kp/150) | `kp`, `Kx`, `Kxd`, `joint_gain_ratio` | 1 s |
| `begin` | `budget`: s in (0, 300] | idle; no other controller running; every joint slower than 0.03 rad/s | start the stock controller with the target at the measured flange pose; record the primitive's start pose, joints, gripper width, raw force and residual | `state` | 15 s |
| `step` | `dpos` (3, m), `drot` (3, rad), `tcp_translation` (3, m), `tcp_quaternion` (4), optional `lead` (m) | active, not holding | move the TCP target (section 4) | `state`, one tick (50 ms) later | 0.3 s |
| `hold` | | active | release ramp (section 5), then keep the controller holding the released target | `state` | 4.5 s |
| `resume` | `budget`, `kp` (6) | holding; `kp` equal to the running gains | start a new primitive on the running controller (start pose, joints, width, force and residual re-recorded) | `state` | 1 s |
| `finish` | | any | release ramp, then terminate the controller (`normal_stop`); if idle, only `end` | `state` | 4.5 s |
| `end` | | any | terminate the own controller immediately, no ramp | `state` | 1 s |

A reply that does not arrive within the timeout ends the session on the workstation ("reply timeout; outcome unknown"),
which closes stdin.

## 3. The `state` object

Every `state` comes from one Polymetis `RobotState` plus one gripper state, after all checks of section 6 passed.

| Field | Unit / frame | Used by |
|---|---|---|
| `timestamp_ns` | int, robot clock | guard (must advance), workstation (fresh readings) |
| `flange_pos`, `flange_quat` | m; xyzw; `panda_link8` in the base frame, from forward kinematics of the measured joints | everything |
| `joints`, `velocities` | rad, rad/s (7 each) | guard, validation, the relaxed still reading |
| `wrench` | `[fx, fy, fz, mx, my, mz]`, N and N·m, base-frame axes, moment about the flange origin; raw sign of `(J^T)^+ tau_ext` (calibrate `force_sign`) | guard caps, all force use |
| `smin` | smallest singular value of the 6x7 Jacobian | guard (conditioning) |
| `residual` | N·m, norm of the part of `tau_ext` the wrench does not explain | guard |
| `gripper_width` | m, from the gripper server | guard, gripper checks |
| `policy_running` | bool: any controller runs on the server | guard, gripper preconditions |
| `error_code` | int, robot error code (0 = none) | guard |
| `motor_torques_external` | 7 values, N·m (`tau_ext`) | the optional posture model of the force zero |
| `episode_start`, `episode_end`, `target_controller_active`, `rpc_accepted_joint_target`, `last_update_index`, `controller_latency_ms`, `command_successful`, `joint_torques_computed`, `prev_joint_torques_computed`, `prev_joint_torques_computed_safened`, `motor_torques_measured`, `motor_torques_desired` | Polymetis diagnostics | logged only |

The wrench: `RobotState.motor_torques_external` is libfranka's filtered external torque estimate, from which libfranka
has already subtracted the end-effector load configured in Desk; `w = (J^T)^+ tau_ext` with the 6x7 Jacobian at
`panda_link8`, base-aligned. At start-up the driver checks the Jacobian's frame against finite differences of the
forward kinematics and refuses to run unless it is world-aligned. The workstation applies `force_sign`, the bias and the
closed-gripper correction from the force calibration file, and tares the reading from still readings before contact.

## 4. Motion: what a `step` does

Each `step` asks for an increment of the **TCP target**: `dpos` in the base frame and `drot` as a rotation vector in
the base frame about the TCP. The TCP is the flange pose times `tcp_translation` / `tcp_quaternion` (the runner sends
the centre of the finger pads, 0.1185 m along the flange z axis, and the Franka Hand's -45° about z). The worker:

1. rejects (fault) a request with `|tcp_translation| > 0.25 m`, `|dpos| > 0.06 m` or `|drot| > 0.10 rad`;
2. clips the increment to the target speed: translation `speed · window`, rotation `angular_speed · window`, where
   `window` is the real time since the last accepted target, at most 0.08 s (both bounds hold together, by one
   common interpolation fraction);
3. clips the flange target to within `lead` (8 mm) of the **measured** flange position. The optional `lead` field may
   ask for up to `contact_lead` (25 mm), granted only once the raw force has changed by at least 3 N since `begin`
   (the hand is pressing on something); a granted larger lead shrinks again at the target speed, never in one jump;
4. checks the flange target against the workspace box and the per-primitive envelope (at most 120 mm and 0.60 rad from
   the `begin` pose);
5. on the next tick, solves IK on the robot PC (Polymetis' model, seeded with the previous joint target, at most
   100 ms) and accepts the joint target only if it is finite, at least 0.03 rad inside the joint box, within 0.9 rad
   of the start joints, within 0.05 rad of the previous joint target and of the measured joints (more, in proportion,
   for a granted contact lead), and reproduces the pose to 0.2 mm; then it sends it with
   `update_desired_joint_positions`.

Force is never commanded directly: with a compliant controller, a target that leads the measured pose into an obstacle
produces a contact force, and the primitives on the workstation regulate it through the lead (paper Eq. 2). The
controller is the stock `torchcontrol.policies.HybridJointImpedanceControl`, unchanged; its gains are fixed for the
lifetime of one controller (change them with `set_kp` between controllers).

**Pacing.** The worker ticks at 50 Hz on its own. The workstation sends `step` at 15 to 20 Hz (each reply waits one
tick). While a primitive is active, a gap of more than 0.30 s between steps (the **lease**) is a fault: the arm never
keeps following a target nobody updates. Between primitives the workstation either holds (`hold`, no lease; the
runner renews a hold with one zero-length motion after 45 s while something is gripped) or stops (`finish`).

## 5. Ending a motion: release ramp, hold and resume

`finish` and `hold` first **release** the target: the target moves from where it was back onto the measured pose over
a nominal 0.3 s (at the unchanged target speed, so a 25 mm pressing lead takes about 3.1 s; the deadline is
`max(2.5, 0.5 + contact_lead/speed)` = 3.6 s), so the controller stops pushing before anything else happens. Then
`finish` terminates the controller (Polymetis then runs its default controller, which holds the arm where it is) and
`hold` keeps it running at the released target. While holding, every tick checks the TCP and the force: a TCP more than
5 mm from the hold target stops the controller at once (`hold_unexpected_motion`); a force change of more than 8 N for
3 consecutive ticks releases the target, stops the controller and faults (`hold_contact`); after 60 s of holding the
controller is terminated (no ramp) and the session is idle. `resume` starts the next primitive on the same controller, which avoids the sag of a stop and
restart under load.

## 6. Limits and checks

All limits live in `robot_side/franka_polymetis/guard.py` (`Limits`). `--limits-json` may only tighten them:
`joint_margin` and `smin` may only grow, every other value may only shrink; unknown keys, non-finite values and
`dt` other than 0.05 are rejected. The workstation sends `speed`, `f_cap` and `t_cap` from its configuration (and
`width_check: 0` for a task whose tool is clamped by the operator).

| Limit | Value | Meaning |
|---|---|---|
| `dt` | 0.05 s | nominal step interval (fixed) |
| `speed`, `angular_speed` | 0.008 m/s, 0.15 rad/s | target speed |
| `lead`, `contact_lead` | 0.008 m, 0.025 m | target ahead of the measured flange; contact lead only after a 3 N rise |
| `tracking` | 0.015 m | flange to target distance (plus any extra lead granted) |
| `f_cap`, `t_cap` | 25 N, 3 N·m | raw force and moment norm: immediate stop |
| `displacement`, `rotation`, `joint_delta` | 0.15 m, 0.60 rad, 0.9 rad | measured travel per primitive |
| `duration` | 300 s | budget of one primitive |
| `stall_seconds` | 4 s | target moving but the flange making no progress (error above 2 mm) |
| `stale` | 0.15 s | robot timestamp not advancing, or a tick missed |
| `lease` | 0.30 s | longest gap between steps of an active primitive |
| `joint_margin` | 0.03 rad | distance to the joint box (FR3 ∩ Panda ∩ hardware configuration) |
| `smin`, `residual`, `residual_rise` | 0.01, 3 N·m, 1 N·m | Jacobian conditioning, wrench residual and its rise within one primitive |
| `width_check` | 1 | 0 skips the gripper-width range and change checks (a tool clamped by the operator) |

Fixed checks on every state read: finite values and a unit quaternion; flange inside the box 0.20 < x < 0.65 m,
|y| < 0.30 m, 0.12 < z < 0.55 m (fingertips about 2 cm above a table at z = 0 when pointing down); gripper width in
[0, 0.085] m; robot error code 0; gripper error code 0 and arm/gripper timestamps within 350 ms; the controller still
the one this worker started. While a primitive runs: the gripper width unchanged within 3 mm, the controller still
running. The box assumes the table top near z = 0 of the base frame; if your table is higher, shrink the box (never
widen it).

## 7. Faults, disconnects and crashes

Any exception (a failed check, a refused request, a request that does not fit the precondition) is a fault. The worker:

1. for the motion-envelope faults (primitive duration exceeded, tracking stalled, requested step or per-primitive
   desired pose exceeds the envelope, tracking error, joint margin, per-primitive displacement / rotation / joint delta,
   flange workspace) while a primitive is active, first prints `{"type":"fault_release_start","error":...}` and runs the
   release ramp; the workstation then waits up to 3.2 s more instead of closing stdin;
2. for every other fault (raw force or moment cap, stale or invalid state, hardware error, lease missed, conditioning,
   lost connection) terminates the controller immediately;
3. latches the fault, prints `{"type":"fault","error":...}` and `{"type":"error","error":...}`, ends the session and
   exits. A session never continues after a fault; the runner ends the episode.

**Disconnect.** When stdin closes (the workstation closed the port, or SSH dropped), the worker ends the session: an
own controller is terminated (a release ramp in progress is abandoned) and the worker exits.

**Crash.** If the worker process dies, nothing terminates its controller: the stock controller has no heartbeat
watchdog and keeps holding its last joint target. The person at the emergency stop covers this case.

**Telemetry.** Besides `ready`, `reply`, `fault_release_start`, `fault` and `error`, the worker prints rows the
workstation only logs (to `<episode>/nuc/nuc.jsonl`, stderr to `nuc.stderr`): `sample` (every checked state),
`target` (every accepted step), `timing` (read, IK and update RPC durations), `controller_start`,
`joint_target_accepted`, `begin`, `resume`, `release_start`, `normal_stop`, `hold_start`, `hold_end`, `hold_timeout`,
`gains`, `end`.

## 8. Deployment and the gripper command

**Bundle.** `scripts/real/deploy_robot_side.py --deploy` packs `guard.py`, `driver.py`, `worker.py`, `selftest.py`,
`wrench.py` and `gripper.py` with a `SHA256SUMS` file into `<deploy_dir>/robot_side_<hash>`, where `<hash>` is the
first 12 hex digits of the SHA-256 of the files' hashes. An existing bundle of the same name is only verified, never
overwritten; every worker or gripper command re-verifies the checksums before it runs. The workstation always starts
the bundle that matches its own copy of `robot_side/franka_polymetis/`, so changing any robot-side file needs a new
deploy (`deploy_robot_side.py` without flags prints the current bundle name).

**Gripper.** The worker never commands the gripper. A grasp or release is one separate command on the robot PC,
run from the same bundle while the worker holds the arm (or while no controller runs):

```
python gripper.py state
python gripper.py close --force 60 --speed 0.05 --port 50052     # Polymetis GripperInterface.grasp
python gripper.py open --width 0.08 --speed 0.05 --port 50052    # GripperInterface.goto
```

It waits until the hand has moved and stopped (`is_moving` seen true then false, or the width changed by more than
2 mm and then stayed still for 0.5 s; at most 8 s) and prints one line:

```json
{"op": "close", "width_before_m": 0.08, "force_N": 60.0, "above_continuous_rating": false, "width_m": 0.0291,
 "is_grasped": true, "is_moving": false, "prev_command_successful": true, "error_code": 0, "settled": true,
 "speed_m_s": 0.05, "elapsed_s": 2.41}
```

`--force` is limited to 140 N (the Franka Hand's peak; 70 N is its continuous rating); the runner uses 60 N. On the
workstation, `RemoteGripper` runs the command and, until it returns and for at least 3.2 s, reads the worker's `state`
every 50 ms, so every guard check keeps running while the hand moves; a state read slower than 0.3 s or a width outside
0 to 80.2 mm is an error. The action is accepted only if `prev_command_successful` is true and, after a close, the width
is above the minimum grasp width (8 mm) and below 75 mm (the runner reports fingers that met as an empty grasp), or,
after an open, at least 78 mm. At least 3 s pass between two commands, one close is allowed per grasp attempt, and
nothing is retried: an error leaves the held state unknown and the backend refuses further gripper actions. This path
replaces the gripper service used for the paper's runs and has **not yet been validated on hardware**.

## 9. Porting the robot side to another arm

Keep the workstation side, the protocol and the guard's checks; replace what touches the arm. The skill
`skills/forceharness-setup/references/porting.md` describes the alternative of a single backend on the workstation, lists the
workstation files that are still specific to the FR3 under Polymetis (transport, cameras, kinematics, gripper limits), and
gives an admittance recipe for arms with a stiff position controller.

**The guard for a new arm.** `guard.py` encodes the FR3's joint count, joint limits and the reference cell's workspace box
next to the arm-independent checks. Copy it for the new arm and change only those arm- and cell-specific constants; keep
the force and torque caps, speed, lead, tracking, lease, envelope and hold values at or below the reference values, and have
a person review the diff before the first deployment.

**Arms without a separate robot PC.** The workstation starts the worker with `ssh <robot_pc.ssh_host> <command>`
(`transport.ssh_argv`, `transport.bundle_command`). If the arm's SDK runs on the workstation itself (for example a UR
arm with ur_rtde), either point `robot_pc.ssh_host` at the workstation (key-based ssh to localhost) or add a local
mode that runs the same command without ssh; the protocol over stdin/stdout stays the same.

**What the arm must offer.** A compliant controller (joint or Cartesian impedance, operational space, or admittance)
whose target can be updated at 20 Hz or faster; joint (or flange) state with a timestamp at 50 Hz or faster; a force
estimate at the hand (a wrist F/T sensor, or external joint torques); a way to stop that holds or releases the target;
and a kinematic model. A stiff position controller alone is not enough: add an admittance stage in the driver that turns
the target's lead into compliant motion, and say so in your documentation.

**What to write.**

1. A driver with the interface `guard.GuardedSession` uses:
   - `metadata()`: `default_Kx`, `default_Kxd` (the gains that `kp` = 150 maps to), `joint_low`, `joint_high`, and
     the descriptive fields you have;
   - `read()`: the `state` fields `timestamp_ns`, `flange_pos`, `flange_quat`, `joints`, `velocities`, `wrench`,
     `smin`, `residual`, `policy_running`, `error_code`, `gripper_width` from one consistent sample (for a wrist F/T
     sensor: the sensor wrench rotated into base axes and moved to the flange origin; an estimate without a Jacobian has
     no conditioning or residual, so report `smin` = 1 and `residual` = 0 and say so in your documentation); add
     `motor_torques_external` only if you use the optional posture model (`forceharness/real/posture_bias.py`);
   - `start(kp, kd)`, `update(pose, kp, kd, heartbeat)` (flange pose `[x, y, z, qx, qy, qz, qw]`), `stop()` and
     `reset_primitive(state)`, with the IK and continuity checks of section 4 or stricter.
2. A `worker.py` that is this one with your driver imported; keep the request handling, the 50 Hz tick, the handshake
   and the stdin-EOF behaviour exactly.
3. A gripper command with the same output line, or remove grasping from the task menus.
4. On the workstation, everything that names the FR3:
   - `forceharness/real/transport.py`: the handshake accepts only `metadata.frame == "world"` and
     `metadata.reference == "panda_link8"`; set the reference to your flange link;
   - `forceharness/real/config.py`: the site-file schema (`load_robot_config`; the `robot_pc` block is Polymetis-specific,
     and the `cameras` block requires a wrist RGB-D camera and an exterior RGB camera) and the ceilings that must match
     the guard;
   - `forceharness/real/kinematics.py`: forward kinematics, IK and joint limits used by the validation (the runner walks
     every path with it before anything moves);
   - `forceharness/real/checks.py` and `forceharness/real/primitives.py`: wrist-joint range and turn limits (the FR3's
     joint 7), the TCP (`BackendConfig.tcp_translation` / `tcp_quaternion`, `wrist_overlay.PAD_CENTRE_TCP_Z`);
   - `forceharness/real/gripper.py`: the Franka Hand widths (empty below 8 mm, open at 80 mm) and grasp force;
   - the workspace box in both `guard.workspace` and `forceharness/real/geometry.check_flange_workspace`;
   - model-facing text that names the arm (`forceharness/real/receipt_text.py`, `forceharness/real/menu.py`,
     `forceharness/prompts/real/*.md`): update it to your hardware, and note that this changes what the model reads.

**Keep these unchanged or tighter**: every value of section 6, the release ramp, the hold checks, the fault handling of
section 7, the read-only default and the one-worker lock. Do not raise the 25 N raw cap or the 8 mm/s speed to make a
task work; tune the controller's stiffness through `set_kp` instead.

**Check before moving.** Run the offline tests (`python -m pytest -q tests/real` on the workstation; they include a
worker served over a pipe by the synthetic plant; add the same kind of test for your driver), your own self-test on the
robot PC, then the read-only force gate
(`scripts/real/check_force.py`) and the staged bring-up of [real_robot.md](real_robot.md).
