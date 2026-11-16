# Real robot: Franka FR3 under Polymetis

This is the reference implementation used for the paper's real-robot experiments: six contact-rich tasks, each run
with ForceHarness (condition **A**, `phys2_force`: force primitives whose tool results carry the physical quantities
computed from the force trace) and with the no-force control (condition **B**, `none_guard`: the same robot, guards,
photos and validation, position actions only, tool results without force). This page covers the hardware, installation,
configuration, the force gate and calibration, deployment, running episodes and the safety limits. The protocol between
the workstation and the robot PC, and how to port it to another arm, are in [robot_protocol.md](robot_protocol.md).
Coding agents follow the staged procedure of the setup skill ([skills/forceharness-setup](../skills/forceharness-setup/SKILL.md)),
which uses the commands on this page.

> **Safety.** Every live run needs a person at the emergency stop and a clear workspace. Everything up to section 7
> is read-only except the gripper close of the mass check, which asks first. Never loosen a limit (section 10); tighten
> only.

## 1. What you need

| Part | Reference setup | Notes |
|---|---|---|
| Arm | Franka Research 3 with the Franka Hand, FCI enabled | Keep the end-effector settings in Desk at the Franka Hand: libfranka subtracts the configured load from the external torques the force is computed from. A Panda has the same kinematic chain but other joint limits; it is untested. |
| Fingers | stock fingers with pads | The TCP is the centre of the pads, 0.1185 m along the flange z axis (Desk TCP 0.1034 m + 15 mm; `PAD_CENTRE_TCP_Z` in `forceharness/real/wrist_overlay.py`). Other fingers need that value changed. |
| Robot PC | real-time Linux kernel, Polymetis with the `franka_hardware` robot client and the `franka_hand` gripper client | Polymetis' Python environment, 3.8 or later (polymetis, torchcontrol, torch, numpy, scipy, pyyaml, grpcio), `sha256sum`, an SSH server. ForceHarness installs nothing there. |
| Workstation | Linux, Python >= 3.10, this repository | SSH client with key-based login to the robot PC, USB 3 ports for the cameras. |
| Wrist camera | Intel RealSense D405 on the hand, looking along the fingers | RGB-D 640x480: object detection, depth at a pixel, grasp geometry. Needs a hand-eye calibration (section 4.3). |
| Exterior camera | Intel RealSense D435 viewing the workspace | RGB 640x480, shown to the model only; its depth and pose are not used. |
| Table | top near z = 0 of the robot base frame | The robot-side workspace box keeps the flange above z = 0.12 m (fingertips about 2 cm above such a table). With a higher table, shrink the box (section 10). |

## 2. How an episode runs

```
workstation                                                robot PC
monitor or a person ── relay files ──► runner              Polymetis robot server :50051 ── arm (FCI)
  (cmd_N.json)         (obs_N.md)      scripts/real/<task>.py   Polymetis gripper server :50052 ── Franka Hand
                                        │  validation, perception, primitives, tool results
                                        ├── one ssh session per episode ─► worker.py (guard, IK, controller)
                                        ├── one ssh command per grasp / release ─► gripper.py
                                        └── RealSense D405 (wrist) + D435 (exterior)
```

1. The runner starts the robot-side worker over SSH with motion enabled, checks its limits, starts the controller at
   the current pose, reads the arm still and records the zero of the force reading. The arm stays where it is.
2. Each turn it takes fresh photos with both cameras (in a separate capture process), detects the objects on the
   table in the wrist RGB-D image, and writes the observation into the relay folder (`obs_N.md` and the images).
3. The model side answers with one tool call in `cmd_N.json`: the web monitor calling a model (see
   [monitor.md](monitor.md)), or a person. The runner never calls a model itself.
4. The runner validates the call before anything moves: argument ranges, the condition's menu, the workspace box, and
   a walk of the whole path through the FR3 kinematics (joint limits, joint travel, reach).
5. The primitive runs: the workstation streams target increments at 15 to 20 Hz, the worker checks every limit at
   50 Hz and stops the controller on any violation. A grasp or release is one gripper command while the worker holds
   the arm.
6. The tool result (A: with the phys2 force quantities; B: without force) goes into the next observation.

## 3. Install

### 3.1 Robot PC

Install Polymetis from source, built against the libfranka version your arm's firmware requires. Start the two
servers (and keep them running during episodes):

```bash
launch_robot.py robot_client=franka_hardware robot_client.executable_cfg.robot_ip=<FCI address of the arm>
launch_gripper.py gripper=franka_hand gripper.executable_cfg.robot_ip=<FCI address of the arm>
```

The gripper server homes the hand when it starts; restart it whenever the empty hand does not open to 78 mm. Check the
environment the worker will use:

```bash
<polymetis env>/bin/python -c "import polymetis, torchcontrol, torch, scipy, yaml, grpc; print('ok')"
```

The workstation logs in with a key and never answers a prompt (SSH runs with `BatchMode=yes`). For example, in the
workstation's `~/.ssh/config`:

```
Host robot-pc
    HostName <address of the robot PC>
    User <user on the robot PC>
    IdentityFile ~/.ssh/id_ed25519
```

then `ssh-copy-id robot-pc`; `ssh -o BatchMode=yes robot-pc true` must succeed.

### 3.2 Workstation

```bash
pip install -e ".[real]"                                 # adds pyrealsense2
python -m forceharness.doctor --target real              # pyrealsense2, cameras, site file
python -m forceharness.doctor --target real --probe-robot   # read-only ssh: echo / uname
python -m pytest -q tests/real                           # offline: no robot, camera or key
```

The runner's own interpreter needs `pyrealsense2` (gripper actions save before/after frames in-process);
`cameras.python` in the site file only moves the per-turn photo capture into another interpreter.

## 4. Configure

### 4.1 Site file

```bash
cp configs/real/robot.example.json configs/real/robot.json      # git-ignored; fill in every <...>
```

The runner and the scripts read `configs/real/robot.json`, or the file named by `$FH_ROBOT_CONFIG` or `--robot-config`.
Keys starting with `_` are notes; unknown keys are errors; the example file itself and any remaining `<...>` value are
refused for live use.

| Key | Default | Meaning |
|---|---|---|
| `robot_pc.ssh_host` | (required) | SSH host or alias with key login |
| `robot_pc.conda_prefix` | (required) | root of the Polymetis environment; exported as `CONDA_PREFIX` (Polymetis reads it on import). A leading `~/` is the remote home. |
| `robot_pc.python` | (required) | that environment's Python (3.8 or later) |
| `robot_pc.hardware_config` | (required) | Polymetis `franka_hardware.yaml`; its joint limits are intersected with the worker's |
| `robot_pc.deploy_dir` | `~/forceharness_robot_side` | folder for the robot-side bundles |
| `robot_pc.urdf` | `null` | Panda URDF for the offline self-test only; `null` = the one shipped with Polymetis (or `$FH_URDF` on the robot PC) |
| `robot_pc.robot_port`, `robot_pc.gripper_port` | 50051, 50052 | Polymetis servers on the robot PC |
| `robot_pc.ssh_options` | `[]` | extra ssh arguments, e.g. `["-p", "2222"]` |
| `gripper.force_N` | 60 | grasp force, at most 70 N (the Franka Hand's continuous rating) |
| `gripper.speed_m_s` | 0.05 | finger speed, at most 0.1 m/s |
| `gripper.open_width_m` | 0.08 | opening of a release, 0.078 to 0.08 m |
| `gripper.command_timeout_s` | 20 | one gripper command, 5 to 60 s |
| `cameras.exterior_serial`, `cameras.wrist_serial` | (required) | RealSense serial numbers |
| `cameras.hand_eye_file` | `configs/real/hand_eye.txt` | wrist camera calibration (section 4.3), relative to the repository root |
| `cameras.python` | `null` | interpreter of the capture process; `null` = the runner's own |
| `cameras.capture_timeout_s` | 20 | one capture of both cameras, 5 to 120 s |

Serial numbers of the connected cameras:

```bash
python -c "import pyrealsense2 as rs; [print(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number)) for d in rs.context().query_devices()]"
```

### 4.2 What the guards assume about your cell

The robot-side workspace box for the flange is 0.20 < x < 0.65 m, |y| < 0.30 m, 0.12 < z < 0.55 m in the base frame.
Objects for the tasks sit in front of the robot within that box (the task checklists below give positions). Keep
cables and the camera mount clear of the fingers.

### 4.3 Hand-eye calibration of the wrist camera

The runner needs `T_flange_camera`: the pose of the D405's colour optical frame (x right, y down, z forward) in the
flange frame `panda_link8`, so that `base_T_camera = base_T_flange @ T_flange_camera`. Calibrate it eye-in-hand on
your own mount, for example with OpenCV's `cv2.calibrateHandEye` from 10 to 20 views of a ChArUco board, taking
`base_T_flange` from Polymetis (`RobotInterface.get_ee_pose()` returns `panda_link8` in the base frame). Moving the arm
between views is outside ForceHarness: do it with your usual tools and the emergency stop at hand. Save the matrix as
4 rows of 4 numbers in metres, `#` comments allowed, as `configs/real/hand_eye.txt` (git-ignored);
`configs/real/hand_eye.example.txt` shows the format and is only right for the mount it was measured on. Before the
first task, place an object at a measured position and compare it with the object list of the first observation;
errors above about 5 mm make grasps miss.

## 5. Deploy the robot side (no motion)

```bash
python scripts/real/deploy_robot_side.py                  # print the bundle name and file hashes (no SSH)
python scripts/real/deploy_robot_side.py --deploy         # copy and verify robot_side_<hash> on the robot PC
python scripts/real/deploy_robot_side.py --selftest       # offline self-test of the deployed bundle on the robot PC
python scripts/real/deploy_robot_side.py --compile-check  # read one state, compile the controller, send nothing
```

`--deploy` unpacks `robot_side/franka_polymetis/` into `<deploy_dir>/robot_side_<hash>` and checks it with
`sha256sum`; an existing bundle of the same name is only verified, never overwritten, and every later start re-verifies
it. Redeploy after any change to `robot_side/`. `--selftest` blocks all network access and runs the guarded session on a
fixture (hold, resume, timeouts, stall), the Jacobian-frame check and a comparison of the stock
`HybridJointImpedanceControl` built by the driver with its reference equation; it ends with a JSON line
`{"type": "offline_compile_check", "ok": true, ...}`. `--compile-check` needs the Polymetis servers running; it ends
with `{"type": "compile_check", "ok": true, "hardware_motion": false, ...}`.

## 6. Force gate and calibration (no arm motion)

ForceHarness on this arm reads force as `w = (J^T)^+ tau_ext` at the flange (no wrist sensor). Before any motion,
measure that reading with `scripts/real/check_force.py`. It starts the worker read-only, so Polymetis' default
controller keeps holding the arm, and samples the exact reading the harness uses. Results are printed and written to
`runs/check_force/<time>.json` (`--out` to change), with the robot-side log next to it; the exit code is 0 unless the
stream cannot be read. Pass/warn/fail thresholds and the verdict are in
[the setup skill's force check](../skills/forceharness-setup/references/force-check.md); the script prints the paper's FR3
values for comparison.

Start with the arm still in free space, about 30 cm above the table with the fingers pointing down, the gripper open and
empty, and nothing touching the hand or its cables. Keep every reading well below the 25 N raw cap: at the cap the
worker stops and the check ends.

### 6.1 Still reading: noise, drift, rate, conditioning

```bash
python scripts/real/check_force.py                 # 30 s; --seconds to change
```

Reported under `still`: `force_std_N` (per axis, tared, reaction sign), `force_norm_std_N`, `drift_N` (mean of the last
5 s minus the first 5 s), `robot_rate_hz` and `request_rate_hz`, `residual_Nm`, `sigma_min`, the raw means
`raw_force_mean_N` and `raw_torque_mean_Nm`, the pose `flange_pos_m` and `flange_quat`, and the guard's caps. On the
authors' FR3: 0.02 to 0.05 N standard deviation per axis with the arm still.

### 6.2 Sign

```bash
python scripts/real/check_force.py --sign          # 3 trials; --repeats to change
```

After the countdown, push the fingertips **up from below** with your hand, gently (5 to 10 N), for about 3 s, the way a
table pushes on a hand that presses on it. The reaction force (the force of the environment on the hand) must come out
with positive z in every trial. If all trials come out negative, flip `force_sign` in the calibration file (section 6.4)
and repeat. `--press down` (pressing on the fingertips from above) expects negative z instead. The paper's FR3 reads a
negative raw z for an upward push, hence `force_sign` = -1.

### 6.3 Known mass and the closed-gripper correction

```bash
python scripts/real/check_force.py --mass --known-grams 320.7 --close-gripper
```

An empty reading first; then you hold the object between the fingers (a sealed water bottle held by its cap works
well) and type `CLOSE`: the gripper closes at the runner's force through the same robot-side command as in episodes
(this is the only motion of the gate). You let go, the loaded reading follows, and you type `OPEN` (anything else leaves
the gripper closed). Reported under `mass`: `estimated_g` (from the change of the still reading, as the harness's weight
line), `estimated_corrected_g` (with the calibration's closed-gripper correction), `closure_correction_estimate_N` (the
part of the change the known weight does not explain: the correction for this grasp width), `width_loaded_m`.
Without `--close-gripper` nothing moves: place or hang the mass in or from the open fingers yourself.

Closing the gripper changes the reading by itself (the paper's FR3: about 1.3 N at a 29 mm grasp). So run the closed
check twice with two different known masses grasped at the same width: the first gives the correction (section 6.4),
the second, after the file is updated, must give `estimated_corrected_g` close to its known mass.

### 6.4 The force calibration file

The repository ships `configs/real/force_calibration.example.json` with the values of the paper's arm and with `verified`
and `gripper_verified` set to `false`. Copy it to `configs/real/force_calibration.json` (git-ignored), replace the values
with your measurements, and set the two flags to `true` only after 6.1 to 6.3 pass: the runner refuses a live run unless
`verified` and `gripper_verified` are `true` and `force_sign` is -1 or +1. `check_force.py` uses your file when it exists,
otherwise the example (for its sign and bias).

| Field | Set it to |
|---|---|
| `force_sign` | -1 or +1, so that the sign check of 6.2 passes |
| `bias_force`, `bias_torque` | `still.raw_force_mean_N`, `still.raw_torque_mean_Nm` of 6.1 (the fixed bias subtracted from the raw wrench) |
| `gripper_delta_wrench` | `[cx, cy, cz, 0, 0, 0]` from `mass.closure_correction_estimate_N` of the first closed check of 6.3 (raw sign) |
| `closed_width_m` | `mass.width_loaded_m` of that check (other grasp widths use the same correction as an approximation, and the tool result says so) |
| `mass_kg`, `bottle_id` | the mass and a description of the object used (record only) |
| `noise_std` | the largest value of `still.force_std_N` (N, positive): the noise the estimators assume |
| `bias_std` | the spread of `still.raw_force_mean_N` between repeated still readings (N, positive): the residual bias the estimators assume |
| `flange_quaternion` | `still.flange_quat` of 6.1 |
| `verified`, `gripper_verified` | `true` only after 6.1 to 6.3 pass |
| `torque_verified` | `false` (moments are not certified on this arm) |
| `position_low`, `position_high`, `orientation_radius_rad`, `translation_scope_record_only`, `orientation_scope_record_only`, `width_scope_m` | keep the shipped values: the pose scope is recorded, not enforced |
| `source`, `note`, `scope`, `calibration_status`, `derived_from`, `physical_validation` | free text and flags for the record |

### 6.5 Posture model (optional)

`configs/real/posture_bias_fr3.json` holds an optional posture-dependent zero; the shipped file has `use_model: false`
(on the FR3 a posture model did not beat a constant zero), so nothing from it is applied and each episode relies on
relaxing the controller and its own zero reading. Keep the file: if it is missing, the first live episode runs a
20-minute calibration that **moves the arm** (6 stations x 11 wrist orientations, raw cap 15 N), unless
`--no-auto-calibration` is given. To recalibrate on purpose (after changing the hand, camera mount, pads or cables):
`python scripts/real/posture_calibration.py --live --execute` (moves the arm; `--demo` runs it on the synthetic plant).

## 7. Staged bring-up (the arm moves from here on)

Each stage needs its own go-ahead, a person at the emergency stop and a clear table. The runner's free task with a
person answering the relay drives single primitives with every guard and every validation in place:

```bash
python scripts/real/free_task.py --live --execute --auto-confirm --no-auto-calibration \
    --calibration configs/real/force_calibration.json --condition A --max-actions 20 \
    --instruction "Bring-up: test motions chosen by the operator." \
    --out runs/real/bringup_s2/episode --relay-dir runs/real/bringup_s2/relay
```

When `runs/real/bringup_s2/relay/pending.json` shows `{"turn": N, "waiting": true}`, read `obs_N.md` and write
`cmd_N.json` with one call (or start the free task from the web monitor with its manual model):

| Stage | Call (`cmd_N.json`) | Expect |
|---|---|---|
| S2 small free motions | `{"calls": [{"name": "move_by", "args": {"dx": 0.01, "dy": 0, "dz": 0}}]}`, then back, for each axis | completes, no fault; the tool result's end position within a few mm |
| S3 first contact | from 3 cm above a soft object: `{"calls": [{"name": "move_until_contact", "args": {"direction": [0, 0, -1], "max_dist": 0.05}}]}` | stops at contact with a positive reaction z, well below 25 N |
| S4 force reference | on a kitchen scale: `{"calls": [{"name": "push", "args": {"direction": [0, 0, -1], "target_force": 5, "duration": 3}}]}` | the scale reads about 510 g |
| S4 weight reference | `grasp` a known mass, then `{"calls": [{"name": "lift", "args": {"height": 0.03}}]}` | the weight line of the tool result near the known mass |
| finish | `{"calls": [{"name": "done", "args": {"success": true, "note": "bring-up"}}]}` | the episode ends |

Each relay turn waits at most 600 s; an unanswered turn ends the episode. Then run one task episode (S5) through the
web monitor with the model, starting with `touch_firmness` or `can_weight_sort`.

## 8. Run the tasks

### 8.1 With the web monitor (recommended)

```bash
bash monitor/run.sh --allow-live          # on the workstation, loopback only; see docs/monitor.md
```

On the **Live** page pick the task, the condition and the model, tick the task's checklist and start. The monitor runs
the same command lines as below (they are registered in `monitor/tasks.json`), answers the relay with the chosen model,
and shows the force, the turns and the photos live. "Interrupt" is a soft stop through the runner, not an emergency
stop.

### 8.2 From the command line

```bash
python scripts/real/<task>.py --live --execute --auto-confirm \
    --calibration configs/real/force_calibration.json --condition A [task flags] \
    --out runs/real/<name>/episode --relay-dir runs/real/<name>/relay
python scripts/real/<task>.py score --out runs/real/<name>/episode --outcome success --note "..."
```

Something must answer the relay folder: the monitor (it can take over a relay of an episode started elsewhere) or a
person writing `cmd_N.json`. Without `--live` the script runs the synthetic plant offline (a fixed task then only checks
the interface and ends with `done(success=false)`; `--demo-model` lets the relay model drive the synthetic plant).

| Task | Script | Task flags of the paper's runs | Setup |
|---|---|---|---|
| `blackboard_wipe` | `scripts/real/blackboard_wipe.py` | none | a whiteboard with writing and an eraser on the table; the model picks the eraser up |
| `screw_insert` | `scripts/real/screw_insert.py` | `--max-actions 60 --max-minutes 60` | the screw in its starting hole or already in the hand (`--resume-held "green screw"`); the part with the threaded hole fixed so it cannot turn |
| `peel_vegetable` | `scripts/real/peel_vegetable.py` | `--max-actions 60 --max-minutes 60 --force-max 20 --slide-force-default 10` | the peeler clamped in the hand at about 70 N (`python scripts/real/gripper_check.py close --force 70`, then `--resume-held peeler`); the cucumber or eggplant taped down below the hand; the start posture set by the operator (below) |
| `usb_insert` | `scripts/real/usb_insert.py` | `--force-max 20 --insert-force-max 20` | the USB plug in the hand (`--resume-held "USB plug"`) or on the table; the power strip fixed below the hand, ports visible, cable slack |
| `can_weight_sort` | `scripts/real/can_weight_sort.py` | `--max-actions 80 --max-minutes 60` | three sealed bottles with different amounts of water (heaviest below about 0.4 kg), at least 12 cm apart, a white and a green plate beyond them; shuffle between episodes |
| `touch_firmness` | `scripts/real/touch_firmness.py` | `--max-actions 15` | a soft and a hard object, wrapped alike, side by side at x ≈ 0.48 to 0.60 m |
| `free` | `scripts/real/free_task.py` | `--instruction "..."` | whatever the instruction asks |

The fixed tasks read their instruction from `forceharness/prompts/real/<task>.md`; both conditions get the same file,
and the episode record stores the text. The weighing instruction states the true bottle weights and the readings
measured on the paper's arm; replace both with yours. Changing any of these files changes results.

**Peeling start posture.** The peeling instruction relies on a start posture the operator sets before the episode: on the
paper's setup the flange at about (0.51, 0.02, 0.26) m, the hand tilted about 57° from vertical with its approach axis
pointing down and back toward the robot base, joints about (0.09, 0.45, -0.08, -1.85, 0.08, 1.30, 2.46) rad. This
release has no script for it: move the arm there with your usual Polymetis tools, outside the harness's guards, slowly
and with the emergency stop at hand.

**Flags** (all scripts; `--help` prints them):

| Flag | Meaning |
|---|---|
| `--out`, `--relay-dir` | episode record and relay folder; both must not exist and must not contain each other |
| `--condition A\|B` | A = ForceHarness, B = no-force control |
| `--live --execute` | drive the real robot (both required); default is the synthetic demo |
| `--calibration FILE` | force calibration (required with `--live`) |
| `--robot-config FILE` | site file (default `$FH_ROBOT_CONFIG`, else `configs/real/robot.json`) |
| `--auto-confirm` | grasps and releases without per-action operator approval. Without it, each grasp and release writes `confirmations/<id>.request.json` and prints a `CHAT_CONFIRM_REQUEST` line, and waits for `<id>.reply.json` with `"answer": "CLOSE"` or `"OPEN"` |
| `--max-actions`, `--max-minutes` | episode budget, default 40 and 40, at most 80 and 60 |
| `--force-max` | contact force targets (`move_until_contact`, `push`, `slide`, `search`), 10 to 20 N, default 10 |
| `--insert-force-max` | `insert` force, 12 to 20 N, default 12 |
| `--slide-force-default` | `slide` pressure when the model gives none, up to `--force-max`, default 1 N |
| `--resume-held OBJECT` | the closed gripper already holds OBJECT (left from the previous episode or closed by hand); start holding it |
| `--posture-calibration`, `--no-auto-calibration`, `--recalibrate` | the posture file of 6.5 |
| `--instruction`, `--instruction-file` | the task of the free task (the fixed tasks refuse them) |
| `--demo-model` | synthetic plant driven by the relay model |

A live episode refuses to start if the gripper is not open (at least 78 mm) and no `--resume-held` names what it holds.

**Gripper by hand.** `scripts/real/gripper_check.py width | close [--force N] | open` runs the same robot-side command
outside an episode (never while one runs) and logs to `runs/real_gripper/gripper_log.jsonl`.

### 8.3 What an episode writes

- `<out>/episode.json`: condition, task and version, instruction, menu, limits, every action with its arguments, tool
  result text and state, scene updates, peak forces, the zero reading, fault and outcome; `score` adds the operator's
  outcome.
- `<out>/action_NN.json`: the full primitive report (force trace, events, derived quantities).
- `<out>/nuc/nuc.jsonl` and `nuc.stderr`: the robot-side log (every checked state, target, timing and fault);
  `<out>/nuc/gripper_*/`: before/after state and frames of each gripper action.
- `<out>/photos/NN_<phase>/`: both camera images, the wrist depth (`scene.npz`), the capture log.
- `<relay-dir>/`: `obs_N.md` (+ images), `cmd_N.json`, `pending.json`, `done.json`: only what the model saw and answered.

## 9. Tools the model can call

Condition A: `scene`, `locate`, `move_to`, `move_by`, `rotate_z`, `screw`, `point_tool`, `turn`,
`move_until_contact`, `probe`, `push`, `slide`, `search`, `insert`, `lift`, `pull`, `grasp`, `release`, `hold`, `done`.
Condition B: `scene`, `locate`, `move_to`, `move_by`, `rotate_z`, `point_tool`, `turn`, `grasp`, `release`, `hold`,
`done`. Tasks add or remove tools (`forceharness/real/tasks.py`): `screw_insert` removes `lift` and adds `wiggle` to A;
`usb_insert` adds `wiggle` to A and uses `search` without its hop mode; `peel_vegetable` adds `move_until_contact` to B
and uses its own system prompts. The descriptions the model reads, with their ranges, are in `forceharness/real/menu.py`.

## 10. Safety limits

Robot side (`robot_side/franka_polymetis/guard.py`, can only be tightened; details in [robot_protocol.md](robot_protocol.md)):

| Limit | Value |
|---|---|
| target speed | 8 mm/s, 0.15 rad/s |
| target lead over the measured flange | 8 mm; up to 25 mm only once the raw force rose by 3 N (pressing) |
| raw force / moment cap | 25 N / 3 N·m: immediate stop |
| tracking error | 15 mm (plus any granted extra lead) |
| per primitive | 0.15 m and 0.60 rad measured travel, 120 mm target envelope, 0.9 rad joint travel, 300 s |
| flange box | 0.20 < x < 0.65 m, \|y\| < 0.30 m, 0.12 < z < 0.55 m |
| joints | 0.03 rad inside FR3 ∩ Panda ∩ hardware-configuration limits |
| state | fresh within 0.15 s; command lease 0.30 s; stall 4 s; Jacobian σ_min >= 0.01; residual <= 3 N·m and rise <= 1 N·m |
| hold | 5 mm deviation, 8 N force change for 3 ticks, 60 s |
| gripper | width 0 to 85 mm and unchanged during an arm primitive (off for a clamped tool) |

Workstation side (`forceharness/real/menu.py`, `validation.py`, `runner.py`): at most 0.10 m per move, 30° per rotation
segment, 180° per `rotate_z` or `screw` call (run as 30° segments), wrist joint 7 within -0.8 to 2.4 rad for turning;
contact force targets 10 N by default (a task may raise them to 20 N = 0.8 x the raw cap), `push` up to 15 N (or a
task's higher contact limit), `insert` 12 N, `pull` 20 N; collision stop at 8 N (12 N while holding); one contact search at most 0.30 m; every path walked
through the kinematics before it starts. The model cannot change any of these.

## 11. Changes for the public release (not yet validated on hardware)

The paper's real-robot runs used this code with the following differences. Behaviour on the arm is meant to be the same,
but these parts have not been re-run on hardware yet:

- **Gripper path.** Grasp and release run `robot_side/franka_polymetis/gripper.py` (Polymetis `GripperInterface`) once
  per action over SSH, from the deployed bundle. The paper's runs used a site-specific gripper service with the same
  semantics (60 N grasp, 80 mm open, at least 3 s between commands, at least 3.2 s of state reads per action, the same
  width checks, no retries). The separate hardware health monitor of those runs is replaced by reading the worker's
  checked state every 50 ms during the action. `scripts/real/gripper_check.py` replaces the operator's gripper tool.
- **Site configuration.** Host, environment, ports, deploy folder, camera serials and the hand-eye file come from
  `configs/real/robot.json`; the worker's lock lives in the deploy folder; the bundle is named `robot_side_<hash>`.
- **Cameras.** The per-turn capture runs as `python -m forceharness.real.cameras` with a configurable interpreter.
- **Model-facing text.** System prompts, task instructions, tool descriptions and tool-result text that were not in
  English in the paper's runs have been translated; text that was already English is unchanged, except that internal
  version tags were removed from a few refusal messages and prompt headers. A task instruction may now be up to
  16000 characters.
- **Peeling start posture.** The helper that moved the arm to it is not included (section 8.2).

## 12. Troubleshooting

| Message | Cause and fix |
|---|---|
| `The gripper is not open at the start ...` | The hand is closed or cannot open to 78 mm. If it is empty: clear the fingers and restart the Polymetis gripper server (it homes). If it holds something: `--resume-held "<object>"`. |
| `Unexpected worker handshake` or `NUC fault: NUC worker EOF` at the start | The bundle is not deployed (`deploy_robot_side.py --deploy`), the checksum failed (never edit files in the deploy folder; redeploy), the environment cannot import Polymetis, or another worker holds the lock. See `<out>/nuc/nuc.stderr`. |
| `NUC guard limits do not match requested tighter configuration` | The worker reports looser limits than the workstation requested (a modified or ported worker that ignores `--limits-json`). |
| `NUC fault: ... Raw force/torque cap exceeded` | 25 N or 3 N·m reached; the controller stopped at once. Check what the hand touched. |
| `NUC fault: ... Start requires stationary arm` | A joint moved faster than 0.03 rad/s at `begin`; wait for the arm to settle. |
| `Live needs a verified force calibration with the gripper closure correction` | `verified`, `gripper_verified` or `force_sign` missing from the calibration file (section 6.4). |
| `Hand-eye file not found` | `cameras.hand_eye_file` points nowhere (section 4.3). |
| `Camera capture failed ...` | See `<out>/photos/NN_*/capture.log`: wrong serial, a camera on USB 2, or another process holding the camera. |
| `NUC reply timeout; outcome unknown` | Network or robot PC stall; the session ended. Check the arm before the next episode. |
