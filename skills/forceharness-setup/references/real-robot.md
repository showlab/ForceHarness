# Real-robot bring-up

The reference system is a Franka FR3 with a Franka Hand under Polymetis, a wrist RealSense D405 and an exterior D435.
Full documentation: [docs/real_robot.md](../../../docs/real_robot.md) and the protocol in
[docs/robot_protocol.md](../../../docs/robot_protocol.md). Other arms: [porting.md](porting.md) first, then the same stages.

Two computers are involved:
- the **robot PC** (real-time kernel, next to the arm) runs the vendor stack (Polymetis arm and gripper servers) and our guard
  process from `robot_side/franka_polymetis/`;
- the **workstation** runs the harness (`forceharness/real/`), the cameras, the web monitor and the model calls, and reaches the
  robot PC over key-based ssh.

## C1 Identify (no motion)

Ask the human, then confirm what you can by reading: arm model and firmware, controller software and version, gripper,
cameras (`python -m forceharness.doctor --target real` lists RealSense devices), the ssh alias of the robot PC
(`python -m forceharness.doctor --target real --probe-robot` runs only `echo` / `uname` there).

## C5a Configure the site file (reference stack, no motion)

`cp configs/real/robot.example.json configs/real/robot.json` (git-ignored) and replace every `<...>` value:
`robot_pc.ssh_host` (an alias in `~/.ssh/config` with key-based login), `robot_pc.conda_prefix` and `robot_pc.python`
(the Polymetis environment, Python 3.8), `robot_pc.hardware_config`, `robot_pc.deploy_dir`, `robot_pc.robot_port` / `gripper_port`
(50051 / 50052), `cameras.exterior_serial` / `wrist_serial` (from `python -m forceharness.doctor --target real`),
`cameras.hand_eye_file` (4×4 `T_flange_camera`, format in `configs/real/hand_eye.example.txt`; calibrate your own mount).
Check: `python -m forceharness.doctor --target real --probe-robot` shows no unfilled placeholders and a working ssh.
Safety limits live in code, not in this file (see SKILL.md C4).

## C5b Deploy the robot side (no motion)

Polymetis' arm and gripper servers must already run on the robot PC (they are not part of this repository).

```bash
python scripts/real/deploy_robot_side.py                  # print the bundle name and file hashes (no SSH)
python scripts/real/deploy_robot_side.py --deploy         # copy and verify robot_side_<hash> on the robot PC
python scripts/real/deploy_robot_side.py --selftest       # offline self-test of the bundle on the robot PC
python scripts/real/deploy_robot_side.py --compile-check  # read one state, build the controller, send nothing
```

Pass: `--selftest` ends with `{"type": "offline_compile_check", "ok": true, ...}` and `--compile-check` with
`{"type": "compile_check", "ok": true, "hardware_motion": false, ...}`. Redeploy after any change to `robot_side/`.

## C6 Force gate, measured (no motion except an approved gripper close)

With the site file (C5a) and the robot side deployed (C5b), follow [force-check.md](force-check.md):

```bash
python scripts/real/check_force.py                                   # M1, M2: 30 s still reading
python scripts/real/check_force.py --sign                            # M3: push the fingertips up from below
python scripts/real/check_force.py --mass --known-grams 320.7        # M4: you place the mass, nothing moves
python scripts/real/check_force.py --mass --known-grams 320.7 --close-gripper   # M4 with a gripper close (go-ahead first)
```

Then copy `configs/real/force_calibration.example.json` (the paper's FR3 values, marked unverified) to
`configs/real/force_calibration.json` and replace it field by field with your measurements as in docs/real_robot.md
section 6.4 (`force_sign`, `bias_force`, `bias_torque`, `gripper_delta_wrench`, `closed_width_m`, `noise_std`, `bias_std`, ...).
The runner refuses live runs until `verified` and `gripper_verified` are `true`; set them only after M1 to M4 pass, and
never use another arm's values.

## C7 Staged bring-up

Ask for a go-ahead before **each** stage and repeat the safety question (person at the emergency stop, workspace clear).

| Stage | What | Pass criterion |
|---|---|---|
| S0 | read-only stream (C6) | rate >= 20 Hz, M1/M2 of force-check.md |
| S1 | force calibration (C6): sign (push up from below), zero, closed-gripper correction, known masses | M3, M4 PASS; values written to `configs/real/force_calibration.json` |
| S2 | tiny free motions: 1 cm along each axis at 8 mm/s, then back | tracking error < 2 mm at rest, no guard fault, M5/M6 |
| S3 | first contact: `move_until_contact` straight down onto a soft object or the table from 3 cm above | stops on contact with reaction Fz > 0 and below the 25 N cap |
| S4 | references: `push` 5 N on a kitchen scale (scale reads about 510 g); `lift` a known mass | push within ±1.5 N of the scale; mass within the M4 tolerance |
| S5 | one task episode with the model through the web monitor, starting with `can_weight_sort` or `touch_firmness` | the episode ends with `done`, no fault; the human watched it |

Stages S2 to S4 drive single primitives through the runner's free task, with every guard and validation in place,
while a person (you, relaying the human's choice) answers each turn instead of a model:

```bash
python scripts/real/free_task.py --live --execute --auto-confirm --no-auto-calibration \
    --calibration configs/real/force_calibration.json --condition A --max-actions 20 \
    --instruction "Bring-up: test motions chosen by the operator." \
    --out runs/real/bringup/episode --relay-dir runs/real/bringup/relay
```

When `runs/real/bringup/relay/pending.json` shows `{"turn": N, "waiting": true}`, read `obs_N.md` and write `cmd_N.json`
with one call, for example `{"calls": [{"name": "move_by", "args": {"dx": 0.01, "dy": 0, "dz": 0}}]}` (S2),
`{"calls": [{"name": "move_until_contact", "args": {"direction": [0, 0, -1], "max_dist": 0.05}}]}` (S3, from 3 cm above a
soft object), `{"calls": [{"name": "push", "args": {"direction": [0, 0, -1], "target_force": 5, "duration": 3}}]}` on a
kitchen scale (S4), and finally `{"calls": [{"name": "done", "args": {"success": true, "note": "bring-up"}}]}`.
Each turn waits at most 600 s. The full table is in docs/real_robot.md section 7. Keep
`configs/real/posture_bias_fr3.json` in place: if it is missing, a live run without `--no-auto-calibration` first runs a
20-minute calibration that moves the arm. S5 runs one task from the web monitor
(`python -m fhmon` from `monitor/`, see docs/monitor.md) with the model, starting with `touch_firmness` or `can_weight_sort`.
