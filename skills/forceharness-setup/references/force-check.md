# Force gate: how to measure an arm's force channel

Run these in order. M1 to M4 do not move the arm (M4 may close the gripper, which needs a go-ahead). M5 and M6 need the
motion stages of [real-robot.md](real-robot.md). Record every number in `runs/setup_report.md`.

On the reference Franka FR3 these measurements are made with `scripts/real/check_force.py`, which starts the robot-side
worker read-only (Polymetis' default controller keeps holding the arm) and samples the exact reading the harness uses:
`python scripts/real/check_force.py` (M1, M2; `--seconds` to change), `--sign` (M3), `--mass --known-grams G` (M4, add
`--close-gripper` to let it close the gripper after you type CLOSE). Results go to `runs/check_force/<time>.json`; field
names are explained in [docs/real_robot.md](../../../docs/real_robot.md) section 6.
On a ported arm, read `Backend.wrench()` and `Backend.reaction_force()` of your backend in a loop with the arm still and
compute the same statistics.

**Measuring before a port.** To decide whether a port is worth it, write a short read-only script on the vendor SDK: read
the force channel the arm offers (the flange wrench, or the external joint torques and the Jacobian to solve
`w = (J^T)^+ tau_ext`, or the motor currents) for 30 s with the arm still and its normal controller holding it, while the human
does the sign push of M3. Do not command any motion in this script. Express everything as a force at the hand in newtons
before applying the thresholds; a channel that cannot be converted to newtons at the hand is NO_FORCE for ForceHarness.
For current-based arms read the SDK's gravity-compensated external joint torques or estimated wrench: raw motor currents
include gravity and friction and do not convert to newtons at the hand. Starting points, to verify against the vendor
documentation: ur_rtde `getActualTCPForce()`, the Kortex BaseCyclic feedback `tool_external_wrench_force_x/y/z`, and the
xArm SDK `ft_ext_force` (with the F/T sensor option) or `joints_torque`.

## Why these thresholds

The harness counts contact when the tared force stays above a threshold (5 N by default on the real robot, 3 N in
simulation) for 3 ticks; motions stop as collisions at 8 N (12 N when holding); the raw force cap is 25 N; the weighing tasks
need to tell apart loads that differ by about 1.5 N (150 g against 320 g bottles), and tool results report mass changes from
20 g. The thresholds below keep false contacts, false collision stops and wrong weights rare.

| Test | How | PASS | WARN | FAIL | Reference FR3 (joint torques) |
|---|---|---|---|---|---|
| M1 static noise | Arm still, empty hand, 30 s or more: standard deviation of each tared force axis and of \|F\| | <= 0.3 N | <= 1.0 N | > 1.0 N | 0.02 to 0.05 N |
| M2 drift | Same recording: tared mean of the last 5 s minus the first 5 s | <= 0.5 N | <= 2 N | > 2 N | well below 0.5 N |
| M3 sign | A person pushes the fingertips **up from below** 3 times (countdown, ~3 s, 5 to 10 N), the way a table pushes back on a hand that presses on it. After `force_sign`, the reaction z must be positive every time (pressing down from above must give negative z) | 3/3 positive and >= 3 N | | any wrong sign: flip `force_sign` and repeat | an upward push reads negative raw Fz, so force_sign = -1 |
| M4 known mass | Empty still reading, then a known mass in or hanging from the closed gripper (e.g. sealed water bottles of about 150 g and 320 g), still reading again; mass = change / 9.81 | error <= 30 g | <= 80 g | > 80 g | two bottles of 148 g and 321 g were read correctly after one common offset |
| M5 settle repeatability (after S2) | Move 1 to 2 cm and back several times; after each move let the target settle onto the measured pose (the harness relaxes for ~0.8 s and averages 0.5 s); median \|difference\| between consecutive still readings | <= 0.3 N | <= 1.5 N | > 1.5 N | median 0.08 N over 999 windows (1.07 N without the settle step) |
| M6 motion bias (after S2) | Tared reading during free motion at 8 mm/s | <= 3 N | <= 6 N | > 6 N (false collision stops at 8 N) | 1.8 to 5 N |

## Verdict

- All PASS: **FORCE_OK**.
- Any WARN, no FAIL: **FORCE_DEGRADED**. Tell the human which behaviours are at risk:
  M1/M2 WARN: contact detection thresholds may need raising (only upward, never lower the safety stops);
  M4 WARN: weighing and centre-of-mass judgements unreliable;
  M5/M6 WARN: false collision stops during motion and noisy quantities in tool results.
  Continue only with the human's agreement.
- Any FAIL (or the stream cannot be read at >= 20 Hz): **NO_FORCE / STOP**. Explain and offer: a wrist F/T sensor, another arm,
  or the simulators.

## Notes

- A joint-torque estimate is biased by pose, motion direction, joint friction and large wrist angles. That is expected:
  ForceHarness calibrates only the offset (tare at rest before each empty-hand primitive, settle then read, report changes
  relative to the empty hand or the moment of the grasp). Do not try to fit a gain unless you have an external force reference.
- The vendor's payload settings matter: a configured end-effector load is subtracted inside the estimate on some arms.
  Keep the gripper's own mass configured and nothing else.
- Never raise the 25 N raw force cap to make a test pass.
