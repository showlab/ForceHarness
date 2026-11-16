# Where an arm's force comes from

ForceHarness needs the force that the environment applies to the hand, as a 3-D vector (torque helps but is optional),
at 20 Hz or faster, quiet enough to detect a 5 N contact reliably and, for weighing, to resolve changes of about 0.2 N
(20 g) between still readings. This table says which channel to **expect**. It is a starting point, not a verdict:
firmware, options and SDK access differ between units, so confirm with the vendor documentation and always measure with
[force-check.md](force-check.md).

| Channel | Typical quality for ForceHarness | Expected verdict |
|---|---|---|
| Wrist 6-axis F/T sensor (built in, or an add-on such as ATI, Bota Systems, OnRobot HEX, Robotiq FT 300-S, Schunk FTS) | Low noise, small drift; reads the hand directly | FORCE_OK |
| Joint torque sensors in every joint + a model-based external wrench, or `(J^T)^+ tau_ext` solved by ForceHarness | Low noise but a bias that changes with pose, motion direction, joint friction and wrist angle (on the reference FR3: 0.03 N static noise, 1.5 to 5 N bias after motions, up to 12 N at large wrist angles). The tare / settle / report-changes rules exist for this channel | FORCE_OK, measure the bias |
| Motor-current estimate of an industrial arm (no torque sensors, no wrist sensor) | Noisy, friction and gear effects of several newtons; contact detection possible, weighing and gentle placing usually not | FORCE_DEGRADED (measure before porting) |
| Coarse load / current readings of hobby or smart servos (Feetech, Dynamixel XL-class), or position feedback only | Not convertible into a reliable force at the hand | NO_FORCE |

## Common arms (typical configuration; verify yours)

| Arm | Force channel usually available | Notes for ForceHarness | Expected |
|---|---|---|---|
| Franka Research 3, Franka Emika Panda | Joint torque sensors; libfranka estimates external torques (`tau_ext_hat_filtered`) and an end-effector wrench | **Reference platform.** Polymetis forwards `tau_ext` but not the vendor wrench, so `robot_side/franka_polymetis` solves `(J^T)^+ tau_ext` itself. Calibrate only the offset; use the tare and settle rules | FORCE_OK |
| KUKA LBR iiwa / LBR Med | Joint torque sensors; external torque and estimated wrench through FRI / Sunrise | Compliant joint and Cartesian impedance modes exist | FORCE_OK |
| Kinova Gen3 (7 DoF) | Torque sensors in each actuator; estimated tool wrench in the Kortex feedback | Expect more noise than a wrist sensor; measure | FORCE_OK or DEGRADED |
| Flexiv Rizon | Joint torque sensors and a wrench estimate through the vendor SDK | Built for force control | FORCE_OK |
| Doosan M-series | Torque sensors in all joints | Check SDK access to the wrench at >= 20 Hz | FORCE_OK |
| Universal Robots e-Series (UR3e, UR5e, UR10e, UR16e), UR20, UR30 | Built-in 6-axis F/T sensor at the tool flange (`getActualTCPForce()` in ur_rtde) | The datasheets give a force precision of a few newtons, so contact and insertion are plausible but weighing small loads may fail M4; the position controller is stiff, so the lead must be produced by an admittance loop or the vendor force mode (see porting.md) | OK or DEGRADED: measure (control needs work) |
| Universal Robots CB-series (UR3, UR5, UR10) | Force estimated from motor currents | Add a wrist F/T sensor | FORCE_DEGRADED |
| UFactory xArm 5/6/7, Lite 6 | Optional 6-axis F/T sensor accessory; otherwise joint-current estimates | With the sensor: FORCE_OK; without: DEGRADED | depends |
| Rethink Sawyer | Series-elastic joints with torque sensing; endpoint wrench estimate | Measure; old software stack | DEGRADED or OK |
| AgileX Piper and similar low-cost 6-DoF arms | Motor current / effort only | No torque or wrist sensor and a stiff position controller; contact detection at best | NO_FORCE (stop at C2) unless a wrist F/T sensor is added |
| SO-100 / SO-101, Koch, other hobby-servo arms; WidowX / ViperX | Servo load or current readings, coarse | Not enough for force primitives; add a wrist sensor | NO_FORCE |
| Industrial arms (ABB, FANUC, Yaskawa, Kawasaki, Staubli, Denso) | Usually an optional wrist F/T sensor with the vendor's force-control package | Without the option there is no force | depends |

If the arm is not listed: find out whether it has a wrist sensor, joint torque sensors, or only motor currents, and whether
the SDK exposes them at >= 20 Hz to an external computer. Then measure.

Force is only half of it: gate C3 also needs a compliant controller that accepts streamed targets. Franka, KUKA LBR and
Flexiv offer impedance control to an external computer; Kinova Gen3 is compliant only in its low-level torque mode (the
high-level API is stiff); UR e-Series, xArm and most industrial arms are stiff position-controlled and need the admittance
stage or vendor force mode of [porting.md](porting.md). Verify this for your arm and controller version.
