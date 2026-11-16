# Robot side: Franka FR3 under Polymetis

The code that runs on the robot PC (the real-time computer next to the arm), in Polymetis' own Python environment
(3.8 or later; numpy, scipy, pyyaml, torch, torchcontrol, polymetis, grpcio). Nothing is installed: the workstation
copies these files as a checksummed bundle (`scripts/real/deploy_robot_side.py --deploy`) and starts them over SSH.
The protocol is specified in [docs/robot_protocol.md](../../docs/robot_protocol.md); setup and use in
[docs/real_robot.md](../../docs/real_robot.md).

| File | Role |
|---|---|
| `worker.py` | One guarded session per process: JSON lines on stdin/stdout, a 50 Hz tick, read-only unless `--enable-motion`. Started by the workstation once per episode. |
| `guard.py` | Every hard limit (`Limits`, can only be tightened) and the session logic: checks on each state, rate and lead limits on each target, release ramp, hold, faults. Plain numpy/scipy, no hardware imports, so the workstation's tests run it too. |
| `driver.py` | Polymetis: robot state, the wrench, IK and the target updates of the stock `HybridJointImpedanceControl`, controller start and stop with ownership checks. |
| `wrench.py` | `w = (J^T)^+ tau_ext` at the flange with its residual and Jacobian conditioning; also a standalone read-only reader. |
| `gripper.py` | One Franka Hand action (`state`, `close`, `open`) through Polymetis' `GripperInterface`, one JSON line out. Changed for the public release; not yet validated on hardware. |
| `selftest.py` | Offline self-test with the network blocked: guard lifecycle, Jacobian frame, the stock controller against its reference equation. |

Run by hand on the robot PC only for debugging, from the deployed bundle folder, with the Polymetis environment
(`CONDA_PREFIX=<env> <env>/bin/python ...`):

```bash
python selftest.py [--urdf <panda_arm.urdf>]          # offline, no robot RPC
python wrench.py --hz 50                              # read-only stream of the wrench estimate (JSON lines)
python worker.py --hardware-config <franka_hardware.yaml>                  # read-only worker; send {"id":1,"op":"state"}
python gripper.py state                               # read the gripper; close / open move the fingers
```

`worker.py --enable-motion` and `gripper.py close|open` move hardware: only with a person at the emergency stop.
Never edit files inside a deployed bundle (the checksum check then refuses to start); change them here and redeploy.
