# Third-party notices

ForceHarness is released under the MIT license (see `LICENSE`). It depends on, derives from, or interoperates with the
following projects, which keep their own licenses.

## Code included in this repository

| File | Derived from | License |
|---|---|---|
| `scripts/forcebench/eval_forcebench.py` | ForceBench's vectorized inference/evaluation script | MIT (ForceBench); see the header of the file |
| `forceharness/backends/forcebench/wire.py` | a re-implementation of ForceBench's websocket message format | written for this project; format by ForceBench |
| `scripts/baselines/forcevla/pi0_force.py` | ForceVLA's model code as bundled in ForceBench's baselines | Apache-2.0; see the header of the file |
| `scripts/baselines/forcevla/limoe_simple.py` | ForceVLA runtime module (Google LLC header kept) | Apache-2.0; see the header of the file |
| `forceharness/baselines/diffusion_policy.py` | a re-implementation of Diffusion Policy (Chi et al., 2023), CNN variant | written for this project |

## Dependencies (installed separately, not redistributed)

| Project | Used for | License |
|---|---|---|
| robosuite 1.5.1 | HiddenPhys environments (`forceharness/backends/mujoco/`) | MIT |
| MuJoCo 3.2.7 | physics for HiddenPhys | Apache-2.0 |
| ForceBench | the ForceBench tasks and evaluator | see the ForceBench repository |
| ForceBench assets (including third-party kitchen scenes) | downloaded by ForceBench's own scripts | their own licenses, some CC BY-NC 4.0 |
| NVIDIA Isaac Sim 6.0.1 / Isaac Lab 3.0 | ForceBench simulator | NVIDIA Omniverse license (you must accept it yourself) / BSD-3-Clause |
| Polymetis | Franka FR3 control on the robot PC | MIT |
| openpi | π0.5 and ForceVLA baselines | Apache-2.0 |
| LeRobot | dataset format for the learned baselines | Apache-2.0 |
| OpenAI Python SDK | model access | Apache-2.0 |
| Anthropic Python SDK | model access | MIT |
