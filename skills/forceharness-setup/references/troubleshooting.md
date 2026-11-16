# Troubleshooting

## Install and simulation

| Symptom | Cause | Fix |
|---|---|---|
| `AssertionError` inside robosuite `get_joint_qpos_addr` | MuJoCo newer than 3.2.7 | `pip install mujoco==3.2.7` |
| `Cannot initialize a EGL device display` / black images | wrong `MUJOCO_EGL_DEVICE_ID`, or no EGL driver | try another device index (EGL order can differ from `nvidia-smi`); on Linux without a GPU `MUJOCO_GL=osmesa`; on macOS the default is `cgl`; or run `check_sim.py --no-render` |
| `ImportError` / `OSError` naming `libEGL.so.1` or `libGL.so.1` | system graphics libraries missing on a bare Linux machine | `sudo apt-get install -y libegl1 libgl1` (and `libosmesa6` for `MUJOCO_GL=osmesa`) |
| `check_sim.py` sign check FAIL | `force_sign` wrong for the sensor convention, or the hand never reached the table | keep `force_sign = -1` for robosuite's Panda; check `contact_z` in the summary |
| A batch reruns nothing after errors | `run_matrix.py` skips every `episode_id` already in `results.jsonl`, including failed ones | remove the lines with an `error` (or `agent_error`) from `results.jsonl`, then rerun |
| Many episodes end with `agent_error` and 0 model calls | the API key ran out of credit or hit a rate limit | top up / wait, remove those lines, rerun |
| `gpt-6-astra` returns 404 / unsupported | it is only served through the Responses API | `provider: openai-responses` |
| A new tool behaves oddly with OpenAI models | strict function schemas make the model fill every field, unused ones with 0 | make 0 harmless for every optional field; run one real episode before a batch |
| Claude episodes cost much more than expected | extended thinking makes more calls per episode, especially without force | estimate cost from that model's own call counts |

## ForceBench

| Symptom | Cause | Fix |
|---|---|---|
| Isaac refuses to start on driver 535 | Isaac Sim 6.0.1 wants >= 550.90.07 | upgrade the driver, or opt in to the documented driver-check workaround (RT-core GPUs only) |
| Run "succeeds" but no result | Isaac's exit code is always 0 | read the evaluator's `FORCEBENCH_VECTOR_RESULT=` line |
| Start hangs with no CPU use | Kit stuck on a lock | the run script's watchdog restarts it after 10 min; rerun the job |
| Later environments look darker or blurred | neighbouring scenes too close | keep the 30 m environment spacing |
| White images | GPU memory exhausted by too many Isaac processes | fewer processes per GPU; the episode aborts on an empty image |
| H100 / H200 / A100 cannot render | no RT cores | use an RTX-class GPU |

## Real robot

| Symptom | Cause | Fix |
|---|---|---|
| ssh asks for a password or hangs | key-based login not set up | set up keys for the robot-PC alias in `~/.ssh/config`; the harness uses BatchMode and never types passwords |
| Guard fault "lease" | no workstation step for 300 ms outside a hold (network, CPU load, a paused process) | keep the workstation idle otherwise, use a wired link, check the step round-trip time in the runner log |
| Guard fault on the wrench residual | joint torques that no flange wrench explains, typical at large wrist angles or with an unconfigured payload | keep joint 7 near its middle, configure only the gripper's mass in the vendor tool |
| Contact detected during free motion | joint friction while moving shows up as force (2 to 5 N on the FR3) | slower motion; the settle rule before readings; never lower the collision stop below the measured motion bias |
| The run refuses to start because the held state is unknown | the gripper may be holding something from a previous run | open the gripper with the human's go-ahead, or confirm it is empty |
| A force-cap stop ends the episode | the raw force passed 25 N | inspect the episode in the monitor; do not raise the cap |
| Weights read wrong after a grasp | reading taken while the target still leads the arm | the runner settles before reading; if you wrote your own script, relax the target onto the measured pose first |
| Ports 50051 / 50052 not reachable | Polymetis arm / gripper servers not running on the robot PC | start them as in docs/real_robot.md; they are not part of this repository |
