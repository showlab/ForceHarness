# Web monitor

`monitor/` is a web console for the real-robot experiments: start an episode, watch it live, review it afterwards and
compare the force-harness condition with the no-force baseline. It shows, turn by turn, what the model saw, what it decided,
which primitive ran and what came back, next to the wrist force with its guard lines and a video of the scene. It uses only
the Python standard library and plain JavaScript, never imports `forceharness`, and never commands the robot itself: it
starts the runner scripts in `scripts/real/` and reads the files they write. The full reference is
[monitor/README.md](../monitor/README.md).

## Review recorded runs

No robot, model or extra package is needed:

```bash
bash monitor/run.sh --runs /path/to/runs        # or: cd monitor && python3 -m fhmon --runs /path/to/runs
```

Open <http://localhost:8765/>. **Records** lists every episode under the runs folder; click one to see its turns (input →
decision → primitive call → tool result), the force and torque curves with action bands and contact events, and for each
action the measured force, the quantities the harness computed and the text the model received. **Compare** gives the success
rate of each condition with a 95 % Wilson interval and puts two episodes side by side. Episodes can be labelled by hand
(success flag and note), which is stored by the monitor and never changes the run folder.

## Run episodes

1. Demo first: on the **Live** page pick a task, a condition and **Demo (no robot)**, and a model (scripted demo, mock,
   manual or a configured API model). The runner drives a synthetic robot, so the whole loop can be checked offline.
2. Models: on **Settings**, create a profile for OpenAI-compatible chat, OpenAI Responses or Anthropic. Keys stay in the
   monitor's memory or come from environment variables such as `FH_API_KEY` and `ANTHROPIC_API_KEY` in `.env` (loaded by
   `monitor/run.sh`). A ChatGPT plan can be used through the Codex CLI (`codex` on PATH, or `--codex-bin`). The monitor answers
   the runner's relay files (`obs_N.md` → `cmd_N.json`) for the model; "External relay" leaves that to another program or a
   person.
3. Real robot: start with `bash monitor/run.sh --allow-live` on the computer next to the robot (loopback only). Put the force
   calibration in `configs/real/force_calibration.json`, choose the runner interpreter with `--python` or `FH_PYTHON` if it is
   not the one running the monitor, tick every item of the task's checklist and keep a person at the emergency stop. Only one
   real-robot episode runs at a time; all limits are the runner's own and stay in force. "Interrupt" is a soft stop through
   the runner, not an emergency stop.

The six fixed tasks of the paper (`blackboard_wipe`, `screw_insert`, `peel_vegetable`, `usb_insert`, `can_weight_sort`,
`touch_firmness`) read their goal from `forceharness/prompts/real/<task>.md`, which the Live page can edit; the free task takes
the task text you type. Command lines, checklists and score options are in `monitor/tasks.json`.

## Session videos

To record every real-robot episode with a global RealSense camera, copy `monitor/video.example.json` to `monitor/video.json`,
set `enabled`, the camera serial, the serials of the cameras the harness itself uses (`never_open`, never opened by the
recorder) and an interpreter with `pyrealsense2`; `ffmpeg` must be on PATH. Videos appear on the episode page and the **Videos**
page.

## Check

```bash
python3 -m pytest -q -p no:cacheprovider monitor/tests       # offline
```
