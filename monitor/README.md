# ForceHarness monitor

A web console to **start, watch, review and compare** real-robot episodes of ForceHarness. It shows the wrist force live,
next to what the model saw, what it decided, which primitive ran and what came back, and it puts the force-harness
condition and the no-force baseline side by side.

It is a small standalone program: Python standard library only on the server, plain JavaScript modules in the browser
(no build step), no imports from the `forceharness` package. It reads the files the harness already writes, starts the
harness's own runner scripts, and can play the model's side of the relay protocol. It never moves the robot itself.

```
browser ──HTTP / SSE──► monitor/fhmon
                           │ reads   episode.json, action_NN.json, nuc/nuc.jsonl, relay obs_N.md / cmd_N.json
                           │ writes  its own session folders, labels and settings, and cmd_N.json (the model's reply)
                           ├─ starts the runner scripts registered in monitor/tasks.json (scripts/real/*.py)
                           └─ starts fhmon/camrec.py to record a session video (optional, one camera)
```

## What it shows

| Page | What it does |
|---|---|
| **Live** | Start an episode: task, condition (A · Force harness, B · Baseline, or both one after the other in demo mode), demo or real robot, model, the fixed task text, an episode name. The exact runner command is shown before you start. On the right: episodes running now (including ones started from a terminal) and the ones started from this page. |
| **Episode** (click any episode) | A strip of the primitive calls in order (`00 move_to → 01 grasp → 02 slide ...`), the step being executed blinking. Each turn is one loop with four stages: **① Input** (the images and the full observation the model got), **② Decision** (its reason, raw reply and thinking, streamed live), **③ Primitive call** (name and every argument), **④ Return** (termination, duration, peak force and the tool result the model reads next turn; the force harness's translation is highlighted, and a baseline turn shows what the model did not get). On the right: the whole-episode wrist force and torque with the guard lines, action bands and contact events; the session video; for the selected action, "measured force → physical quantities the harness computed → what the model was told"; the camera images before and after; the episode settings and the runner command. Ended episodes can be scored with the runner's own `score` command, or labelled by hand. |
| **Records** | Every episode found under the runs folder, with search and filters (task, condition, mode, result). |
| **Videos** | All session videos, newest first, playable in the page, each linked to its episode. |
| **Compare** | Per task, real robot and/or demo: the success rate of each condition with its 95 % Wilson interval, valid episodes, mean actions, duration, median peak force, how often the model claimed success; then any pair of episodes turn by turn, side by side. Below 20 valid episodes per condition the page says the numbers are a pipeline check, not a result. For the free task, pick one task text first: success rates over different instructions are not comparable. |
| **Settings** | Model profiles (see [Models](#models)), with a connection test, and the system prompt the relay sends to API models. |

Adding `?blind` to the address (`http://localhost:8765/?blind#/episodes`) hides calendar dates, for screenshots.

## Run it

```bash
bash monitor/run.sh                       # review recorded runs, run demo episodes (no robot)
bash monitor/run.sh --runs /data/fh_runs  # index another runs folder
bash monitor/run.sh --allow-live          # also allow real-robot episodes from the page
```

Then open <http://localhost:8765/>. `run.sh` runs `python -m fhmon` from `monitor/` with `$FH_PYTHON` (else `python3`)
and, when `<repository>/.env` exists, loads its variables (API keys) into the monitor process only. Every option of
`python -m fhmon --help`:

| Option | Default | Meaning |
|---|---|---|
| `--harness-root DIR` | parent of `monitor/` | the repository; runner scripts, prompt files and input files are relative to it |
| `--runs DIR` | `<harness-root>/runs` | the folder whose episodes are listed (searched up to three levels deep for `episode.json`) |
| `--data-dir DIR` | `<runs>/monitor` | where episodes started from the page go, with `settings.json` and `labels.json` |
| `--python PATH` | `$FH_PYTHON`, else the interpreter running the monitor | interpreter that runs the runner scripts (it needs the `forceharness` package and, for the real robot, the `[real]` extra) |
| `--codex-bin PATH` | `$CODEX_BIN`, else `codex` on PATH | Codex CLI executable for the `codex-cli` model source |
| `--env-file FILE` | none (`run.sh` passes `<repository>/.env`) | load `KEY=VALUE` lines into the monitor process; values are never printed |
| `--host`, `--port` | `127.0.0.1`, `8765` | where to listen |
| `--allow-live` | off | allow real-robot episodes; refused unless the host is loopback |
| `--video-config FILE` | `monitor/video.json`, else `monitor/video.example.json` | camera for session videos |
| `--no-video` | off | do not record videos this time |

To look from another computer, forward the port (`ssh -L 8765:localhost:8765 <robot workstation>`) and open the same
address locally. Stop with Ctrl+C: running episodes get an interrupt and stop through the runner's normal path.

### Review recorded runs

Review needs no robot, no model and no runner: point `--runs` at any folder that holds run folders (an `episode.json` with
its `<name>_relay/` folder next to it, or the monitor's own session folders) and browse the Records, Episode, Compare and
Videos pages. The page follows a running episode live, whoever started it (the page, a terminal, another program): it
watches the files, so an episode started from a terminal shows up under "Running now" on its own.

### Demo episodes (no robot)

In demo mode the runner drives a synthetic robot and scene, so the whole loop (runner, relay, model, records, the
episode page) can be checked without hardware. Model choices in demo mode: the runner's **scripted demo** (fixed actions,
no model call), the offline **mock model**, **manual**, an **external relay**, or a configured **API model**. The demo has
no physics of the real objects, so it cannot verify task success; it checks the plumbing.

### Real-robot episodes

A real-robot episode starts only when all of these hold: the monitor was started with `--allow-live` on loopback; the
task has a `live` entry in `tasks.json`; every item of the task's checklist is ticked (a person at the robot with the
emergency stop in hand, the scene set up, the robot PC services and the force calibration ready); the input files are
chosen from the listed ones; no other real-robot episode is running; the model is not the mock. The runner is started
as

```
<python> scripts/real/<task>.py --condition A|B --live --execute --auto-confirm --calibration <file> [task flags] --out <episode> --relay-dir <relay>
```

with the force calibration chosen on the page (default `configs/real/force_calibration.json`; any
`configs/real/*calibration*.json` can be picked). Every force, torque, speed, workspace and joint-limit guard is the
runner's own and stays in force; the monitor adds none and removes none.

**Interrupt** sends Ctrl+C (SIGINT) to the runner, which brings the controller to a soft stop; it is **not an emergency
stop**. **Force quit** (SIGKILL to the runner's process group) is offered only if the runner has not exited 8 s after an
interrupt. **Finish next turn** makes the relay answer the next turn with `done` (recorded as ended by the operator, not
by the model).

## Tasks

`monitor/tasks.json` is the only place where the monitor knows the runners' command lines. Each entry gives the runner
script, the condition flags, the demo and live flags, the live checklist, the guard lines drawn on the force chart, the
score options and, for a fixed task, the prompt file with its goal.

| Task id | Task | Runner |
|---|---|---|
| `blackboard_wipe` | Wipe the whiteboard | `scripts/real/blackboard_wipe.py` |
| `screw_insert` | Screw the green screw into the purple hole | `scripts/real/screw_insert.py` |
| `peel_vegetable` | Peel the vegetable (cucumber) | `scripts/real/peel_vegetable.py` |
| `usb_insert` | Insert the USB plug into the power strip (USB-A) | `scripts/real/usb_insert.py` |
| `can_weight_sort` | Sort bottles by weight (three water bottles) | `scripts/real/can_weight_sort.py` |
| `touch_firmness` | Tell soft from hard by touch | `scripts/real/touch_firmness.py` |
| `free` | Free task: you write the task | `scripts/real/free_task.py` |

- **Fixed tasks** read their goal from `forceharness/prompts/real/<task>.md` when an episode starts; conditions A and B get
  the same text. The Live page shows it and can edit it: saving writes the file back and affects only episodes started
  afterwards (each episode record keeps the text it used). A save is refused when the file changed elsewhere since the
  page loaded it, or while the second episode of a pair is still queued. If the file is missing, the task's records stay
  viewable but the task cannot be launched.
- **Free task**: the text you type is the task; it goes into the system prompt in full (the same for both conditions),
  as one command-line argument (`--instruction=...`), never through a shell.
- **Resume holding an object**: for tasks that allow it, a live episode can start with the gripper already closed on an
  object you name (`--resume-held=<object>`); tick it only when the gripper really holds something.
- Records are kept apart by the record's `task_id`, so each task has its own comparison.

Adding a fixed task on the same runner: add its entry to `tasks.json` and one `FixedTaskAdapter` to `ADAPTERS` in
`fhmon/episodes.py`. A runner with a different record format needs its own adapter class there.

## Models

The runners talk to the model through the harness relay protocol: the runner writes `obs_N.md` (+ `obs_N_*.png`) and
`pending.json`, and waits (up to 10 minutes) for `cmd_N.json` with `{"calls": [...], "note": "..."}`. Anything that
writes that file can be the model, so the model choices are:

| Choice | What happens |
|---|---|
| **API profile** | The monitor sends the full observation text and the images of the newest turns (2 by default) to the model, parses the JSON action out of the reply and writes `cmd_N.json`. The reply and thinking stream into the page; latency, tokens and cost (when prices are set) are logged. A reply without a parsable action is answered with a short correction and the model is asked again (at most three attempts per turn). |
| **Codex CLI** | Same conversation, sent through `codex exec --ephemeral --sandbox read-only --json` in an empty temporary folder, one call per turn, images attached with `-i`. It is billed to the ChatGPT plan the CLI is signed in with, not per token; every `*_API_KEY` variable is removed from its environment so it cannot fall back to API billing. Install the Codex CLI, sign in (`codex login`), and have `codex` on PATH or pass `--codex-bin` / set `CODEX_BIN`. The Settings page lists the plan's models from Codex's local cache. Each turn takes roughly 20–60 s (at most 420 s). Codex adds its own instructions around the model, so keep its results apart from API runs (the session log records `provider: codex-cli`). |
| **Scripted demo** | Demo mode only: the runner's own fixed action sequence, no model call. |
| **Mock model** | Demo mode only: replies with the example action from the observation twice, then `done`. Checks the pipeline at no cost. |
| **Manual** | Each turn the page shows the example JSON; edit it and submit, it is written as is. |
| **External relay** | The monitor only watches; another agent or a person writes the `cmd_N.json` files. |

API profiles support OpenAI-compatible chat (`/chat/completions`, also for other compatible servers such as a local vLLM),
OpenAI Responses (`/responses`) and Anthropic Messages (`/v1/messages`). A profile stores no secret: a key typed on the
Settings page lives only in the monitor's memory (gone on restart, and dropped when the interface or Base URL changes so it
is never sent to another host), or the profile names an environment variable such as `FH_API_KEY` or `ANTHROPIC_API_KEY`
(put them in `<repository>/.env`). The model picker lists the models the key can use (the server asks the API; the key never
reaches the page), common names from the harness price table `forceharness/eval/prices.yaml`, and a free-text entry; picking a
model fills in its price. "Test connection" sends one short request.

Retries: a failed call is retried twice (after 3 s and 6 s); a rate limit (429, "too many requests") waits 20 s, 40 s and
80 s; a used-up quota is not retried: the turn shows an error with a **Retry turn** button, and the runner keeps the arm
holding while it waits. A Codex call that prints nothing for 150 s is stopped and retried like a rate limit.

**Take over a relay**: an episode started elsewhere that is waiting for its model shows "Take over relay with API model".
The earlier turns are sent to the model as history. Use it only when nobody else is answering in that relay folder.

**Extra instruction**: for tasks without a fixed goal or task text, an optional note can be sent with the observation of
turn 0 (the same text in both conditions of a pair). Prompts, the action menu and the force translation are the runner's and
are never changed by the monitor. The relay's system prompt is shown under Settings → About this monitor.

## Session videos

Each episode started from the page can be recorded by one global RealSense camera (`fhmon/camrec.py`, a separate process:
RealSense → ffmpeg → MP4). Copy `monitor/video.example.json` to `monitor/video.json` (it is git-ignored) and fill in:

| Field | Meaning |
|---|---|
| `enabled` | `true` to record (the example ships with `false`) |
| `modes` | which episodes to record: `["live"]` (default) or also `"demo"` |
| `dir` | where videos go; `runs/...` means the runs folder in use (it must be inside it for the page to play them) |
| `python` | interpreter with `pyrealsense2` and Pillow for the recorder; empty means the monitor's own interpreter; `ffmpeg` must be on PATH |
| `camera` | `serial` of the recording camera, a `label`, `width`, `height`, `fps` |
| `never_open` | serial → reason for every camera the harness itself opens (exterior and wrist cameras): a RealSense serves one program at a time, so the recorder refuses them and never touches the device |
| `tail_s`, `max_hours` | seconds recorded after the runner exits, longest recording |

The recorder starts right after the runner and stops `tail_s` seconds after it exits; the runner never waits for it, so a
missing or busy camera only marks that video failed. Videos are `<dir>/<episode name>.mp4` with `.json` (status, camera,
first-frame time, frame counts, sha256), `.preview.jpg` and `.log` beside it; the monitor never deletes them. Video time
`t` is wall time `first_frame_unix + t`, which lets the episode page jump to the moment a turn's action started. Restart the
monitor after editing the file; `--no-video` skips recording for one run.

## Where things are written

| Path | Content |
|---|---|
| `<data-dir>/<session>/session.json` | what was asked: task, condition, mode, model (no key), instruction, runner command, status, exit code |
| `<data-dir>/<session>/episode/` | the runner's `--out`, the harness's own record |
| `<data-dir>/<session>/relay/` | the runner's `--relay-dir`, the public side of the relay |
| `<data-dir>/<session>/bridge.jsonl` | one line per model turn: observation size, images sent, reply, thinking, latency, tokens, cost, retries |
| `<data-dir>/<session>/runner.log` | the runner's output |
| `<data-dir>/settings.json` | model profiles, never keys |
| `<data-dir>/labels.json` | hand-set success flags and notes; a flag wins over the harness score in every table and comparison, clearing it falls back |
| `<data-dir>/attach/` | logs of relays taken over from episodes started elsewhere |
| `<runs>/videos/` | session videos (see above) |

The monitor writes nothing else, except `cmd_N.json` in a relay folder it answers for and the fixed task prompt files you
save from the Live page.

## Reading the force panel

- The **Force** panel shows the raw reading of the episode's force stream (for the FR3, the wrist wrench estimated from
  joint torques, not tared, so an offset of 1–2 N is normal), in seconds since the episode started, with the task's guard
  lines, grey bands for the actions and ▼ for contact events. The robot PC clock need not match this computer's; the stream
  is aligned using the file's modification time (about ±1 s).
- The curve under **Action details** is the primitive's own record (tared, reaction-force convention: the force of the
  environment on the hand), the same data the harness turned into physical quantities.
- Demo runs have no continuous force log: the primitives' own traces are joined end to end.

## Safety

- The monitor never commands the robot. It starts the runner, which enforces every limit and check of its own; real-robot
  episodes need `--allow-live`, loopback only, the full checklist, and only one runs at a time.
- Interrupt is a soft stop through the runner, not an emergency stop. Keep a person at the emergency stop.
- Keys: never written to disk by the monitor; runners are started without any `*_API_KEY` variable (in relay mode they
  never call a model themselves).
- The page: every POST carries a per-start token embedded in the page (no cross-site requests), the Host header must name
  the server on loopback (no DNS rebinding), and `/api/file` serves only images, text and MP4 files from the runs folders.
  Listening beyond loopback (`--host`) prints a warning: anyone who can open the page can start demo episodes.

## How it fits the harness

- `fhmon` never imports `forceharness`, and the harness never imports `fhmon` (a test checks both).
- Command lines live only in `tasks.json`; run-folder formats are parsed only in `fhmon/episodes.py` (and the location of
  the force stream in `fhmon/streams.py`); the patterns that recognise the harness's own wording in tool results (which lines
  are force translation, which are scene notes) live only at the top of `fhmon/receipts.py`. When a runner changes its flags,
  record format or wording, those are the places to update.

## Tests

```bash
python3 -m pytest -q -p no:cacheprovider monitor/tests     # from the repository root; offline, no robot, no network
```

They cover the tool-result split, reply parsing, the relay bridge (images, history, correction round, never overwriting an
answered turn, manual, mock, finishing early, retries and quotas), the Codex CLI source with a fake `codex`, the episode
adapters, stream alignment and tailing, the page token, Host check and path traversal, the live gate and checklist, editing
fixed task goals, labels, and the import boundary. The video tests need `ffmpeg` and use synthetic frames only.

## Limits

- Only the real-robot record format (`free_task/1`) is read; simulator runs (`runs/<run>/transcripts/`) are not shown.
- After a restart, episodes started earlier are still listed, but the stop controls work only for episodes this process
  started; the manual, mock and take-over relays live in memory and must be started again.

## Files

```
monitor/
  run.sh                 start script
  tasks.json             runner registry: command lines, conditions, live checklist, score options, fixed goal files
  video.example.json     session-video settings to copy to video.json
  fhmon/
    __main__.py          command line
    server.py            HTTP, SSE, static files, request checks
    episodes.py          run folder -> episode (the only place that knows the record format)
    receipts.py          tool result -> motion / force translation / scene lines (all harness wording patterns)
    streams.py           force streams: whole-episode overview, live tailing, clock alignment
    live.py              episode index, watchers, event hub, comparison statistics (Wilson interval)
    bridge.py            relay bridge: obs -> model -> cmd (API, Codex CLI, mock, manual)
    llm.py               streaming clients on urllib: OpenAI-compatible chat, OpenAI Responses, Anthropic
    codex_cli.py         the Codex CLI as a model source
    sessions.py          start and stop runners, pairs, take-over, scoring
    settings.py          model profiles (keys only in memory or environment variables)
    labels.py            operator labels
    video.py, camrec.py  session videos and the recorder
  web/                   the page (plain JavaScript modules)
  tests/                 offline tests
```
