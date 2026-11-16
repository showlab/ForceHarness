# HiddenPhys: a MuJoCo benchmark for force in contact-rich manipulation

HiddenPhys is the benchmark that ships with ForceHarness: seven manipulation tasks in robosuite 1.5.1 on MuJoCo 3.2.7,
each built around **one hidden physical variable that decides success** and that a camera cannot see. It exists to measure
what wrist force contributes to a frozen model, so it puts in one place

- tasks whose success depends on a physical variable that position alone does not reveal,
- classical and learned reference policies that run on exactly the same episodes (same seeds, same hidden values, same
  success checks), and
- a count of the harm done by pressing too hard (damaged handles).

Code: task definitions in `forceharness/hiddenphys/tasks.py`, the simulator wrapper in `forceharness/backends/mujoco/`
(`env.py`, custom scenes in `custom_envs.py`, the demonstration recorder in `recorder.py`), the scripted references in
`forceharness/baselines/` and `scripts/hiddenphys/`, one config per result in `configs/hiddenphys/`.

## Install and check

```bash
pip install -e ".[sim]"            # robosuite==1.5.1 and mujoco==3.2.7 (newer MuJoCo breaks robosuite 1.5.1)
python scripts/check_sim.py         # builds every expert task, saves camera images, checks the force sign, runs the expert once
```

Camera rendering is off-screen: `MUJOCO_GL` defaults to `egl` on Linux and `cgl` on macOS (set it yourself to override;
on a Linux machine without a GPU use `osmesa`). The paper's configs run without images, and `check_sim.py --no-render`
needs no OpenGL at all. On a multi-GPU machine the EGL device numbering (`MUJOCO_EGL_DEVICE_ID`) can differ from the
`nvidia-smi` order.

Without MuJoCo, an offline fake backend (`forceharness/backends/fake.py`: a point-like tool, a table and one cube) runs the
whole stack on `weight_sort` only; the unit tests and the smoke config use it:

```bash
python scripts/run_matrix.py --matrix configs/hiddenphys/smoke.yaml --agent scripted --backend fake --workers 1
python scripts/run_matrix.py --matrix configs/hiddenphys/smoke.yaml --agent scripted --workers 2     # the same on MuJoCo
```

## The tasks

The instruction states the range of the hidden variable, never its value; the value is drawn from the episode seed. Each
task covers one way force matters: judging a held load (Weight, CoM), finding an uncertain surface (Wipe, Stack), searching
by touch (Peg), and operating a mechanism without damage (Latch, Lock).

| Task (paper name) | Scene | Hidden variable | What the model is told | Success check (final state) |
|---|---|---|---|---|
| `weight_sort` (Weight) | robosuite Lift | cube mass, 0.15 kg (even seed) or 0.60 kg (odd seed) | both masses; place the cube in the matching 16 cm zone and give a verdict | cube centre < 6 cm above the table and within 8 cm of the true zone centre in x and y, fingers open > 3 cm, no gripper contact |
| `com_sort` (CoM) | ComSortBar (custom) | centre of mass of a 12 cm, 0.8 kg bar: centred (odd seed) or 4 cm along the long axis (even seed; direction alternates with seed // 2), 0.31 N·m at the wrist in theory | the mass and both cases; grasp at the geometric centre, place in the matching zone, give a verdict | as Weight, with the bar and fingers open > 5 cm |
| `wipe_hard` (Wipe) | robosuite Wipe (finger-less wiping tool) | table height: min(0, N(0, 3 cm)) below the nominal 0.90 m | the table may sit up to about 6 cm low; a spot comes off only under at least 5 N; more than 15 N fails | all 12 spots wiped, each counted only at a normal force of at least 5 N (lighter wipes are put back); peak normal force below 15 N |
| `stack_gentle` (Stack) | robosuite Stack | top of the green cube, reported with a fixed error uniform in ±1 cm | the top is known to about ±1 cm; place the red cube gently | robosuite's check: red cube centre more than 4 cm above the table, not grasped, touching the green cube |
| `peg_in_hole_hard` (Peg) | PegInHoleRound (custom) | hole position, reported with 15 mm Gaussian error per axis; the square peg clears the round wall by 4.5 mm; the peg starts in the gripper | clearance about 4 mm, position off by up to about 3 cm, feel for the hole, insert at least 2 cm | peg centre within 7.5 mm of the hole centre in x and y, bottom more than 2 cm below the reported hole height (release is not checked) |
| `door_latch` (Latch) | robosuite Door with latch | unlock angle of the lever handle, uniform in 15-35°, end stop 8° beyond; a detent torque rises to 0.45 N·m over the last 8° before unlocking; hinge damping U[0.5, 4], friction U[0, 1] | the range, the end stop, the 15 N damage line; press until the latch releases, keep it pressed while starting to pull, open at least 17° | door open beyond 0.3 rad (17°) and no handle damage |
| `lock_or_stiff` (Lock) | robosuite Door | locked (even seed) or stiff (odd seed); a stiff door breaks free when the force at the handle reaches F_b ~ U[6, 11] N and moves exactly like a locked one until then | the handle is already held; both classes, the 6-11 N, the 15 N damage line; open a stiff door at least 17° and answer `stiff`, leave a locked one and answer `locked` | no damage; door > 17° if stiff, < 2° if locked. The verdict is recorded as `verdict_correct`; the expert scripts also require it |

Details that matter when reading results:

- **Damage.** On `door_latch` and `lock_or_stiff` any wrist reaction above 15 N while the gripper is commanded closed damages
  the handle (`task_state["damaged"]`, `truth.damaged` in the records) and fails the episode; on `lock_or_stiff` it is
  counted from 1 s after the gripper closes, because closing on the handle causes a short transient. The **Damage** column
  of the tables counts these episodes. On `wipe_hard`, a normal force of 15 N or more fails the episode but is not counted
  as damage.
- **Safety stop.** Every harness, script and program row runs through the same primitives, which stop at 25 N or 3 N·m
  (the model cannot change this). The stop acts after the crossing, so logged peaks can exceed 25 N. Learned policies drive
  the controller directly with no stop, which is why their damage and peak force are reported.
- **Pre-grasp on Lock.** The environment grasps the handle before every `lock_or_stiff` episode (`TaskSpec.pregrasp_hook`),
  for every method; this is not counted as a tool call and its result is not shown to the model.
- **Mechanism logic.** On `door_latch` the hinge is held by a joint limit until the lever passes the unlock angle; if the
  lever drops more than 2° below it while the door is still within 1.5° of the frame, the latch locks again. On
  `lock_or_stiff` locked and stiff doors are held by the same joint limit, so before breakaway they are indistinguishable in
  position; a soft constraint such as MuJoCo friction loss would let a stiff door creep and leak the class.

## What the model sees

**Poses are given as noisy text, not images.** HiddenPhys is about force, not visual perception, so the `scene` tool
returns ground-truth object poses plus an offset per object that is fixed within an episode (`ForceSim.noised`): 4 mm
standard deviation per axis in x and y and 2° in yaw (15 mm on the Peg hole, 25 mm on its shifted variant). Heights are
not perturbed by this offset; height uncertainty is part of each task (table height on Wipe, the reported cube height on
Stack). Geometry a camera would see, such as the door handle bar and panel, is described in the scene text. Repeated
`scene` calls return the same offsets. The offsets come from the episode's random generator, which also draws the force
noise, so they depend on what ran before the first `scene` call: harness and expert rows see identical offsets on a seed,
learned policies offsets from the same distribution but not the same values.

Tool results omit durations, simulated time and the finger opening while the gripper is closed, since these leak the
load (`proprio: minimal`). Under the no-force control (`none|pose`) the menu has only position tools and the results carry
kinematics only; the safety limits stay. Some kinematic signals remain readable and were left in: a heavy cube lowers the
hand by 2-3 mm on Weight, and on Wipe the scene lists only the spots that are left.

**Force** is read from MuJoCo's wrist force/torque sensors, rotated to the world frame and reported as the reaction of the
environment on the hand (pressing down on the table gives +z). It is tared at rest (the mean of 10 ticks after reset, and
again after the primitives level the tool), and a noise model makes it closer to a real arm's estimate:
0.5 N (0.05 N·m) Gaussian noise per axis per tick and a residual per-episode bias of 0.3 N (0.03 N·m) per axis after taring.

## Protocol

All results use the episode block shared by the configs in `configs/hiddenphys/` (`scripts/run_episode.py
--paper-defaults` applies the same values to a single episode):

| Item | Value |
|---|---|
| Robot and control | Panda, robosuite BASIC composite controller (OSC pose), fixed impedance, damping ratio 1, 20 Hz; full scale 5 cm and 0.5 rad per tick |
| Stiffness | kp 150; the primitives use 40 along a pressed axis and 10 on wrist yaw during `pull`; kp 1500 only for the stiff-arm Lock checks |
| Arrival tolerance | 3 mm (0.1 mm on Lock) |
| Contact / stop | contact at 3 N; safety stop at 25 N and 3 N·m |
| Force noise | 0.5 N per tick, residual bias 0.3 N per episode (`force_noise_std`, `force_bias_std`) |
| Scene noise | 4 mm in x and y, 2° in yaw (`scene_noise_pos`, `scene_noise_yaw_deg`) |
| Budget | 40 turns, 80 tool calls, 300 s of simulated time, 1200 s of wall clock |
| Images | none (`vision: false`) |

**Seeds.** Development 9000-9019; test 9100-9149; demonstrations 20000 and up (20000-20209, up to 20239 on Peg). The main
table uses test seeds 9100-9119 (20 per cell); the scripted references and Diffusion Policy also ran 9100-9149, and the
Code-as-Policies baseline and the other models 9100-9104. Checks and tuning used other seed blocks, never the test seeds.
A seed fixes the hidden variable and the layout, so the hidden values match across all rows; per-tick force noise is not
paired. Weight, CoM and Lock balance their classes by seed parity.

**Scoring.** A cell counts successes by the checks above, after `done` or the budget (learned policies run a fixed
horizon). A record with a non-empty `error` is an infrastructure failure (API or simulator), not a task failure: remove
its line from `results.jsonl` and resume to rerun it (the paper's tables contain none). A crashing Code-as-Policies program
is a task failure and has no `error`. We report Wilson 95% intervals and exact McNemar tests on paired
seeds; at 20 episodes, 20/20 has a lower bound of about 84%, so differences of 2-3 episodes in a cell are not conclusions.

## Running the references

| Row | How |
|---|---|
| Expert force controller, five tasks | `python scripts/run_matrix.py --matrix configs/hiddenphys/expert.yaml --agent scripted --workers 8` (force interface rows) |
| Position only, four tasks | the same run (pose interface rows: the expert script without force) |
| Position only, Peg | `configs/hiddenphys/expert_peg_position_only.yaml` (`baselines/classical.py::PositionOnlyPeg`), `--agent scripted` |
| Expert / position only / oracle on Latch | `python scripts/hiddenphys/door_latch_expert.py F` (expert), `N4tol` (position only, stepped probe), `N5half` (alternating probe), `O` (position + oracle) |
| Expert / position only / oracle on Lock | `python scripts/hiddenphys/lock_or_stiff_expert.py F`, `N3` (position only, 0.5° rule), `N3f` (0.1° rule), `O`; add `--kp 1500` for the stiff arm |
| Expert with constants from the instruction (supplement) | `configs/hiddenphys/expert_default_params_b.yaml` (Peg search radius 3 cm) and `expert_default_params.yaml` (1 cm), `--agent scripted` |
| Code-as-Policies (paid API calls) | `python scripts/run_matrix.py --matrix configs/hiddenphys/code_as_policies.yaml` (`cap\|pose` and `cap\|force`) |
| Diffusion Policy, π0.5, ForceVLA (± force) | demonstrations from `scripts/hiddenphys/record_demos.py` and `record_demos_mechanisms.py`, then [baselines.md](baselines.md); evaluations go to `runs/learned/<policy>_<task>/` and `runs/learned_shift/<policy>_<task>_ood/` |

The expert is one hand-written state machine per task over the same primitives, tool results, noise and safety stop as
the model. Its constants were chosen knowing the task design (for example the weight threshold is the midpoint of the two
masses, the Peg search radius of 5 cm was chosen knowing the 15 mm hole noise), so it measures how far a hand-tuned expert
gets, not a task-free method. The position-only variants used in the main table for Latch (`N4tol`) and Lock (`N3`) were
fixed beforehand as the best variants on development seeds. The mechanism scripts write
`runs/expert_checks/door_latch_<strategy>.json` and `runs/expert_checks/lock_or_stiff_<strategy>_kp<kp>.json`; use
`--seed-start`, `--n` or `--seeds` to change the seeds (default: 9100-9149).

Demonstrations: `record_demos.py` (five tasks) and `record_demos_mechanisms.py` (Latch, Lock) record successful expert
episodes from seed 20000 with two 128x128 cameras; `replay_demos.py` replays them and checks that they reproduce the
recorded outcome. The released copy is described in [dataset_card.md](dataset_card.md).

## Running ForceHarness

Model runs make paid API calls: put the keys in `.env` (see `.env.example`), and run each config with `--dry-run` first to
see the number of episodes. Runs resume; episodes already in `runs/<run_name>/results.jsonl` are skipped.

| Result | Config | Rows / cells |
|---|---|---|
| Main table, GPT-6 Astra | `forceharness_astra.yaml` | `phys2\|force` = GPT-6 Astra + ForceHarness, `none\|pose` = GPT-6 Astra, no force; 7 tasks x 20 seeds |
| Main table, Claude Fable 5.2 | `forceharness_fable.yaml` | the same two rows (Anthropic API, `ANTHROPIC_API_KEY`) |
| Other models | `forceharness_other_models.yaml` | the same two rows for GPT-6.1 Sol and six GPT-5.x models, 5 seeds |
| Shift column | `forceharness_shift.yaml`, `forceharness_shift_fable.yaml`, `expert_shift.yaml`, `expert_peg_position_only_shift.yaml` | both rows of GPT-6 Astra and of Claude Fable 5.2, the expert force controller and the position-only controller on the four `*_ood` tasks |
| Form of force | `force_form_sol.yaml`, `force_form_sol_ext.yaml`, `force_form_cheap.yaml` (+ `forceharness_other_models.yaml`) | the same force primitives with force rendered as event words, a plot, raw numbers, derived signals, L1 or L2 quantities |
| Translation levels (extra run, not reported in the paper) | `translation_levels.yaml` | `phys1`-`phys4` vs `signals` and the no-force control on Weight and CoM |

```bash
python scripts/run_matrix.py --matrix configs/hiddenphys/forceharness_astra.yaml --dry-run
python scripts/run_matrix.py --matrix configs/hiddenphys/forceharness_astra.yaml --workers 6
# one episode with the paper's protocol
python scripts/run_episode.py --task peg_in_hole_hard --model gpt-6-astra --provider openai-responses \
    --reasoning-effort low --paper-defaults --seed 9100
```

Conditions are named `fmt|interface`: `phys2|force` is ForceHarness (force primitives, physics translation at L2),
`none|pose` the no-force control, `cap|pose` / `cap|force` the Code-as-Policies baseline.

## Tables

The run folder of a config is `runs/<run_name>`, and every `run_name` equals the config's file name.

```bash
python scripts/hiddenphys/compare_table.py --runs runs          # main table: success per task, total, damage, shift
python scripts/hiddenphys/compare_table.py --runs runs --n 5    # first five seeds (next to Code-as-Policies)
python scripts/hiddenphys/expert_table.py --runs runs           # scripted references on 50 seeds (supplement)
python scripts/hiddenphys/learned_table.py --runs runs          # learned policies: peak force, damage, Lock by class
python scripts/hiddenphys/force_form_table.py --runs runs       # form of force, with paired McNemar tests
python scripts/aggregate.py --run runs/forceharness_astra       # per-condition summary of any run (cost, calls, ...)
```

| Row of the main table | Source under `runs/` | Produced by |
|---|---|---|
| Expert force controller | `expert/` (force interface), `expert_checks/door_latch_F.json`, `expert_checks/lock_or_stiff_F_kp150.json` | `expert.yaml`; `door_latch_expert.py F`; `lock_or_stiff_expert.py F` |
| Position only | `expert/` (pose interface), `expert_peg_position_only/`, `expert_checks/door_latch_N4tol.json`, `expert_checks/lock_or_stiff_N3_kp150.json` | `expert.yaml`; `expert_peg_position_only.yaml`; `door_latch_expert.py N4tol`; `lock_or_stiff_expert.py N3` |
| Oracle position script (supplement) | `expert_checks/*_O*.json` | `door_latch_expert.py O`; `lock_or_stiff_expert.py O` |
| DP, DP + force, DP images only (+ force), π0.5, π0.5 + force, ForceVLA | `learned/<policy>_<task>/` with policy `dp`, `dp_force`, `dpnoscene`, `dpnoscene_force`, `pi05_f0`, `pi05_f2`, `forcevla` | [baselines.md](baselines.md) |
| GPT-6 Astra, no force / + ForceHarness | `forceharness_astra/` (`none\|pose` / `phys2\|force`) | `forceharness_astra.yaml` |
| Claude Fable 5.2, no force / + ForceHarness | `forceharness_fable/` | `forceharness_fable.yaml` |
| Code-as-Policies, no force / + force | `code_as_policies/` (`cap\|pose` / `cap\|force`) | `code_as_policies.yaml` |
| Shift column | `expert_shift/`, `forceharness_shift/`, `learned_shift/` against the same rows on the base tasks | `expert_shift.yaml`, `forceharness_shift.yaml`, [baselines.md](baselines.md) |

The table scripts count only seeds 9100 .. 9100+n-1 (`--n`, default 20), so rows that ran 50 seeds are cut to the same
seeds and episodes are paired across rows. Folder names can be changed with flags (`--learned-run`, `--learned-shift-run`,
`--checks-run`, `--harness-runs`, ...; see `--help`). On Lock, script rows count success only with the right verdict, while
the success of model rows is the door outcome (their verdicts are in `verdict_correct`).

## Distribution shift

Four variants change only the distribution of the hidden variable; object names, scene format and success checks stay
the same, so trained policies run without retraining.

| Variant | Change | Instruction says |
|---|---|---|
| `weight_sort_ood` | masses 0.30 / 0.45 kg (base 0.15 / 0.60) | about 0.3 kg / about 0.45 kg |
| `com_sort_ood` | centre-of-mass offset 2.5 cm (base 4 cm) | shifted 2.5 cm |
| `wipe_hard_ood` | table height min(0, N(-3 cm, 3 cm)), i.e. 3 cm lower on average | up to about 9 cm below nominal |
| `peg_in_hole_hard_ood` | hole-position noise 25 mm per axis (base 15 mm) | off by up to about 5 cm |

The model reads the new range in its instruction, while the expert keeps its constants and learned policies keep their
training prompt (the policy server is told the base task name). The shift column reports successes over the four tasks
on seeds 9100-9119 (out of 80), in distribution and shifted, for the expert (`expert_shift.yaml`), ForceHarness
(`forceharness_shift.yaml`) and the learned policies (`runs/learned_shift/`). The scene notes of the shifted variants are
not updated (CoM still says 4 cm, Peg about 1 cm, Wipe ±2 cm).

## Known mismatches

These hold for every row and are kept so that results stay comparable:

- The shared system prompt puts the table top near 0.80 m and calls position errors a few millimetres, which is wrong for
  Wipe and Peg; their scene notes still say ±2 cm and about 1 cm (the instructions give the right ranges).
- On Peg the reported hole height is 2 cm below the true opening (`HollowCylinderObject` heights are half-heights), so
  success needs about 4 cm of insertion; levelling the tool at reset also tilts the welded peg by 4-12°.
- The word "gently" on Stack is not scored.
- On Wipe the table is often at the nominal height: the draw is min(0, Gaussian), so about half of the seeds have no
  offset (30 of the 50 test seeds are within 0.5 mm of nominal).
