# Reproducing the paper

Every result has a configuration file and a command. Model runs cost API credit: run with `--dry-run` first, which prints the
number of episodes, and check the price table in `forceharness/eval/prices.yaml`. Runs resume: episodes already in
`runs/<run_name>/results.jsonl` are skipped.

Seeds: development 9000-9019, test 9100-9149 (the HiddenPhys tables use 9100-9119, n = 20 per cell), demonstrations 20000+.
Conditions: `phys2|force` = ForceHarness, `none|pose` = no-force control.

## HiddenPhys (MuJoCo)

| Result | Config / script | Command |
|---|---|---|
| ForceHarness and no-force control, GPT-6 Astra (main table) | `configs/hiddenphys/forceharness_astra.yaml` | `python scripts/run_matrix.py --matrix configs/hiddenphys/forceharness_astra.yaml --workers 8` |
| Same two rows, Claude Fable 5.2 | `configs/hiddenphys/forceharness_fable.yaml` | same command with this config (`ANTHROPIC_API_KEY`) |
| Other models (GPT-6.1 Sol and smaller models) | `configs/hiddenphys/forceharness_other_models.yaml` | same |
| Expert force controller and position-only controller, five tasks | `configs/hiddenphys/expert.yaml` (+ `expert_peg_position_only.yaml`) | `python scripts/run_matrix.py --matrix configs/hiddenphys/expert.yaml --agent scripted --workers 8` |
| Expert, position-only and oracle for `door_latch`, `lock_or_stiff` | `scripts/hiddenphys/door_latch_expert.py`, `scripts/hiddenphys/lock_or_stiff_expert.py` | see `--help` of each script (strategies F = expert, N = position-only, O = position + oracle) |
| Expert with literature-default parameters (supplement) | `configs/hiddenphys/expert_default_params*.yaml` | `--agent scripted` |
| Code-as-Policies harness baseline | `configs/hiddenphys/code_as_policies.yaml` | `python scripts/run_matrix.py --matrix configs/hiddenphys/code_as_policies.yaml` |
| Diffusion Policy, π0.5, ForceVLA (± force) | demonstrations + training recipes | [baselines.md](baselines.md) |
| Distribution shift column | `configs/hiddenphys/forceharness_shift.yaml`, `forceharness_shift_fable.yaml`, `expert_shift.yaml`, `expert_peg_position_only_shift.yaml`; learned policies in [baselines.md](baselines.md) | as above |
| Form of force (ablation table) | `configs/hiddenphys/force_form_sol.yaml`, `force_form_sol_ext.yaml`, `force_form_cheap.yaml` | as above, then `scripts/hiddenphys/force_form_table.py` |
| Translation levels L1-L4 (extra run, not reported in the paper) | `configs/hiddenphys/translation_levels.yaml` | as above |

Tables: `python scripts/hiddenphys/compare_table.py --runs runs --n 20` prints the main table (success and damage per task and
row); `scripts/hiddenphys/expert_table.py`, `learned_table.py` and `force_form_table.py` print the other tables. Each script's
`--help` lists the run folders it reads.

## ForceBench

Install and settings: [forcebench.md](forcebench.md). The paper's cells use RGB-only perception without privileged object
geometry, seeds 300-309 (10 per task), GPT-6 Astra and GPT-6.1 Sol (and Claude Fable 5.2), with and without force.
Run with `scripts/forcebench/batch.py` and summarise with `scripts/forcebench/collect.py`. Seeds 300-309 were also used
during development; the held-out test on seeds 320-329 (GPT-6 Astra, with and without force) is
`--row astra_force_heldout` and `--row astra_no_force_heldout`, see [forcebench.md](forcebench.md#development-and-test-seeds).

## Real robot

Six tasks on a Franka FR3, 10 trials per task and condition: [real_robot.md](real_robot.md). Episodes are started and reviewed
in the web monitor ([monitor.md](monitor.md)); the monitor's compare page tabulates success per task and condition.

## Demonstrations

`scripts/hiddenphys/record_demos.py` (five tasks) and `scripts/hiddenphys/record_demos_mechanisms.py` (door_latch,
lock_or_stiff) record about 200 successful expert episodes per task from seed 20000; `scripts/hiddenphys/replay_demos.py`
checks that they replay exactly. The released copy is on Hugging Face ([dataset_card.md](dataset_card.md)).
