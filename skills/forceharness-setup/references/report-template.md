# Setup report template

Keep this file as `runs/setup_report.md` and update it at every gate. Show the human the gate table after each stage.

```markdown
# ForceHarness setup report

- Date / machine: <date, OS, Python, GPU>
- Repository commit: <git rev-parse --short HEAD, or "no git">
- Target(s): HiddenPhys | ForceBench | real robot (<arm>, <controller>, <gripper>, <cameras>)

## Gates

| Gate | Verdict | Evidence (command -> key numbers) | Notes |
|---|---|---|---|
| 0 install + offline tests | PASS | `pytest -q` -> N passed, M skipped; offline episode success | |
| A0 simulator install | | doctor --target sim: mujoco 3.2.7, robosuite 1.5.1 | |
| A1 check_sim | | summary.json: sign_ok, expert_success per task | |
| A2 mechanism experts | | door_latch / lock_or_stiff F on seed 9100 | |
| A3 first model episode | | runs/... | cost US$ |
| C1 identify | | arm, controller, gripper, cameras | |
| C2 force source (spec) | expected OK / DEGRADED / NO_FORCE | force-sources.md row, vendor docs | |
| C3 control gate | PASS / WARN / FAIL | controller type, stream rates | |
| C4 safety gate | | e-stop confirmed by <human>, box x.. y.. z.. | |
| C5 backend | | deploy + selftest, or port + contract check | |
| C6 force gate (measured) | FORCE_OK / DEGRADED / NO_FORCE | M1 std 0.04 N, M2 drift 0.1 N, M3 3/3, M4 err 12 g | |
| C7 S2..S5 | | | |

## Hardware identification (real robot)
| Item | Value | How determined |
|---|---|---|

## Deviations from the reference setup
- <anything configured differently, every limit you tightened, anything not run and why>

## Next steps for the human
- <exact commands>
```
