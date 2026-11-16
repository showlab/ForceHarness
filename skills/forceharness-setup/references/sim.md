# HiddenPhys (MuJoCo) details

- **Versions**: robosuite 1.5.1 with mujoco 3.2.7 exactly. With newer MuJoCo, robosuite 1.5.1 fails an assertion in
  `get_joint_qpos_addr`; `pip install -e ".[sim]"` pins both.
- **Rendering**: the paper's HiddenPhys runs give the model no images (`vision: false`), so rendering is only needed for
  `check_sim.py` images (skip them with `--no-render`), demonstration recording and the learned baselines. Defaults:
  `MUJOCO_GL=egl` on Linux (choose the GPU with `MUJOCO_EGL_DEVICE_ID`; the EGL index may not equal the `nvidia-smi` index,
  so test with one short render), `cgl` on macOS. Linux without a GPU: `MUJOCO_GL=osmesa` (slow).
- **System libraries (Linux)**: MuJoCo loads EGL even with `--no-render`, so a bare machine needs
  `sudo apt-get install -y libegl1 libgl1` (add `libosmesa6` for osmesa), as the CI workflow does. Windows: use WSL2.
- **Force sign**: with `force_sign = -1`, pressing the empty hand down onto the table gives reaction Fz > 0. `check_sim.py`
  verifies it per task.
- **Noise model** (paper protocol): force noise 0.5 N per tick, residual bias 0.3 N per episode, object poses as noisy text
  (4 mm, 2 degrees; the peg task uses 15 mm). Budgets: 40 turns, 80 tool calls, 300 s simulated time.
- **Seeds**: development 9000-9019, test 9100-9149 (tables use 9100-9119), demonstrations 20000 and up. Never tune on
  test seeds.
- **Conditions**: `phys2|force` is ForceHarness (force primitives + L2 physics translation); `none|pose` is the no-force
  control. Other forms (`raw`, `events`, `signals`, `plot`, `phys1`, `phys3`, `phys4`) are ablations.
- **Workers**: `run_matrix.py --workers N` runs N episodes in parallel processes; each holds one MuJoCo instance (and an
  EGL context when rendering). Episodes already in `results.jsonl` are skipped, so an interrupted run resumes.
- **Costs**: always `--dry-run` first. GPT-6 Astra without images costs roughly US$0.2 to 0.6 per episode; the no-force
  control uses more calls on hard tasks. Claude models with extended thinking can use two to three times more calls.
- **Strict tool schemas**: OpenAI function tools run in strict mode, where the model fills every field (unused ones with 0).
  If you add a tool, make sure a 0 in an optional field cannot mean something harmful, and run one real episode before a batch.
