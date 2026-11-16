# ForceBench checklist

Full instructions: [docs/forcebench.md](../../../docs/forcebench.md). Checklist for an agent:

1. `python -m forceharness.doctor --target forcebench`: glibc >= 2.35, a GPU with RT cores, driver >= 550.90.07
   (or the human explicitly opts in to the driver-check workaround after reading its warning), `websockets` and `msgpack`.
2. Ask the human to read and accept the NVIDIA Omniverse EULA and export the acceptance variable themselves.
3. Run `scripts/forcebench/install.sh` as documented (it downloads Isaac Sim 6.0.1, Isaac Lab 3.0 beta and ForceBench at a
   pinned commit; ForceBench's own scripts fetch its assets, some under CC BY-NC 4.0). About 40 GB.
4. One scripted episode (no API key), then one model episode after the human's OK.
5. Read success from the evaluator's `FORCEBENCH_VECTOR_RESULT=` line; Isaac's exit code is always 0.
6. Pitfalls: environments closer than 30 m apart darken each other's images (keep the default spacing); a Kit start can hang
   on a lock (the run script has a watchdog); too many Isaac processes on one GPU can render blank images (an empty-image check
   aborts the episode). The paper setting is RGB-only perception without privileged object geometry.
