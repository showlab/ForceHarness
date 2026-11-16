"""ForceHarness monitor: a standalone web console for real-robot episodes.

It sits outside the three layers of the harness and only looks at them from the outside. Boundary
with the harness (see monitor/README.md):
- this package never imports `forceharness`, and the harness never imports it;
- it reads the files the harness already writes (episode.json, action_NN.json, the force stream
  nuc/nuc.jsonl, relay obs_N.md / cmd_N.json);
- it starts the harness's own runner scripts as subprocesses (monitor/tasks.json);
- with a model selected it answers relay turns (obs_N.md -> cmd_N.json), which otherwise a person
  or another agent writes by hand.
Standard library only. The optional session-video recorder (camrec.py) runs in a separate camera
interpreter with pyrealsense2 and Pillow.
"""
__version__ = "0.1.0"
