You control a Franka Panda arm in a tabletop manipulation simulation by writing ONE complete Python program. The program drives the
arm through a small set of motion primitives, exposed as methods of the object `robot`. It runs from start to finish without you: you
see nothing while it runs, so the program itself has to read what each primitive returns and decide what to do next.
Each primitive runs its own closed-loop controller at 20 Hz and returns when it finishes, when it is blocked, or when a safety limit trips.

Frames and units: world frame = robot base frame; x points away from the robot base, y to the robot's left, z up; metres.
The "tool centre point" is the point between the fingertips. The gripper opens to about 80 mm. The table top is near z = 0.80 m.
The wrist force/torque sensor sits between the arm and the gripper; readings are gravity-compensated at the start of the episode.

Action interface for this episode: $interface.
What your program can sense: $sensing_note

Robot API (every call blocks until the primitive has finished; arguments can be given by position or by name):
$api_doc

Every primitive returns a read-only dict (copy it with dict(...) before modifying) with these fields:
$returns_doc

Rules:
1. The scene below is also available at run time as robot.scene(); positions you are given are approximate (a few millimetres of error).
2. Prefer small, deliberate motions near objects. Any motion whose wrist force exceeds $f_cap N is aborted by the safety layer.
3. When something is uncertain and cannot be seen, the program may probe it physically (touch, lift, push) if the API allows it.
4. End with robot.done(success=..., verdict=...): success=True only if the program has good reason to believe the task is actually
   complete; give a verdict when the task asks for one. robot.done ends the program.
5. Budget: at most $max_tool_calls robot calls (robot.scene counts; robot.state and robot.done do not) and $max_sim_time s of simulated
   time. The program is stopped as soon as either is exceeded. An uncaught exception also ends the episode. A failed robot call raises
   an exception (its message says why); you may catch it with `except Exception`.
6. numpy (as np) and math are already imported; you may also import json, itertools, collections, statistics, functools, random, re,
   copy, typing. No file, network or system access. Whatever the program prints is recorded.
7. Reply with exactly one ```python code block that contains the complete program, and nothing else.
