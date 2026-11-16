"""ForceHarness: lets a frozen frontier model, reached only through an API, use wrist force for contact-rich
manipulation.

The model calls tools; the harness executes each call as a self-terminating force primitive and returns physical
quantities computed from the force trace as the tool result.  Three layers, bottom-up:

1. Force sensing and calibration: the backend supplies a calibrated estimate of the force the environment applies
   to the hand (a wrist force/torque sensor, or (J^T)^+ tau_ext from the joint torques of a torque-controlled arm),
   tared before contact, read after settling, reported as changes (``backends``, ``real``).
2. Self-terminating force primitives (``primitives``): one shared control loop; force through the lead of the
   target pose over the measured pose; admittance control; stop on contact events; a fixed guard on force, speed
   and workspace that the model cannot change.
3. Physics translation layer (``translation``): fixed estimators turn each primitive's force trace into physical
   quantities with uncertainties and flags, rendered as the tool result (level L2, "phys2", by default).  The same
   package holds the other force forms used as ablations (none / raw / stats / events / signals / plot).

Package layout:
  primitives   primitive library, contact event detector, ContactReport
  translation  estimators and renderers of the force information (the force forms)
  agent        model API clients (OpenAI-compatible and Anthropic), tool schemas, the episode loop
  backends     robots and simulators under the harness: MuJoCo/robosuite, ForceBench (Isaac Lab), a fake for tests
  hiddenphys   tasks of the HiddenPhys benchmark (MuJoCo)
  baselines    scripted expert, classical controller, Code-as-Policies, Diffusion Policy
  data         demonstration loading for the learned baselines
  eval         matrix runner, metrics (Wilson intervals), model prices
  real         the real Franka FR3 backend
  prompts      system prompts per benchmark

The model is never inside the millisecond control loop: it decides only between primitives (about 0.1-0.5 Hz).
"""
__version__ = "0.1.0"
