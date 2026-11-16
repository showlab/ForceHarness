"""Real-robot backend (Franka FR3 under Polymetis): workstation side of the three layers.

Layer 1 (force sensing): config.py, backend.py (ContinuousBackend: calibrated wrench, force sign, zero), relax.py,
posture_bias.py / posture_calibration.py. Layer 2 (primitives and guards): primitives.py (ContinuousPrimitives),
events.py, targets.py, rotation_progress.py, stop_validation.py, checks.py, validation.py, menu.py. Layer 3 and the
episode: runner.py (the loop and the tool results), receipt_text.py, scene_objects.py / pixel_target.py /
wrist_overlay.py / cameras.py (perception), relay.py (the file relay to the model), tasks.py (the six tasks).
Transport to the robot PC: transport.py and gripper.py; the robot PC side is robot_side/franka_polymetis.
plant.py is a synthetic plant for offline tests. Importing this package has no hardware side effects.
"""
