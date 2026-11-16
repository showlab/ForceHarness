# Fake-model program for peg_in_hole_hard (pose interface): no search or insert primitives, so it can only press straight
# down at the noisy hole position to 2 cm depth (usually blocked by the hole block, tripping the safety stop).
scene = robot.scene()
hole = next(o for o in scene if o["name"] == "hole")
hx, hy, hz = hole["pos"]
z_in = hz + 0.04 - 0.02
robot.move_to(hx, hy, hz + 0.07)
r = robot.move_to(hx, hy, z_in)
print("press:", r["result"], r["end_pos"], r.get("short_of_target_m"))
try:
    robot.search()                         # not on the pose interface: must raise RobotError, which the program catches
except Exception as e:
    print("expected error:", e)
inserted = r["end_pos"][2] <= z_in + 0.005
robot.release()
robot.move_by(0, 0, 0.08)
robot.done(success=bool(inserted))
