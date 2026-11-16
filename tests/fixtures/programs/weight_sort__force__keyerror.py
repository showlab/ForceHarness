# Fake-model program with a deliberate bug: after weighing it reads a wrong field name (payload_kg); the KeyError fails the
# episode and the traceback shows only the program's frames.
scene = robot.scene()
find = lambda n: next(o for o in scene if o["name"] == n)
cube, table = find("cube"), find("table")
top = table["top_z"]
a = ((cube.get("yaw_deg", 0.0) + 45.0) % 90.0) - 45.0
robot.rotate_z(a)
robot.move_to(cube["pos"][0], cube["pos"][1], top + 0.15)
robot.move_until_contact([0, 0, -1], max_dist=0.2, f_threshold=3.0)
robot.move_by(0, 0, 0.018)
robot.grasp()
lift = robot.lift(height=0.10, hold_time=1.0)
m = lift["derived"]["payload_kg"]
robot.done(success=True, verdict="heavy" if m > 0.375 else "light")
