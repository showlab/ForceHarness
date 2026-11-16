# Fake-model program for weight_sort (pose interface, no force), following the expert on the pose interface: descend to
# the nominal height, grasp, lift, and guess light.
scene = robot.scene()
find = lambda n: next(o for o in scene if o["name"] == n)
cube, table = find("cube"), find("table")
top = table["top_z"]
zones = {o["name"]: o for o in scene if o.get("kind") == "zone"}
a = ((cube.get("yaw_deg", 0.0) + 45.0) % 90.0) - 45.0
robot.rotate_z(a)
x, y = cube["pos"][0], cube["pos"][1]
robot.move_to(x, y, top + 0.15)
robot.move_to(x, y, top + 0.02)
g = robot.grasp()
up = robot.move_by(0, 0, 0.10)
print("grasp:", g, "\nlift:", up)
assert "force" not in str(sorted(up.keys())), "pose results must not contain force fields"
verdict = "light"                          # no force: can only guess
z = zones[verdict + "_zone"]
robot.move_to(z["pos"][0], z["pos"][1], top + 0.15)
robot.move_to(z["pos"][0], z["pos"][1], top + 0.025)
robot.release()
robot.move_by(0, 0, 0.10)
robot.done(success=True, verdict=verdict, note="guessed: no force information")
