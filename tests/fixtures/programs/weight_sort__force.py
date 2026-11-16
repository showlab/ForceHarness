# Fake-model program for weight_sort (force interface): the state machine of baselines/expert.py::_weight_sort written as
# one program: grasp, lift and weigh, sort at 0.375 kg into a zone.
scene = robot.scene()
find = lambda n: next(o for o in scene if o["name"] == n)
cube, table = find("cube"), find("table")
top = table["top_z"]
zones = {o["name"]: o for o in scene if o.get("kind") == "zone"}
yaw_acc = 0.0


def rot(a, period):
    """Among a + k * period, pick the angle that turns least and keeps the cumulative yaw in range."""
    global yaw_acc
    best = min((a + k * period for k in range(-4, 5)),
               key=lambda c: abs(c) + (0 if -110 <= yaw_acc + c <= 140 else 1000 + abs(yaw_acc + c)))
    yaw_acc += best
    return robot.rotate_z(best)


rot(cube.get("yaw_deg", 0.0), 90.0)
robot.move_to(cube["pos"][0], cube["pos"][1], top + 0.15)
for attempt in range(3):
    r = robot.move_until_contact([0, 0, -1], max_dist=0.2, f_threshold=3.0)
    if r["result"] == "contact" and r["end_pos"][2] > top + 0.03 and attempt < 2:
        robot.move_by(0, 0, 0.03)          # fingers landed on top of the cube: lift a little, turn 45 deg, descend again
        rot(45.0, 90.0)
        continue
    break
robot.move_by(0, 0, 0.018)
g = robot.grasp()
lift = robot.lift(height=0.10, hold_time=1.0)
m = lift["derived"].get("payload_estimate_kg")
verdict = "heavy" if (m is not None and m > 0.375) else "light"
print("grasp:", g["closed_on_something"], "payload_estimate_kg:", m, "->", verdict)
z = zones[verdict + "_zone"]
robot.move_to(z["pos"][0], z["pos"][1], top + 0.15)
robot.move_until_contact([0, 0, -1], max_dist=0.2, f_threshold=3.0)
robot.release()
robot.move_by(0, 0, 0.10)
robot.done(success=True, verdict=verdict)
