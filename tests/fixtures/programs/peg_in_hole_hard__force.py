# Fake-model program for peg_in_hole_hard (force interface), following baselines/expert.py::_peg: touch the hole block
# 3 cm above the hole, spiral search sliding under pressure, compliant insertion (on a jam back off 1 cm and insert again).
scene = robot.scene()
hole = next(o for o in scene if o["name"] == "hole")
hx, hy, hz = hole["pos"]
z_in = hz + 0.04 - 0.02                    # tool-centre height with the peg 2 cm in (the peg tip is 4 cm below the tool centre)
robot.move_to(hx, hy, hz + 0.04 + 0.03)
last = robot.move_until_contact([0, 0, -1], max_dist=0.08, f_threshold=3.0)
print("touch:", last["result"], last["end_pos"])
if last["result"] == "contact" and last["end_pos"][2] > z_in + 0.005:
    last = robot.search(normal=[0, 0, -1], radius_max=0.05, pitch=0.004, drop=0.012, mode="slide", normal_force=4.0, speed=0.02)
    print("search:", last["result"], last["derived"].get("advance_along_normal_m"))
    if last["result"] in ("dropped", "stuck"):
        last = robot.insert(depth=0.02, max_force=12.0)
        if last["result"] == "jammed":
            robot.move_by(0, 0, 0.01)
            last = robot.insert(depth=0.025, max_force=15.0)
        if last["result"] not in ("seated", "stalled") and last["end_pos"][2] > z_in + 0.003:
            last = robot.insert(depth=0.02, max_force=14.0)
        print("insert:", last["result"], last["end_pos"])
inserted = last["end_pos"][2] <= z_in + 0.005
robot.release()
robot.move_by(0, 0, 0.08)
robot.done(success=bool(inserted))
