# Budget test: hold for 20 s per call until the simulated time exceeds max_sim_time (stop_reason=max_sim_time).
while True:
    r = robot.hold(20.0)
    print("held", r["result"], robot.state()["tool_pos"])
