# Budget test: hold in place until the call budget (max_tool_calls) is used up; the executor must raise its stop signal
# and end the program with stop_reason=max_tool_calls.
n = 0
while True:
    robot.hold(0.05)
    n += 1
    if n % 20 == 0:
        print("calls so far:", n)
