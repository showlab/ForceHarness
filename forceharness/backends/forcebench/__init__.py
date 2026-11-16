"""ForceBench backend: ForceHarness as a policy server for ForceBench's Isaac Lab evaluator (8 tasks).

Modules: wire (ForceBench's msgpack wire format), server + session (websocket server; one harness thread per rollout),
fbsim (the backend adapter the primitives drive), prims (ForceBench-only primitives and tool schemas), tasks (instructions and
scenes, privileged and RGB-only), episode (the agent loop on ForceBench), vision + rgb_detect (image tools and the scripted
agent's classical vision), scripted (scripted reference agent). Entry points are in scripts/forcebench/; see docs/forcebench.md.
"""
