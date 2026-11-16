"""Offline tests of the ForceBench bridge: a fake evaluator (the same msgpack websocket protocol + a toy arm) <-> PolicyServer
<-> Session <-> FBSim <-> primitives.

Toy arm: each tick the TCP moves a fraction BETA of the way to its target (first-order approximation of the joint PD). The
table is at z = TABLE; when the target is below it, the hand stays on the table and the environment pushes back with
K_ARM x (target depth below the table). The hand's own weight (0.7 kg) appears in the parent-to-child reading, and readings
are in the hand frame (hand pointing down = Rx(pi)).
Needs websockets + msgpack (pip install -e ".[forcebench]"); skipped without them.
"""
from __future__ import annotations

import socket
import threading

import numpy as np
import pytest

websockets = pytest.importorskip("websockets")
msgpack = pytest.importorskip("msgpack")

from scipy.spatial.transform import Rotation as R  # noqa: E402

from forceharness.primitives.library import Primitives  # noqa: E402
from forceharness.backends.forcebench import wire
from forceharness.backends.forcebench.fbsim import FBSim, FBTask  # noqa: E402
from forceharness.backends.forcebench.server import PolicyServer, parse_seed  # noqa: E402

TABLE = 0.10
K_ARM = 600.0          # N/m: order of magnitude of ForceBench's joint PD as seen at the hand
BETA = 0.4
M_HAND = 0.7


class ToyArm:
    def __init__(self, load_N: float = 0.0, k_sag: float = 130.0, m_hand: float = M_HAND):
        self.x = np.array([0.5, 0.0, 0.30])
        self.rpy = np.array([np.pi, 0.0, 0.0])
        self.grip = 1.0
        self.f_env = np.zeros(3)
        self.load_N, self.k_sag, self.m_hand = load_N, k_sag, m_hand   # closed gripper carries load_N; the hand sags by load_N / k_sag (no integral term)

    def apply(self, a):
        tgt = np.asarray(a[:3], dtype=float).copy()
        self.rpy = np.asarray(a[3:6], dtype=float)
        self.grip = float(a[6])
        carrying = self.load_N > 0 and self.grip < 0 and self.x[2] > TABLE + 0.005
        if carrying:
            tgt[2] -= self.load_N / self.k_sag
        nxt = self.x + BETA * (tgt - self.x)
        self.f_env = np.array([0.0, 0.0, -self.load_N]) if carrying else np.zeros(3)
        if nxt[2] < TABLE:
            nxt[2] = TABLE
        if tgt[2] < TABLE:
            self.f_env[2] = K_ARM * (TABLE - tgt[2])
        self.x = nxt

    def wrench_hand(self):
        f_pc_world = np.array([0, 0, self.m_hand * 9.81]) - self.f_env
        Rw = R.from_euler("xyz", self.rpy).as_matrix()
        return np.concatenate([Rw.T @ f_pc_world, np.zeros(3)])

    def request(self, step, rid):
        img = np.zeros((8, 8, 3), np.uint8)
        return {"contract": "forcebench_policy", "step": step, "rollout_id": rid, "control_rate_hz": 15.0,
                "observation": {"state": np.array([*self.x, *self.rpy, self.grip], np.float32),
                                "images": {"main": img, "wrist": img}, "eef_wrench": self.wrench_hand().astype(np.float32),
                                "prompt": "toy"},
                "privileged": {"table_top_z": TABLE}}


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _scene(sim):
    return [{"name": "table", "kind": "surface", "top_z": round(float(sim.privileged.get("table_top_z", 0.0)), 4)}]


def test_seed_parse():
    assert parse_seed("weight_sort:seed=307") == 307


def test_bridge_contact_push_lift():
    out = {}

    def make_worker(rid, req):
        def worker(sess):
            sim = FBSim(sess, FBTask(name="toy", instruction="toy", describe_scene=_scene), seed=parse_seed(rid) or 0,
                        stiffness_gain={150.0: 1.0, 40.0: 1.0})
            prims = Primitives(sim, interface="force")
            r1 = prims.call("move_until_contact", direction=[0, 0, -1], max_dist=0.3, f_threshold=2.0)
            r2 = prims.call("push", direction=[0, 0, -1], target_force=5.0, duration=3.0)
            r3 = prims.call("move_by", dx=0.0, dy=0.0, dz=0.05)
            out.update(r1=r1, r2=r2, r3=r3, scene=sim.scene(), steps=sim.n_steps)
            return dict(ok=True)
        return worker

    port = _free_port()
    srv = PolicyServer(make_worker, port=port)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    from websockets.sync.client import connect
    import time

    for _ in range(50):
        try:
            ws = connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None, open_timeout=2)
            break
        except OSError:
            time.sleep(0.1)
    meta = wire.loads(ws.recv())
    assert meta["contract"] == "forcebench_policy"
    arm = ToyArm()
    for step in range(900):
        ws.send(wire.dumps(arm.request(step, "toy:seed=3")))
        rep = wire.loads(ws.recv())
        a = np.asarray(rep["actions"], dtype=np.float32)
        assert a.shape == (1, 7) and np.isfinite(a).all()
        arm.apply(a[0])
        if out.get("r3") is not None and step > 5:
            break
    ws.close()
    assert out["r1"].termination == "contact", out["r1"].termination
    assert abs(out["r1"].end_pos[2] - TABLE) < 0.01
    f_end = abs(out["r2"].final_wrench_net[2])
    assert 3.0 < f_end < 8.0, out["r2"].final_wrench_net
    assert out["r3"].end_pos[2] > TABLE + 0.03
    assert out["scene"][0]["top_z"] == TABLE


def _run_toy(make_worker, arm, out, max_steps=1500, key="done"):
    from websockets.sync.client import connect
    import time

    port = _free_port()
    srv = PolicyServer(make_worker, port=port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    for _ in range(50):
        try:
            ws = connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None, open_timeout=2)
            break
        except OSError:
            time.sleep(0.1)
    wire.loads(ws.recv())
    for step in range(max_steps):
        ws.send(wire.dumps(arm.request(step, "toy:seed=5")))
        a = np.asarray(wire.loads(ws.recv())["actions"], dtype=np.float32)
        arm.apply(a[0])
        if out.get(key) is not None and step > 5:
            break
    ws.close()


def test_sag_compensation_no_ratchet():
    """Carrying 1.8 N with a 130 N/m sag (about 14 mm, more than the 12 mm lead cap of guarded moves): with sag compensation
    the carried load does not sink."""
    out = {}

    def make_worker(rid, req):
        def worker(sess):
            sim = FBSim(sess, FBTask(name="toy", instruction="toy", describe_scene=_scene), seed=5)
            prims = Primitives(sim, interface="force")
            out["grasp"] = prims.call("grasp")
            out["lift"] = prims.call("lift", height=0.05, hold_time=1.0)
            out["move"] = prims.call("move_to", x=0.62, y=0.10, z=0.30)
            out["sag"] = sim.sag_comp
            out["done"] = True
        return worker

    arm = ToyArm(load_N=1.8, k_sag=130.0, m_hand=0.0)
    _run_toy(make_worker, arm, out)
    end = np.asarray(out["move"].end_pos)
    assert out["move"].termination == "completed", out["move"].termination
    assert abs(end[2] - 0.30) < 0.006, end
    assert 0.005 < out["sag"] < 0.03, out["sag"]
