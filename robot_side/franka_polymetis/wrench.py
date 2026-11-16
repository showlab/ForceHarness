"""Read-only wrench estimate at the flange from joint torques (layer 1: force sensing), one JSON line per sample.

Runs on the real-time computer in the Polymetis Python environment (Python 3.8). It calls only the two
read-only Polymetis RPCs GetRobotClientMetadata and GetRobotState; it never sends a controller and never
changes a parameter. driver.py imports solve_wrench and Kinematics from here, so the guard, the harness and
this standalone reader all use the same estimate.

Principle: RobotState.motor_torques_external is libfranka's tau_ext_hat_filtered, the 7 external joint
torques, from which libfranka has already subtracted the end-effector load set in the vendor's Desk tool (so a
held object appears in it). The flange wrench is w = (J^T)^+ tau_ext, where J is the 6x7 Jacobian at the flange
(panda_link8) with axes aligned to the base frame (3 linear rows, then 3 angular rows).
w = [fx, fy, fz, mx, my, mz]: force in N, moment in N*m about the flange origin, both in the base frame.
The libfranka sign is kept as it is; whether pressing on a table reads +z or -z is calibrated on the workstation
(force_sign). At start-up the frame of compute_jacobian is checked against finite differences of the forward
kinematics (world or local is detected automatically; if neither fits, the reader exits).

Environment: the Polymetis Python on the robot PC, with CONDA_PREFIX set (Polymetis reads it on import):
  CONDA_PREFIX=<polymetis env> <polymetis env>/bin/python -u wrench.py --hz 50
  CONDA_PREFIX=<polymetis env> <polymetis env>/bin/python -u wrench.py --selftest   # offline, no Polymetis server
Output: first {"type": "header"}, then {"type": "sample"} per tick; errors as {"type": "error"}.
"""
import argparse
import json
import math
import os
import sys
import tempfile
import time

import numpy as np

VERSION = "fh_wrench_reader/1"
URDF_RELATIVE = "franka_panda/panda_arm.urdf"
DEFAULT_EE_LINK = "panda_link8"
# rest_pose of conf/robot_model/franka_panda.yaml; only used to choose the poses that check the Jacobian, never sent to the robot
REST_POSE = [-0.13935, -0.02048, -0.05201, -2.06913, 0.05059, 2.00287, -0.91679]
FD_STEP = 1e-4
FRAME_TOL = 5e-3


def default_urdf():
    """The Panda URDF that ships with Polymetis, for the offline checks (live mode uses the server's URDF).

    Order: the FH_URDF environment variable, Polymetis' own data-directory helper, then the data folders next to
    the installed polymetis package. Returns None when nothing is found; pass --urdf explicitly then."""
    path = os.environ.get("FH_URDF")
    if path:
        return path
    try:
        from polymetis.utils.data_dir import get_full_path_to_urdf
        return get_full_path_to_urdf(URDF_RELATIVE)
    except Exception:
        pass
    try:
        import polymetis
        root = os.path.dirname(os.path.abspath(polymetis.__file__))
    except Exception:
        return None
    for candidate in (os.path.join(root, "data", URDF_RELATIVE),
                      os.path.join(root, "..", "..", "data", URDF_RELATIVE),
                      os.path.join(root, "..", "..", "..", "polymetis", "data", URDF_RELATIVE)):
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None


def emit(obj):
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def quat_xyzw_to_R(q):
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def solve_wrench(J, tau):
    """w = (J^T)^+ tau (least squares). Returns (w, residual norm, smallest singular value of J).

    The residual is the part of tau that no wrench at the flange explains (contact on a link, model error);
    a small smallest singular value means a near-singular configuration where w cannot be trusted."""
    J = np.asarray(J, float)
    tau = np.asarray(tau, float)
    w = np.linalg.lstsq(J.T, tau, rcond=None)[0]
    resid = float(np.linalg.norm(tau - J.T @ w))
    smin = float(np.linalg.svd(J, compute_uv=False)[-1])
    return w, resid, smin


def frame_errors(J_raw, fd_lin, R):
    """Relative error between the linear rows of compute_jacobian and the FK finite differences:
    world = compared as they are, local = rotated into the base frame first."""
    scale = max(float(np.abs(fd_lin).max()), 1e-9)
    return {
        "world": float(np.abs(J_raw[:3] - fd_lin).max()) / scale,
        "local": float(np.abs(R @ J_raw[:3] - fd_lin).max()) / scale,
    }


def pick_frame(errs, tol=FRAME_TOL):
    best = min(errs, key=errs.get)
    if errs[best] > tol:
        raise RuntimeError("cannot identify Jacobian frame: %s" % errs)
    return best


class Kinematics:
    """Polymetis' own Pinocchio robot model (the same one the controller and get_ee_pose use)."""

    def __init__(self, urdf_path, ee_link):
        import torch
        import torchcontrol as toco

        self.torch = torch
        self.model = toco.models.RobotModelPinocchio(urdf_path, ee_link)
        self.frame = None

    def _t(self, q):
        return self.torch.tensor(np.asarray(q, float), dtype=self.torch.float64)

    def fk(self, q):
        pos, quat = self.model.forward_kinematics(self._t(q))
        return pos.numpy().astype(float), quat.numpy().astype(float)

    def jacobian_raw(self, q):
        return self.model.compute_jacobian(self._t(q)).numpy().astype(float)

    def calibrate(self, poses):
        worst = {"world": 0.0, "local": 0.0}
        for q in poses:
            q = np.asarray(q, float)
            fd = np.zeros((3, len(q)))
            for i in range(len(q)):
                dq = np.zeros(len(q))
                dq[i] = FD_STEP
                fd[:, i] = (self.fk(q + dq)[0] - self.fk(q - dq)[0]) / (2 * FD_STEP)
            errs = frame_errors(self.jacobian_raw(q), fd, quat_xyzw_to_R(self.fk(q)[1]))
            worst = {k: max(worst[k], errs[k]) for k in worst}
        self.frame = pick_frame(worst)
        return self.frame, worst

    def jacobian(self, q):
        """6x7 Jacobian at the flange with axes aligned to the base frame."""
        J = self.jacobian_raw(q)
        if self.frame == "local":
            R = quat_xyzw_to_R(self.fk(q)[1])
            J = np.vstack([R @ J[:3], R @ J[3:]])
        return J


def check_poses(center):
    """The three poses used for the frame check: the centre pose and two small perturbations (computed only, never sent)."""
    c = np.asarray(center, float)
    return [c, c + np.array([0.2, -0.1, 0.15, 0.1, -0.2, 0.1, 0.3]), c + np.array([-0.15, 0.2, -0.1, -0.1, 0.25, -0.1, -0.4])]


def header(kin, source, ee_link, errs, extra):
    h = {
        "type": "header", "version": VERSION, "urdf": source, "ee_link": ee_link,
        "jacobian_frame": kin.frame, "frame_check": errs,
        "wrench": "[fx,fy,fz,mx,my,mz] base frame, N / N*m about flange origin; raw libfranka sign",
    }
    h.update(extra)
    return h


def run_selftest(args):
    if not args.urdf:
        raise RuntimeError("no Panda URDF found: pass --urdf or set FH_URDF")
    kin = Kinematics(args.urdf, args.ee_link)
    frame, errs = kin.calibrate(check_poses(REST_POSE))
    emit(header(kin, args.urdf, args.ee_link, errs, {"mode": "selftest"}))
    rng = np.random.default_rng(0)
    worst = 0.0
    for q in check_poses(REST_POSE):
        J = kin.jacobian(q)
        for w_true in [np.array([0, 0, -9.81, 0, 0, 0.0]), rng.normal(0, 5, 6)]:
            w, _, _ = solve_wrench(J, J.T @ w_true)
            worst = max(worst, float(np.abs(w - w_true).max()))
    ok = worst < 1e-6
    p, quat = kin.fk(REST_POSE)
    emit({"type": "selftest", "ok": ok, "max_wrench_err": worst, "rest_flange_pos": p.round(4).tolist(),
          "rest_flange_quat": quat.round(4).tolist()})
    return 0 if ok else 1


def run_live(args):
    import grpc
    from polymetis_pb2 import Empty
    from polymetis_pb2_grpc import PolymetisControllerServerStub

    with grpc.insecure_channel("127.0.0.1:%d" % args.port) as channel:
        stub = PolymetisControllerServerStub(channel)
        try:
            meta = stub.GetRobotClientMetadata(Empty(), timeout=2.0)
        except grpc.RpcError as exc:
            emit({"type": "error", "fatal": True, "error": "no robot client metadata (is the arm client running?): %s" % exc.details()})
            return 2
        ee_link = meta.ee_link_name or args.ee_link
        with tempfile.NamedTemporaryFile("w", suffix=".urdf") as f:
            f.write(meta.urdf_file)
            f.flush()
            kin = Kinematics(f.name, ee_link)
        center = list(meta.rest_pose) if len(meta.rest_pose) == 7 else REST_POSE
        frame, errs = kin.calibrate(check_poses(center))
        emit(header(kin, "server metadata", ee_link, errs, {
            "mode": "live", "hz": args.hz, "port": args.port, "polymetis_version": meta.polymetis_version}))

        period = 1.0 / args.hz
        t0 = time.monotonic()
        next_t = t0
        while True:
            if args.duration and time.monotonic() - t0 > args.duration:
                return 0
            if os.getppid() == 1:  # the ssh session or parent is gone: leave no orphan process on a shared robot PC
                return 0
            try:
                st = stub.GetRobotState(Empty(), timeout=0.2)
                q = list(st.joint_positions)
                tau = list(st.motor_torques_external)
                if len(q) != 7 or len(tau) != 7:
                    emit({"type": "error", "error": "state has %d joints / %d torques" % (len(q), len(tau))})
                else:
                    w, resid, smin = solve_wrench(kin.jacobian(q), tau)
                    pos, quat = kin.fk(q)
                    emit({
                        "type": "sample", "t_robot_ns": st.timestamp.seconds * 1000000000 + st.timestamp.nanos,
                        "error_code": st.error_code, "q": q, "tau_ext": tau,
                        "wrench": [round(float(v), 5) for v in w], "resid": round(resid, 5), "smin": round(smin, 5),
                        "flange_pos": [round(float(v), 6) for v in pos], "flange_quat": [round(float(v), 6) for v in quat],
                    })
            except grpc.RpcError as exc:
                emit({"type": "error", "error": "GetRobotState: %s" % exc.details()})
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--port", type=int, default=50051, help="Polymetis robot server port on this computer")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = read until stopped")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--urdf", default=None,
                    help="only for --selftest (default: the URDF that ships with Polymetis); live mode uses the server's URDF")
    ap.add_argument("--ee-link", default=DEFAULT_EE_LINK)
    args = ap.parse_args()
    if not 1.0 <= args.hz <= 200.0:
        ap.error("--hz must be in [1, 200]")
    if args.selftest and args.urdf is None:
        args.urdf = default_urdf()
    try:
        return run_selftest(args) if args.selftest else run_live(args)
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:  # failures before the header line must also reach the workstation
        emit({"type": "error", "fatal": True, "error": "%s: %s" % (type(exc).__name__, exc)})
        return 1


if __name__ == "__main__":
    try:
        code = main()
        sys.stdout.flush()
    except BrokenPipeError:
        code = 0
    os._exit(code)
