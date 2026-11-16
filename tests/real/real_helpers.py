"""Helpers for the real-robot tests. Everything is offline: synthetic plant, worker fixture, synthetic captures.

No robot, no network, no SSH: transports are replaced by in-process or subprocess fixtures.
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[2]
ROBOT_SIDE = ROOT/'robot_side/franka_polymetis'
PROMPTS = ROOT/'forceharness/prompts/real'
FIXTURES = Path(__file__).resolve().parent/'fixtures'
CALIBRATION = FIXTURES/'force_calibration_fr3.json'     # the paper's FR3 calibration, marked verified, for tests only
FORCE_ONLY = {'move_until_contact', 'probe', 'push', 'slide', 'search', 'insert', 'lift', 'pull'}


def robot_side(name):
    """A robot-side module (guard, driver, worker, wrench, gripper, selftest), imported by its plain name as on the
    robot PC (its own imports are plain names too)."""
    if str(ROBOT_SIDE) not in sys.path:
        sys.path.insert(0, str(ROBOT_SIDE))
    return importlib.import_module(name)


def run_task(task_id, argv):
    from forceharness.real.runner import main
    from forceharness.real.tasks import TASKS
    return main(argv, task=TASKS[task_id])


def task_file(task_id):
    return PROMPTS/f'{task_id}.md'


# ---- synthetic wrist captures --------------------------------------------------------------------------
WRIST_INTRINSICS = dict(serial='synthetic', fx=390., fy=390., ppx=320., ppy=240., coeffs=[0., 0., 0., 0., 0.],
                        time_unix=0., depth_enabled=True)


def hand_eye():
    from forceharness.real.wrist_overlay import calibration_for
    return np.loadtxt(calibration_for({}))


def make_scene(folder, *, tcp=(.42, 0., .30), boxes=(), table_z=.005, width=.08, size=(640, 480), seed=0):
    """Write scene.json, scene.npz (wrist depth, metres) and scene_wrist.png for a hand pointing straight down.

    boxes: dicts with centre=(x, y), size=(length along x, width along y), height (m), optional yaw_deg.
    Depth is ray-cast against the table plane and the box tops (sides are ignored), with a little noise."""
    from PIL import Image
    from forceharness.real.checks import TCP_TRANSLATION
    from forceharness.real.config import BackendConfig
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    q_tcp = np.asarray(BackendConfig().tcp_quaternion)
    # Hand pointing down: TCP z axis = -z of the base. Flange orientation so that flange*tcp_rotation points down.
    r_tcp = R.from_matrix(np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]))
    r_flange = r_tcp*R.from_quat(q_tcp).inv()
    flange_pos = np.asarray(tcp, dtype=float)-r_flange.apply(TCP_TRANSLATION)
    T_flange = np.eye(4)
    T_flange[:3, :3] = r_flange.as_matrix()
    T_flange[:3, 3] = flange_pos
    T_cam = T_flange@hand_eye()
    w, h = size
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    rays_cam = np.stack([(uu-WRIST_INTRINSICS['ppx'])/WRIST_INTRINSICS['fx'],
                         (vv-WRIST_INTRINSICS['ppy'])/WRIST_INTRINSICS['fy'], np.ones_like(uu, dtype=float)], -1)
    rays = rays_cam@T_cam[:3, :3].T
    origin = T_cam[:3, 3]
    depth = np.full((h, w), np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_table = (table_z-origin[2])/rays[..., 2]
    depth = np.where(t_table > 0, t_table, np.nan)       # camera z = 1 along each ray, so t is the depth
    for b in boxes:
        top = table_z+b['height']
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (top-origin[2])/rays[..., 2]
        p = origin+rays*t[..., None]
        yaw = np.radians(b.get('yaw_deg', 0.))
        dx, dy = p[..., 0]-b['centre'][0], p[..., 1]-b['centre'][1]
        lx = np.cos(yaw)*dx+np.sin(yaw)*dy
        ly = -np.sin(yaw)*dx+np.cos(yaw)*dy
        inside = (np.abs(lx) <= b['size'][0]/2) & (np.abs(ly) <= b['size'][1]/2) & (t > 0)
        depth = np.where(inside & (t < np.nan_to_num(depth, nan=np.inf)), t, depth)
    rng = np.random.default_rng(seed)
    depth = depth+rng.normal(0, .0003, depth.shape)
    np.savez_compressed(folder/'scene.npz', wrist_depth=depth.astype(np.float32))
    rgb = np.full((h, w, 3), 200, np.uint8)
    Image.fromarray(rgb).save(folder/'scene_wrist.png')
    Image.fromarray(rgb).save(folder/'scene_exterior.png')
    state = dict(timestamp_ns=1, flange_pos=flange_pos.tolist(), flange_quat=r_flange.as_quat().tolist(),
                 gripper_width=width, joints=[0.]*7)
    scene = dict(state, wrist=dict(WRIST_INTRINSICS), exterior=dict(WRIST_INTRINSICS, depth_enabled=False),
                 capture_version='synthetic')
    (folder/'scene.json').write_text(json.dumps(scene))
    return folder


def photographer_from(make):
    """A live_photographer replacement: every call writes a fresh synthetic capture with make(folder)."""
    def factory(out, backend, cameras=None):
        n = iter(range(1, 1000))

        def photograph(phase, tick):
            folder = out/'photos'/f'{next(n):02d}_{phase}'
            make(folder)
            return dict(wrist_images=[str(folder/'scene_wrist.png')])
        return photograph
    return factory
