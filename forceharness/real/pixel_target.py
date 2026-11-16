"""locate: the surface point under a pixel the model selects in the latest wrist RGB-D frame (perception).

No segmentation, semantic labels, robot access or motion. The model selects a visible part (for example a
handle that has no detection box of its own); this measures that part's surface from the capture's pose and the
wrist hand-eye calibration, and refuses unreliable depth, depth edges and points on the gripper.
"""
import json
from pathlib import Path

import numpy as np

from forceharness.real.config import BackendConfig
from forceharness.real.scene_objects import HAND_BOX, NEAR
from forceharness.real.wrist_overlay import PAD_CENTRE_TCP_Z, deproject, scene_geometry


def locate(scene_dir, u, v, tcp_translation=(0., 0., PAD_CENTRE_TCP_Z), tcp_quaternion=None):
    if type(u) is not int or type(v) is not int:
        raise ValueError('u and v must be integer pixel coordinates in the original wrist image')
    folder = Path(scene_dir)
    scene = json.loads((folder/'scene.json').read_text())
    with np.load(folder/'scene.npz') as data:
        depth = data['wrist_depth'].astype(float)
    height, width = depth.shape
    if not 3 <= u < width-3 or not 3 <= v < height-3:
        raise ValueError(f'pixel outside the measurable range: the image is {width}\u00d7{height}; keep 3 pixels away from the edge')
    patch = depth[v-3:v+4, u-3:u+4]
    valid = np.isfinite(patch) & (patch > NEAR[0]) & (patch < NEAR[1])
    centre = patch[2:5, 2:5][valid[2:5, 2:5]]
    if len(centre) < 5 or valid.sum() < 25:
        raise ValueError('the selected point has no reliable local depth; select another point inside the same handle or change view')
    z = float(np.median(centre))
    lo, hi = np.percentile(patch[valid], [10, 90])
    if hi-lo > .012 or abs(float(np.median(patch[valid]))-z) > .006:
        raise ValueError('the selected point spans a depth edge and may include background; select a point inside the handle')
    T_cam, T_tcp, opening = scene_geometry(scene, tcp_translation,
                                          BackendConfig().tcp_quaternion if tcp_quaternion is None else tcp_quaternion)
    point = deproject(scene['wrist'], u, v, z)@T_cam[:3, :3].T+T_cam[:3, 3]
    local = (point-T_tcp[:3, 3])@T_tcp[:3, :3]
    if (abs(local[0]) < HAND_BOX['x'] and abs(local[1]) < opening/2+HAND_BOX['y_margin'] and
            HAND_BOX['z'][0] < local[2] < HAND_BOX['z'][1]):
        raise ValueError('the selected point is on the gripper and cannot be used as an object surface')
    return dict(pixel=[u, v], image_size=[width, height], surface_point_m=point.round(4).tolist(),
                surface_minus_tcp_mm=((point-T_tcp[:3, 3])*1000).round(1).tolist(),
                valid_pixels=int(valid.sum()), depth_spread_mm=round(float(hi-lo)*1000, 2),
                note='This is the surface reference point at the chosen pixel, not an automatic grasp pose, and the '
                     'pixel is not guaranteed to be on the handle. Coordinates use the wrist hand-eye calibration; '
                     'local depth spread is not extrinsics accuracy.')
