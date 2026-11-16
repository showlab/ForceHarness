"""Wrist-camera geometry and overlay for the tool results (perception; offline geometry, no robot access).

The model sees wrist images in which the fingertip centre is outside the frame (the D405 sits 5-7 cm in
front of the fingers), so it cannot judge a grasp from pixels alone. This module projects with the wrist
camera's hand-eye calibration (T_flange_camera, the file named by cameras.hand_eye_file in
configs/real/robot.json) and draws:

- red boxes:   the two finger pads where they are now;
- green boxes: where the pads will be after a straight vertical move that puts
  the TCP (pad centre) at ``target_z``;
- green cross: that vertical aim point; magenta cross: the measured target.

Pad geometry (measured from wrist depth and a table touch): the black finger pads
span 8-23 mm below the Desk TCP (0.1034 m), i.e. +/-7.5 mm about the pad-centre TCP
at 0.1185 m; pad centres sit 3-5 mm outside the opening on each side. deproject /
project (librealsense's inverse Brown-Conrady model) and scene_geometry are shared
with scene_objects and pixel_target.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[2]
HAND_EYE_EXAMPLE = ROOT/'configs/real/hand_eye.example.txt'
HAND_EYE_FILE = None        # set by use_hand_eye(); otherwise taken from the site configuration
DESK_TCP_Z = 0.1034
PAD_CENTRE_TCP_Z = 0.1185          # Desk TCP + 15 mm: centre of the black pads
PAD_HALF_HEIGHT = 0.0075
PAD_HALF_WIDTH = 0.010             # along the hand x axis (not measured; drawing only)
PAD_THICKNESS = 0.008
EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def project(intr, points_cam):
    """librealsense rs2_project_point_to_pixel for inverse_brown_conrady."""
    c = intr['coeffs']
    p = np.asarray(points_cam, dtype=float)
    x, y = p[..., 0]/p[..., 2], p[..., 1]/p[..., 2]
    r2 = x*x+y*y
    f = 1+c[0]*r2+c[1]*r2*r2+c[4]*r2*r2*r2
    x, y = x*f, y*f
    dx = x+2*c[2]*x*y+c[3]*(r2+2*x*x)
    dy = y+2*c[3]*x*y+c[2]*(r2+2*y*y)
    return np.stack([dx*intr['fx']+intr['ppx'], dy*intr['fy']+intr['ppy']], -1)


def deproject(intr, u, v, depth):
    """librealsense rs2_deproject_pixel_to_point for inverse_brown_conrady (iterative)."""
    c = intr['coeffs']
    xo = (np.asarray(u, dtype=float)-intr['ppx'])/intr['fx']
    yo = (np.asarray(v, dtype=float)-intr['ppy'])/intr['fy']
    x, y = xo.copy(), yo.copy()
    for _ in range(10):
        r2 = x*x+y*y
        icd = 1/(1+((c[4]*r2+c[1])*r2+c[0])*r2)
        xq, yq = x/icd, y/icd
        x = (xo-2*c[2]*xq*yq-c[3]*(r2+2*xq*xq))*icd
        y = (yo-2*c[3]*xq*yq-c[2]*(r2+2*yq*yq))*icd
    depth = np.asarray(depth, dtype=float)
    return np.stack([x*depth, y*depth, depth], -1)


def _pose(position, quaternion):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quaternion).as_matrix()
    T[:3, 3] = position
    return T


def use_hand_eye(path):
    """Use this T_flange_camera file for every capture of this process (the runner sets it at a live start)."""
    global HAND_EYE_FILE
    HAND_EYE_FILE = None if path is None else Path(path)


def calibration_for(state):
    """Hand-eye file of the wrist camera: the one set with use_hand_eye(), else cameras.hand_eye_file of the site
    configuration, else (offline only, when no site file exists) the example matrix shipped with the code."""
    if HAND_EYE_FILE is not None:
        return HAND_EYE_FILE
    from forceharness.real.config import load_robot_config
    path = load_robot_config().cameras.hand_eye_path
    return path if path.is_file() else HAND_EYE_EXAMPLE


def scene_geometry(scene, tcp_translation, tcp_quaternion, calibration=None):
    """Base-frame transforms for one saved capture (photographer scene.json)."""
    state = scene.get('robot_state', scene)
    calibration = calibration_for(state) if calibration is None else calibration
    T_base_flange = _pose(state['flange_pos'], state['flange_quat'])
    T_base_camera = T_base_flange@np.loadtxt(calibration)
    T_base_tcp = T_base_flange@_pose(tcp_translation, tcp_quaternion)
    return T_base_camera, T_base_tcp, float(state.get('gripper_width') or .08)


def pad_boxes(opening):
    """Eight corners of each pad box in the TCP frame (fingers close along TCP y)."""
    boxes = []
    for side in (1, -1):
        y0, y1 = side*opening/2, side*(opening/2+PAD_THICKNESS)
        boxes.append(np.array([[x, y, z] for x in (-PAD_HALF_WIDTH, PAD_HALF_WIDTH)
                               for y in (y0, y1) for z in (-PAD_HALF_HEIGHT, PAD_HALF_HEIGHT)]))
    return boxes


def to_pixels(intr, T_base_camera, points_base):
    T = np.linalg.inv(T_base_camera)
    pc = np.asarray(points_base, dtype=float)@T[:3, :3].T+T[:3, 3]
    return project(intr, pc), pc[..., 2]


def overlay(scene_dir, out_path, tcp_translation, tcp_quaternion, target=None, target_z=None):
    """Write an annotated copy of scene_wrist.png; return the numbers drawn."""
    scene_dir = Path(scene_dir)
    scene = json.loads((scene_dir/'scene.json').read_text())
    intr = scene['wrist']
    T_cam, T_tcp, opening = scene_geometry(scene, tcp_translation, tcp_quaternion)
    image = Image.open(scene_dir/'scene_wrist.png').convert('RGB')
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:   # Pillow without FreeType
        font = ImageFont.load_default()

    def box(corners_tcp, T, colour, width):
        pb = corners_tcp@T[:3, :3].T+T[:3, 3]
        px, depth = to_pixels(intr, T_cam, pb)
        for a, b in EDGES:
            if depth[a] > .02 and depth[b] > .02:
                draw.line([tuple(px[a]), tuple(px[b])], fill=colour, width=width)

    def cross(point, colour, size=9):
        px, depth = to_pixels(intr, T_cam, np.asarray(point, dtype=float)[None])
        if depth[0] <= .02:
            return None
        u, v = px[0]
        draw.line([(u-size, v), (u+size, v)], fill=colour, width=2)
        draw.line([(u, v-size), (u, v+size)], fill=colour, width=2)
        return [float(u), float(v)]

    tcp = T_tcp[:3, 3]
    info = dict(tcp=tcp.round(4).tolist(), opening_m=round(opening, 4))
    for corners in pad_boxes(opening):
        box(corners, T_tcp, (255, 40, 40), 2)
    if target_z is not None and tcp[2] > target_z+.002:
        T_ghost = T_tcp.copy()
        T_ghost[2, 3] = target_z
        for corners in pad_boxes(opening):
            box(corners, T_ghost, (40, 230, 40), 2)
        info['aim_px'] = cross(T_ghost[:3, 3], (40, 230, 40))
        info['aim_point'] = T_ghost[:3, 3].round(4).tolist()
    if target is not None:
        info['target_px'] = cross(target, (255, 0, 255))
        info['target_minus_tcp_mm'] = ((np.asarray(target, dtype=float)-tcp)*1000).round(1).tolist()
    lines = [f'TCP(pad centre) {np.round(tcp, 3).tolist()}  red=pads now']
    if 'aim_point' in info:
        lines.append(f'green=pads after straight descent to z={target_z:.3f}')
    if target is not None:
        dx, dy, dz = info['target_minus_tcp_mm']
        lines.append(f'magenta=measured target  target-TCP [{dx:+.0f},{dy:+.0f},{dz:+.0f}] mm')
    for i, text in enumerate(lines):
        y = image.height-18*(len(lines)-i)-4
        draw.rectangle([0, y-1, image.width, y+15], fill=(0, 0, 0))
        draw.text((4, y), text, fill=(255, 255, 255), font=font)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return info
