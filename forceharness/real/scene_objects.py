"""Objects on the table from one wrist RGB-D capture (perception for the observations; no robot access).

The model gets no task-specific measurement, so every turn the wrist photo is turned into a plain list of
what stands or lies on the table, in base coordinates, and the same objects are drawn on the wrist image with
their numbers. Fixed geometry only, no learned detector and no semantics: fit the table plane, group the points
about 1 cm or more above it, and measure each group. The point cloud is the calibrated one of wrist_overlay
(hand-eye calibration, pad-centre TCP); the numbers are approximate (stereo depth, 2/4 mm grids, the side of an
object facing away from the camera is not seen) and are labelled as such for the model.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from forceharness.real import receipt_text as T
from forceharness.real.config import BackendConfig
from forceharness.real.wrist_overlay import PAD_CENTRE_TCP_Z, deproject, overlay, scene_geometry, to_pixels

GRID = .004              # m, occupancy cells in the table plane
MIN_HEIGHT = .010        # m above the table
MAX_HEIGHT = .35
MIN_AREA = 6e-4          # m^2 of occupied cells (about 2.5 x 2.5 cm)
FINE_GRID = .002        # preserve narrow objects that the 4 mm opening would erase
FINE_MIN_AREA = 4.8e-5  # 48 mm^2; a thin peg occupies about 288 mm^2
FINE_MIN_POINTS = 30
FINE_MIN_HEIGHT = .006  # at least 6 mm AND three times the measured table residual
NEAR = (.07, .60)        # m, D405 depth range used
REGION = dict(x=(.12, .90), y=(-.50, .50))
# The fingers sit in the top corners of every wrist frame (camera 7 cm in front of them, 8 cm
# above the pads): points in this TCP-frame box are the hand, not the scene.
HAND_BOX = dict(x=.045, y_margin=.035, z=(-.14, .02))
# Stereo depth on a uniform white table makes 2-4 cm high blobs (three of them were seen next to one
# bottle). A low blob is kept only when most of the image ring around it is well measured table;
# tall objects and objects in close-up views (no table around them) always are.
LOW_TOP = .045
RING_TABLE_MIN = .5
COLOUR = (255, 200, 0)
HAND_COLOUR = (0, 230, 255)
# Grasp angle: the model rarely turned the hand by itself before closing, so the list says how far to turn.
# The fingers close along the hand y axis, so a long object is held across its narrow side.
GRIPPER_SPAN_M = .075      # widest part the open fingers can straddle (80 mm opening)
ROUND_RATIO = 1.25         # top-view length/width below this: no preferred closing direction
ANGLE_MIN_LENGTH = .03     # smaller footprints have no reliable direction (oblique views stretch them)
ALIGNED_DEG = 10.


def table_plane(points):
    """Plane of the dominant horizontal surface: (point, unit normal, rms, inliers).

    Starts from the most populated height of the lowest 8 cm, so a band of noisy white table
    does not tilt the fit (a lowest-layer fit once put a mat 1.6 cm above the plane and merged
    it with the object standing on it)."""
    z = points[:, 2]
    z5 = float(np.percentile(z, 5))
    low = points[z < z5+.08]
    hist, edges = np.histogram(low[:, 2], bins=np.arange(z5, z5+.0805, .002))
    mode = float(edges[hist.argmax()]+.001)
    keep = np.abs(low[:, 2]-mode) < .006
    for _ in range(6):
        centre = low[keep].mean(0)
        normal = np.linalg.svd(low[keep]-centre, full_matrices=False)[2][-1]
        normal *= np.sign(normal[2])
        residual = np.abs((low-centre)@normal)
        keep = residual < max(.0015, 3*np.median(residual[keep]))
    return centre, normal, float(np.median(residual[keep])), int(keep.sum())


def fold_deg(angle):
    """A closing direction is a line: fold any angle into [-90, 90)."""
    return ((float(angle)+90.) % 180.)-90.


def closing_deg(T_tcp):
    """Direction the fingers close along in the table plane (deg from +x, CCW), None when the hand lies on its side."""
    c = T_tcp[:3, 1]
    if math.hypot(c[0], c[1]) < .5:
        return None
    return fold_deg(math.degrees(math.atan2(c[1], c[0])))


def grasp_angle(obj, closing):
    """The rotate_z that makes the fingers straddle the object's narrow side (the gripper is symmetric,
    so +-90 deg covers every case). Geometry only; the runner adds whether the arm can turn that far."""
    out = dict(closing_now_deg=None if closing is None else round(closing, 1))
    if obj['width'] > GRIPPER_SPAN_M:
        return dict(out, kind='too_wide')
    if obj['length'] < ANGLE_MIN_LENGTH or obj['length'] < ROUND_RATIO*obj['width']:
        return dict(out, kind='any')
    target = fold_deg(obj['axis_deg']+90.)
    out.update(kind='narrow_side', target_closing_deg=round(target, 1))
    if closing is not None:
        out['rotate_z_deg'] = round(fold_deg(target-closing), 1)
    return out


def _direction_word(v):
    """Image direction (u right, v down) as one of eight words."""
    words = ('right', 'down-right', 'down', 'down-left', 'left', 'up-left', 'up', 'up-right')
    return words[int(round(math.degrees(math.atan2(v[1], v[0]))/45.)) % 8]


def image_axes(intr, T_cam, table_z):
    """Pixel directions of base +x and +y on the table at the image centre (None if the camera looks sideways)."""
    ray = T_cam[:3, :3]@deproject(intr, np.array([intr['ppx']]), np.array([intr['ppy']]), np.array([1.]))[0]
    if ray[2] > -.3:
        return None
    p0 = T_cam[:3, 3]+(table_z-T_cam[2, 3])/ray[2]*ray
    px, _ = to_pixels(intr, T_cam, np.array([p0, p0+[.05, 0, 0], p0+[0, .05, 0]]))
    vx, vy = px[1]-px[0], px[2]-px[0]
    return dict(x=np.round(vx, 1).tolist(), y=np.round(vy, 1).tolist(),
                x_word=_direction_word(vx), y_word=_direction_word(vy))


def _clusters(points, heights, *, grid_size, min_height, min_area, min_points, padding=3):
    """Point indices for connected footprints; padding preserves objects at the view edge."""
    indices = np.flatnonzero(heights > min_height)
    if len(indices) < min_points:
        return []
    xy = points[indices, :2]
    cell = np.floor((xy-xy.min(0))/grid_size).astype(int)+padding
    grid = np.zeros(cell.max(0)+padding+1, bool)
    grid[cell[:, 0], cell[:, 1]] = True
    grid = ndimage.binary_closing(ndimage.binary_opening(grid, iterations=1), iterations=2)
    labels, n = ndimage.label(grid, structure=np.ones((3, 3)))
    area = np.bincount(labels.ravel(), minlength=n+1)*grid_size**2
    lab = labels[cell[:, 0], cell[:, 1]]
    counts = np.bincount(lab, minlength=n+1)
    return [indices[lab == k] for k in range(1, n+1)
            if area[k] >= min_area and counts[k] >= min_points]


def detect(scene_dir, tcp_translation=(0., 0., PAD_CENTRE_TCP_Z), tcp_quaternion=None):
    """dict(table_z, view, objects=[...]) for one capture folder (scene.json + scene.npz)."""
    scene_dir = Path(scene_dir)
    scene = json.loads((scene_dir/'scene.json').read_text())
    depth = np.load(scene_dir/'scene.npz')['wrist_depth'].astype(float)
    if tcp_quaternion is None:
        tcp_quaternion = BackendConfig().tcp_quaternion
    T_cam, T_tcp, opening = scene_geometry(scene, tcp_translation, tcp_quaternion)
    valid = (depth > NEAR[0]) & (depth < NEAR[1])
    vv, uu = np.nonzero(valid)
    if len(vv) < 2000:
        raise ValueError('wrist depth has too few valid pixels (hand too close to something?)')
    pts = deproject(scene['wrist'], uu, vv, depth[vv, uu])@T_cam[:3, :3].T+T_cam[:3, 3]
    T_inv = np.linalg.inv(T_tcp)
    local = pts@T_inv[:3, :3].T+T_inv[:3, 3]
    hand = ((np.abs(local[:, 0]) < HAND_BOX['x']) & (np.abs(local[:, 1]) < opening/2+HAND_BOX['y_margin']) &
            (local[:, 2] > HAND_BOX['z'][0]) & (local[:, 2] < HAND_BOX['z'][1]))
    pts, uu, vv = pts[~hand], uu[~hand], vv[~hand]
    origin, normal, rms, inliers = table_plane(pts)
    height = (pts-origin)@normal

    def table_z(xy):
        return float(origin[2]-(normal[:2]@(np.asarray(xy)-origin[:2]))/normal[2])

    on_table = np.abs(height) < .005
    table_px = np.zeros_like(valid)
    table_px[vv[on_table], uu[on_table]] = True
    view = dict(x=np.percentile(pts[on_table, 0], [2, 98]).round(3).tolist() if on_table.sum() > 100 else None,
                y=np.percentile(pts[on_table, 1], [2, 98]).round(3).tolist() if on_table.sum() > 100 else None)
    fine_height = max(FINE_MIN_HEIGHT, 3*rms)
    roi = ((height > min(MIN_HEIGHT, fine_height)) & (height < MAX_HEIGHT) &
           (pts[:, 0] > REGION['x'][0]) & (pts[:, 0] < REGION['x'][1]) &
           (pts[:, 1] > REGION['y'][0]) & (pts[:, 1] < REGION['y'][1]))
    closing = closing_deg(T_tcp)
    out = dict(version='scene_objects/angle1',
               table_z=round(table_z(T_tcp[:2, 3]), 4), table_fit_rms_m=round(rms, 4), view=view, objects=[], candidates=[],
               fine_min_height_m=round(fine_height, 4),
               tcp=T_tcp[:3, 3].round(4).tolist(), gripper_opening_m=round(opening, 4), dropped_low_blobs=0,
               closing_deg=None if closing is None else round(closing, 1),
               image_axes=image_axes(scene['wrist'], T_cam, table_z(T_tcp[:2, 3])))
    if roi.sum() < FINE_MIN_POINTS:
        return out
    P, H, U, V = pts[roi], height[roi], uu[roi], vv[roi]
    tcp = T_tcp[:3, 3]

    def measure(indices):
        p, h, u, v = P[indices], H[indices], U[indices], V[indices]
        top_h = float(np.percentile(h, 95))
        mean = p[:, :2].mean(0)
        axis = np.linalg.svd(p[:, :2]-mean, full_matrices=False)[2][0]
        axis *= 1 if axis[0] >= 0 else -1                 # long axis reported pointing +x
        perp = np.array([-axis[1], axis[0]])
        s, t = (p[:, :2]-mean)@axis, (p[:, :2]-mean)@perp
        s_lo, s_hi = np.percentile(s, [2, 98])
        t_lo, t_hi = np.percentile(t, [2, 98])
        centre = mean+(s_lo+s_hi)/2*axis+(t_lo+t_hi)/2*perp
        z0 = table_z(centre)
        length, width = float(s_hi-s_lo), float(t_hi-t_lo)
        inside = (abs((tcp[:2]-centre)@axis) < length/2+.01 and abs((tcp[:2]-centre)@perp) < width/2+.01)
        under_hand = bool(inside and tcp[2] < z0+top_h+.03)
        if top_h < LOW_TOP and not under_hand:
            mask = np.zeros_like(valid)
            mask[v, u] = True
            mask = ndimage.binary_closing(mask, iterations=3)
            ring = ndimage.binary_dilation(mask, iterations=12) & ~ndimage.binary_dilation(mask, iterations=3)
            if not ring.any() or table_px[ring].mean() < RING_TABLE_MIN:
                out['dropped_low_blobs'] += 1
                return None
        top = h > top_h-.012                               # the highest layer (cap, ridge, lid)
        top_xy = p[top, :2].mean(0)
        border = bool(((u < 4) | (u > depth.shape[1]-5) | (v < 4) | (v > depth.shape[0]-5)).any())
        top_point = np.r_[top_xy, z0+top_h]
        return dict(
            centre=[round(float(centre[0]), 3), round(float(centre[1]), 3), round(z0+top_h/2, 3)],
            top=[round(float(top_xy[0]), 3), round(float(top_xy[1]), 3), round(z0+top_h, 3)],
            height=round(top_h, 3), length=round(length, 3), width=round(width, 3),
            axis=[round(float(axis[0]), 3), round(float(axis[1]), 3)],
            axis_deg=round(math.degrees(math.atan2(axis[1], axis[0])), 1),
            points=int(len(indices)), partial=border, under_hand=under_hand,
            top_minus_tcp_mm=np.round((top_point-tcp)*1000, 1).tolist())

    claimed = np.zeros(len(P), bool)
    for fine, kwargs in ((False, dict(grid_size=GRID, min_height=MIN_HEIGHT, min_area=MIN_AREA, min_points=80, padding=0)),
                         (True, dict(grid_size=FINE_GRID, min_height=fine_height,
                                     min_area=FINE_MIN_AREA, min_points=FINE_MIN_POINTS))):
        for indices in _clusters(P, H, **kwargs):
            # One physical component must not become two objects simply because both grids saw it.
            if fine and claimed[indices].mean() > .2:
                continue
            obj = measure(indices)
            if obj is None:
                continue
            claimed[indices] = True
            uncertain = fine and obj['height'] < LOW_TOP
            obj['detection'] = 'candidate' if uncertain else 'object'
            if uncertain:
                obj['note'] = 'low or small geometric candidate, possibly depth noise; confirm with RGB or another view first'
            out['candidates' if uncertain else 'objects'].append(obj)
    # Confident geometry first, uncertain additions separately; each group is sorted left to right.
    numbered = []
    for key in ('objects', 'candidates'):
        out[key].sort(key=lambda o: -o['centre'][1])
        numbered.extend(out[key])
    for i, obj in enumerate(numbered, 1):
        obj['id'] = i
    for obj in out['objects']:
        obj['grasp'] = grasp_angle(obj, closing)
    return out


def _grasp_text(o):
    g = o.get('grasp')
    if not g:
        return None
    if g['kind'] == 'too_wide':
        return (T.GRASP_ANGLE+f"width from above {o['width']*1000:.0f} mm exceeds the {GRIPPER_SPAN_M*1000:.0f} mm the "
                'fingers can span, so it cannot be grasped whole; only a local part (edge, handle) can be, and its '
                'direction is yours to judge from the photo and the long axis.')
    if g['kind'] == 'any':
        return T.GRASP_ANGLE+'nearly round or very small from above; the closing direction needs no rotation.'
    text = (T.GRASP_ANGLE+f"grip across its narrow side (width {o['width']*1000:.0f} mm); the fingers should close "
            f"along {g['target_closing_deg']:+.0f}°")
    if g.get('rotate_z_deg') is not None:
        turn = g['rotate_z_deg']
        text += f", now {g['closing_now_deg']:+.0f}°"
        text += (f" → aligned (off by {turn:+.0f}°), no rotation needed" if abs(turn) <= ALIGNED_DEG
                 else f" → first rotate_z({turn:+.0f})")
        reach = g.get('reach')
        if reach and abs(turn) > ALIGNED_DEG:
            if reach.get('ok'):
                text += ' (estimated from the current hand pose, the joints can turn that far)'
            elif reach.get('alternative_ok'):
                text += (f" (in this direction the joints allow at most about {reach['max_deg']:.0f}°; turning "
                         f"{reach['alternative_deg']:+.0f}° the other way is equivalent; split turns above 90° into two)")
            else:
                text += (f" (from the current hand pose the joints allow at most about {reach['max_deg']:.0f}°; "
                         'turn from another hand pose)')
    if o.get('partial'):
        text += '; it is partly out of frame, so the direction may be off, confirm from another view'
    return text+'.'


def describe(found, *, synthetic=False):
    """Plain text for the model: one line per object (and candidate), then notes, image axes and the table."""
    if found is None:
        return ''
    head = (T.SCENE_OBJECTS+' (synthetic demo: a synthetic scene, not a camera measurement)' if synthetic else
            T.SCENE_OBJECTS+' (detected automatically by the wrist depth camera, base frame, meters; about '
            '±1–2 cm; the side facing away from the camera is not seen)')
    lines = [head+':']
    for o in found['objects']+found.get('candidates', []):
        notes = []
        if o.get('partial'):
            notes.append('partly out of frame, size underestimated')
        if o.get('under_hand'):
            notes.append('right under or in the hand')
        if o.get('label'):
            notes.append(o['label'])
        if o.get('note'):
            notes.append(o['note'])
        lines.append(f"#{o['id']} center {o['centre'][:2]}, highest point {o['top']} ({o['height']:.3f} above the table), "
                     f"from above length {o['length']:.3f} × width {o['width']:.3f}, long axis {o['axis']} "
                     f"({o['axis_deg']:+.0f}°)"
                     + ('; '+', '.join(notes) if notes else ''))
        if 'top_minus_tcp_mm' in o:
            lines.append(T.TOP_OFFSET+f" [dx, dy, dz] = {o['top_minus_tcp_mm']} mm (surface reference for locating, "
                         'not an automatic grasp point).')
        grasp = _grasp_text(o)
        if grasp:
            lines.append(grasp)
    if not found['objects'] and not found.get('candidates'):
        lines.append(T.NO_OBJECTS+', which does not mean nothing is in the image; small, flat or depth-less objects '
                     'can still be missed.)')
    lines.append(T.NO_HOLES+'; for parts visible in RGB but without their own coordinates, use locate to read the '
                 'surface depth at that pixel, and change view if depth is missing. The center of a hole may read the '
                 'bottom of the hole, not its rim or a grasp pose; do not guess coordinates from pixels alone.')
    axes = found.get('image_axes')
    if axes:
        line = (T.WRIST_AXES+f"+x (away from the base) points {axes['x_word']} in the image, +y (the robot's left) "
                f"points {axes['y_word']} (arrows at the lower left); cyan circle = the table point directly below the "
                'finger-pad center, cyan line = the direction the fingers close now')
        if found.get('closing_deg') is not None:
            line += f" ({found['closing_deg']:+.0f}°, counterclockwise from +x)"
        lines.append(line+'.')
    view = found.get('view') or {}
    tail = T.TABLE_Z+f"{found['table_z']:.3f}"
    if view.get('x') and view.get('y'):
        tail += (f"; the table range seen in this wrist photo is x {view['x'][0]:.2f}–{view['x'][1]:.2f}, "
                 f"y {view['y'][0]:.2f}–{view['y'][1]:.2f}")
    lines.append(tail+'.')
    return '\n'.join(lines)


def annotate(scene_dir, out_path, found, tcp_translation, tcp_quaternion):
    """The pad overlay of wrist_overlay plus a yellow box and number for every detected object."""
    scene_dir = Path(scene_dir)
    marks = overlay(scene_dir, out_path, tcp_translation, tcp_quaternion)
    scene = json.loads((scene_dir/'scene.json').read_text())
    T_cam, T_tcp, opening = scene_geometry(scene, tcp_translation, tcp_quaternion)
    intr = scene['wrist']
    image = Image.open(out_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()

    def segment(a, b, colour, width=2):
        px, d = to_pixels(intr, T_cam, np.array([a, b]))
        if (d > .02).all():
            draw.line([tuple(px[0]), tuple(px[1])], fill=colour, width=width)

    if found is not None:
        # where the pad centre would land on the table, and the line the fingers close along
        tcp, z = T_tcp[:3, 3], found['table_z']
        under = np.array([tcp[0], tcp[1], z])
        c = T_tcp[:3, 1]*[1, 1, 0]
        if np.linalg.norm(c) > .5:
            c = c/np.linalg.norm(c)*max(opening, .02)/2
            segment(under-c, under+c, HAND_COLOUR, 3)
        px, d = to_pixels(intr, T_cam, under[None])
        if d[0] > .02:
            u, v = px[0]
            draw.ellipse([u-6, v-6, u+6, v+6], outline=HAND_COLOUR, width=2)
        for h0 in np.arange(z+.01, tcp[2], .02):              # dashed plumb line from the pads down
            segment([tcp[0], tcp[1], h0], [tcp[0], tcp[1], min(h0+.01, tcp[2])], HAND_COLOUR, 1)
        axes = found.get('image_axes')
        if axes:
            o = np.array([46., image.height-70.])
            for key, name in (('x', '+x'), ('y', '+y')):
                v = np.asarray(axes[key], dtype=float)
                tip = o+v/max(float(np.linalg.norm(v)), 1e-9)*36
                draw.line([tuple(o), tuple(tip)], fill=HAND_COLOUR, width=3)
                draw.ellipse([tip[0]-3, tip[1]-3, tip[0]+3, tip[1]+3], fill=HAND_COLOUR)
                draw.text((tip[0]+4, tip[1]-8), name, fill=HAND_COLOUR, font=font)
    drawn = 0
    items = (found or {}).get('objects', [])+(found or {}).get('candidates', [])
    for o in items:
        colour = (255, 140, 40) if o.get('detection') == 'candidate' else COLOUR
        c, a = np.array(o['centre'][:2]), np.array(o['axis'])
        p = np.array([-a[1], a[0]])
        z0 = o['top'][2]-o['height']
        foot = [c+sa*o['length']/2*a+sp*o['width']/2*p for sa, sp in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        corners = np.array([[*xy, z] for z in (z0, z0+o['height']) for xy in foot])
        px, d = to_pixels(scene['wrist'], T_cam, corners)
        if (d <= .02).any():
            continue
        for i in range(4):
            j = (i+1) % 4
            draw.line([tuple(px[i]), tuple(px[j])], fill=colour, width=1)          # footprint on the table
            draw.line([tuple(px[4+i]), tuple(px[4+j])], fill=colour, width=2)      # top outline
            draw.line([tuple(px[i]), tuple(px[4+i])], fill=colour, width=1)
        label, depth = to_pixels(scene['wrist'], T_cam, np.array([o['top']]))
        if depth[0] > .02:
            u, v = label[0]
            text = f"#{o['id']}"+('?' if o.get('detection') == 'candidate' else '')
            draw.rectangle([u-2, v-20, u+8*len(text)+4, v-2], fill=(0, 0, 0))
            draw.text((u+1, v-20), text, fill=colour, font=font)
            draw.line([(u-4, v), (u+4, v)], fill=colour, width=2)
            draw.line([(u, v-4), (u, v+4)], fill=colour, width=2)
        if (o.get('grasp') or {}).get('kind') == 'narrow_side':
            # the closing line that straddles the narrow side, at the top
            t = math.radians(o['grasp']['target_closing_deg'])
            half = np.array([math.cos(t), math.sin(t), 0.])*(o['width']/2+.01)
            segment(np.array(o['top'])-half, np.array(o['top'])+half, colour, 3)
        drawn += 1
    if found is not None:
        y = 4
        draw.rectangle([0, y-1, image.width, y+15], fill=(0, 0, 0))
        draw.text((4, y), 'yellow = objects (thick line = closing across the narrow side); orange ? = uncertain; '
                  'cyan = under the pads / closing now',
                  fill=(255, 255, 255), font=ImageFont.load_default())
    image.save(out_path)
    return dict(marks, objects_drawn=drawn)
