"""Perception stage of the RGB-only scripted agent, one function per task (classical vision; part of the scripted baseline).

perceive_<task>(agent, sim, k) looks at the current images on its k-th call and returns (next tool calls, None), or
(None, scene) once it is done. The scene has the format of the privileged describe_scene in tasks.py (name, pos, grasp_point,
pivot, ...), so the action sequences of the privileged scripted agent in scripted.py are reused unchanged.

Inputs are only the two RGB images, the camera calibration and the dimensions / heights stated in the instructions; no ground
truth is read (it is only logged through agent._log, to measure localisation error). Thresholds were set on images that a
development probe recorded on seeds 300-307. How each object is found:
- parcel (CoM Sort): tan blob -> projection fit of the known box shape (the oblique wrist view includes a side face, which
  shifts the blob centroid by 7-14 mm; the fit brings the error to 1-4 mm).
- black basket: centroid of the dark blob on the bright background (max over channels < 75), intersected at 5 cm height (the
  main view cuts off part of the basket: about 2 cm of error, against a tolerance of +-10 / +-5.8 cm).
- conveyor: the left edge of the blue blob is the near end (smallest y), its top and bottom edges are the belt sides; the drop
  point is 10 cm in from the near end, midway between the sides.
- faucet rod: light-grey metal blob inside the sink (the largest dark blob, holes filled). Main view: the top of its leftmost
  part. In the wrist view the rod top points down (the wrist camera looks back obliquely, so higher points appear lower in the
  image), so the lowest edge of the metal blob is used there (logged only).
- stove knob: the largest bright ellipse on the cooktop, refined by a cylinder fit in the wrist view.
- drawer handle: the three brass handles sorted top to bottom; the middle one, intersected at its known height.
- nut and bolt: the two dark-grey blobs on the wooden table (the nearer one is the nut), refined by cylinder fits in the wrist view.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from forceharness.backends.forcebench.tasks import (DR_HANDLE_Z, FT_ROD_TOP_ABOVE_PIVOT, NT_NUT_BOTTOM_Z, NT_NUT_TOP_Z, NT_SHANK_TOP_Z, NT_TARGET_BOTTOM_Z,
                    RS_BASKET_FLOOR_Z, RS_BELT_Z, RS_TOP_Z, ST_KNOB_BASE_Z, ST_KNOB_TOP_Z)
from forceharness.backends.forcebench.vision import box_outline, color_mask, component_near, cylinder_outline, dark_mask, fit_projection


def _fail(ag, what: str):
    ag._log(what, None, "not_found")
    return [ag._tc("done", success=False, verdict="", note=f"scripted rgb: could not find {what}")], None


def _centroid(mask: np.ndarray) -> Tuple[float, float]:
    vs, us = np.nonzero(mask)
    return float(us.mean()), float(vs.mean())


def metal_mask(img: np.ndarray, lo: int = 90, sat: int = 40, blue: int = 22) -> np.ndarray:
    """Light-grey / silver metal: no channel dark and all channels close; B - R <= 22 excludes the light-blue cup (cup B - R
    ~ 30-39, valve 13-18)."""
    im = np.asarray(img, dtype=int)
    return (im.min(axis=2) >= lo) & (im.max(axis=2) - im.min(axis=2) <= sat) & (im[..., 2] - im[..., 0] <= blue)


def _lowest_component(mask: np.ndarray, min_area: int = 15) -> Optional[np.ndarray]:
    """Connected component whose lower edge is lowest (largest v): the valve pipe runs along the sink's near edge, while the
    bright line of the sink rim is at the top."""
    from scipy import ndimage

    lab, n = ndimage.label(mask)
    best, best_v = None, -1
    for i in range(1, n + 1):
        vs, _ = np.nonzero(lab == i)
        if len(vs) >= min_area and vs.max() > best_v:
            best, best_v = i, vs.max()
    return None if best is None else (lab == best)


def _filled(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    return ndimage.binary_fill_holes(mask)


def _hull(mask: np.ndarray, shrink: int = 2) -> np.ndarray:
    """Convex hull of a component, eroded by `shrink` pixels: a dark region cut apart by the cup and the valve, such as the
    sink, cannot be closed by filling holes, so its hull bounds it."""
    from scipy import ndimage

    from forceharness.backends.forcebench.vision import _hull_mask

    vs, us = np.nonzero(mask)
    h = _hull_mask(mask.shape, np.c_[us, vs].astype(float))
    if h is None:
        return mask
    return ndimage.binary_erosion(h, iterations=shrink) if shrink else h


# ------------------------------------------------------------------ rattle_sort
RS_BOX = box_outline((0.120, 0.054, RS_TOP_Z), 0.0)


def perceive_rattle_sort(ag, sim, k: int):
    if k == 0:
        main = sim.images_raw["main"]
        m = component_near(color_mask(main, "tan"), min_area=60)
        if m is None:
            return _fail(ag, "the parcel")
        q = sim.locate_on_plane("main", *_centroid(m), RS_TOP_Z / 2)
        xy, _ = fit_projection(sim, "main", m, RS_BOX, q[:2])
        ag.est["parcel_main"] = xy
        ag._log("parcel", [xy[0], xy[1], RS_TOP_Z / 2], "main_fit")
        d = component_near(dark_mask(main), min_area=500)
        if d is None:
            return _fail(ag, "the black basket")
        b = sim.locate_on_plane("main", *_centroid(d), 0.05)
        ag.est["basket"] = b
        ag._log("black_basket", b, "main_centroid_z0.05")
        bl = component_near(color_mask(main, "blue"), min_area=500)
        if bl is None:
            return _fail(ag, "the conveyor")
        vs, us = np.nonzero(bl)
        v_mid = 0.5 * (vs.min() + vs.max())
        row = us[np.abs(vs - v_mid) <= 1]
        near = sim.locate_on_plane("main", float(row.min()), v_mid, RS_BELT_Z)          # near end of the belt (left edge)
        top = sim.locate_on_plane("main", float(us[vs == vs.min()].mean()), float(vs.min()), RS_BELT_Z)
        bot = sim.locate_on_plane("main", float(us[vs == vs.max()].mean()), float(vs.max()), RS_BELT_Z)
        c = np.array([0.5 * (top[0] + bot[0]), near[1] + 0.10, RS_BELT_Z])
        ag.est["conveyor"] = c
        ag._log("conveyor", c, "main_blue_near_end")
        return [ag._tc("move_to", x=float(xy[0]) + 0.07, y=float(xy[1]), z=RS_TOP_Z + 0.15)], None
    wrist = sim.images_raw["wrist"]
    p0 = ag.est["parcel_main"]
    m = component_near(color_mask(wrist, "tan"), sim.project("wrist", [p0[0], p0[1], RS_TOP_Z]), min_area=200)
    xy = p0
    if m is not None:
        xy, _ = fit_projection(sim, "wrist", m, RS_BOX, p0)
        ag._log("parcel", [xy[0], xy[1], RS_TOP_Z / 2], "wrist_fit")
    b, c = ag.est["basket"], ag.est["conveyor"]
    scene = [dict(name="parcel", kind="object", pos=[float(xy[0]), float(xy[1]), RS_TOP_Z / 2], yaw_deg=0.0),
             dict(name="black_basket", kind="container", pos=[float(b[0]), float(b[1]), RS_BASKET_FLOOR_Z]),
             dict(name="conveyor", kind="zone", pos=[float(c[0]), float(c[1]), RS_BELT_Z]),
             dict(name="table", kind="surface", top_z=0.0)]
    return None, scene


# ------------------------------------------------------------------ faucet_turn / faucet_force_turn
def _perceive_faucet(ag, sim, k: int):
    ex = sim.task.extra
    top_z, below = float(ex["rod_top_z"]), float(ex["grasp_below_top"])
    if k == 0:
        main = sim.images_raw["main"]
        sink = component_near(dark_mask(main), min_area=500)
        if sink is None:
            return _fail(ag, "the sink")
        mm = metal_mask(main) & _hull(sink)
        comp = _lowest_component(mm)
        if comp is None:
            return _fail(ag, "the faucet valve")
        vs, us = np.nonzero(comp)
        band = us <= us.min() + 3                       # leftmost part of the pipe = the rod
        v_top = vs[band].min()
        u_c = float(us[band & (vs <= v_top + 2)].mean())
        q = sim.locate_on_plane("main", u_c, float(v_top) + 0.5, top_z)
        ag.est["rod_main"] = q
        ag._log("rod_top", q, "main_left_end_top")
        return [ag._tc("move_to", x=float(q[0]), y=float(q[1]), z=top_z + 0.095)], None
    wrist = sim.images_raw["wrist"]
    q = ag.est["rod_main"]
    comp = component_near(metal_mask(wrist), sim.project("wrist", q), min_area=30)
    if comp is not None:                                # logged only: seen obliquely from behind, higher is lower in the image and the rod top is
        vs, us = np.nonzero(comp)                       # the blob's lowest edge; on the rounded top that point is ~4.5 mm off the axis, worse than the main view
        v_max = vs.max()
        u_c = float(us[vs >= v_max - 2].mean())
        ag._log("rod_top", sim.locate_on_plane("wrist", u_c, float(v_max) + 0.5, top_z), "wrist_bottom_edge(log only)")
    x, y = float(q[0]), float(q[1])
    scene = [dict(name="faucet_lever", kind="lever", pivot=[x, y, top_z - FT_ROD_TOP_ABOVE_PIVOT], axis=[0.0, 1.0, 0.0],
                  grasp_point=[x, y, top_z - below], rod_diameter_m=0.0104),
             dict(name="countertop", kind="surface", top_z=0.0)]
    return None, scene


def perceive_faucet_turn(ag, sim, k: int):
    return _perceive_faucet(ag, sim, k)


def perceive_faucet_force_turn(ag, sim, k: int):
    return _perceive_faucet(ag, sim, k)


# ------------------------------------------------------------------ stove_turn_on / stove_press_turn
ST_CYL = cylinder_outline(0.019, ST_KNOB_BASE_Z, ST_KNOB_TOP_Z)


def _round_blobs(mask: np.ndarray, area=(10, 4000), min_fill: float = 0.5, aspect=(0.55, 1.8)):
    from forceharness.backends.forcebench.vision import _components

    return [c for c in _components(mask) if area[0] <= c["area"] <= area[1] and c["fill"] >= min_fill
            and aspect[0] <= c["bw"] / max(1, c["bh"]) <= aspect[1]]


def _perceive_stove(ag, sim, k: int):
    above = float(sim.task.extra.get("grasp_above_base", 0.002))
    if k == 0:
        main = sim.images_raw["main"]
        cook = component_near(dark_mask(main), min_area=500)            # dark cooktop
        if cook is None:
            return _fail(ag, "the cooktop")
        # silver knob (a flat ellipse of about 17 x 10 px in the oblique main view); threshold 120 (at 150, two of eight environments kept only half the bright face)
        bright = metal_mask(main, lo=120, sat=30) & _hull(cook)
        cands = _round_blobs(bright, area=(40, 600), min_fill=0.55, aspect=(0.5, 2.2))
        if not cands:
            return _fail(ag, "the knob")
        c = max(cands, key=lambda c: c["area"])
        q = sim.locate_on_plane("main", c["u"], c["v"], 0.5 * (ST_KNOB_BASE_Z + ST_KNOB_TOP_Z))
        ag.est["knob_main"] = q
        ag._log("stove_knob", q, "main_round_bright")
        return [ag._tc("move_to", x=float(q[0]) + 0.06, y=float(q[1]), z=ST_KNOB_TOP_Z + 0.12)], None
    wrist = sim.images_raw["wrist"]
    q = ag.est["knob_main"]
    xy = np.asarray(q[:2], dtype=float)
    m = component_near(metal_mask(wrist, lo=120, sat=35), sim.project("wrist", [q[0], q[1], ST_KNOB_TOP_Z]), min_area=40)
    if m is not None:
        fit, iou = fit_projection(sim, "wrist", m, ST_CYL, xy)
        ok = iou > 0.6 and float(np.linalg.norm(fit - xy)) < 0.006      # reject if far from the main-view estimate (blob merged with a shadow / the white line)
        ag._log("stove_knob", [fit[0], fit[1], ST_KNOB_BASE_Z], f"wrist_fit iou={iou:.2f} used={ok}")
        if ok:
            xy = fit
    x, y = float(xy[0]), float(xy[1])
    scene = [dict(name="stove_knob", kind="knob", pos=[x, y, ST_KNOB_BASE_Z], diameter_m=0.038, height_m=0.0192,
                  grasp_point=[x, y, ST_KNOB_BASE_Z + above]),
             dict(name="countertop", kind="surface", top_z=0.0)]
    return None, scene


def perceive_stove_turn_on(ag, sim, k: int):
    return _perceive_stove(ag, sim, k)


def perceive_stove_press_turn(ag, sim, k: int):
    return _perceive_stove(ag, sim, k)


# ------------------------------------------------------------------ radmanso_drawer_sort_table
def brass_mask(img: np.ndarray) -> np.ndarray:
    """Brass handle (~ 180, 155, 105): brighter and yellower (large G - B) than the walnut (~ 110-170, 80-130, 45-90)."""
    im = np.asarray(img, dtype=int)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    return (R > 150) & (G > 125) & (B < 150) & (G - B > 40) & (R - G < 70)


def _handles(img: np.ndarray):
    from forceharness.backends.forcebench.vision import _components

    return [c for c in _components(brass_mask(img)) if 30 <= c["area"] <= 400 and c["bh"] <= 16 and c["bw"] >= c["bh"]]


def perceive_radmanso_drawer_sort_table(ag, sim, k: int):
    hz = float(sim.task.extra.get("handle_z", DR_HANDLE_Z))
    if k == 0:
        hs = _handles(sim.images_raw["main"])
        if len(hs) < 3:
            return _fail(ag, "three drawer handles")
        hs = sorted(sorted(hs, key=lambda c: -c["area"])[:3], key=lambda c: c["v"])   # three drawers one above the other: take the middle one
        c = hs[1]
        q = sim.locate_on_plane("main", c["u"], c["v"], hz)
        ag.est["handle_main"] = q
        ag._log("drawer_handle", q, "main_middle_brass")
        return [ag._tc("point_tool", direction=[0, -1, 0]),
                ag._tc("move_to", x=float(q[0]), y=float(q[1]) + 0.12, z=hz)], None
    q = ag.est["handle_main"]
    hs = _handles(sim.images_raw["wrist"])
    uv = sim.project("wrist", q)
    p = np.asarray(q, dtype=float)
    if hs and uv is not None:
        c = min(hs, key=lambda c: (c["u"] - uv[0]) ** 2 + (c["v"] - uv[1]) ** 2)
        r = sim.locate_on_plane("wrist", c["u"], c["v"], float(q[1]), axis=1)    # wrist camera looks along -y at the drawer front: plane y = handle y
        if r is not None and float(np.linalg.norm(r - p)) < 0.03:
            ag._log("drawer_handle", r, "wrist_plane_y")
            p = np.array([r[0], q[1], r[2]])
    scene = [dict(name="drawer_handle", kind="handle", pos=[float(p[0]), float(p[1]), float(p[2])], opens_along=[0.0, 1.0, 0.0]),
             dict(name="table", kind="surface", top_z=0.0)]
    return None, scene


# ------------------------------------------------------------------ nut_thread
_NUT, _NUT_POST = cylinder_outline(0.013, NT_NUT_BOTTOM_Z, NT_NUT_TOP_Z), cylinder_outline(0.009, 0.0, NT_NUT_BOTTOM_Z)
_BOLT_BASE, _BOLT_SHANK = cylinder_outline(0.012, 0.0, 0.010), cylinder_outline(0.008, 0.010, NT_SHANK_TOP_Z)


def NT_NUT(x, y):
    """Hexagonal nut as a cylinder (radius 13 mm, between its 12 mm inner and 13.9 mm outer radius) + the small post below it
    (fitting the nut alone gave IoU 0.23-0.36 and was rejected)."""
    return np.concatenate([_NUT(x, y), _NUT_POST(x, y)])


def NT_BOLT(x, y):
    return np.concatenate([_BOLT_BASE(x, y), _BOLT_SHANK(x, y)])


def steel_mask(img: np.ndarray, thr: int = 110) -> np.ndarray:
    """Main view: dark-grey steel parts (nut / bolt ~ 60-100) on the light wooden table (~ 190, 165, 140). The robot itself is
    grey-white, so a greyness test cannot be used in the main view."""
    return np.asarray(img, dtype=int).max(axis=2) < thr


def grey_mask(img: np.ndarray, chroma: int = 25, top: int = 235) -> np.ndarray:
    """Wrist view: steel parts are grey (R ~ G ~ B, bright faces 100-180), the wooden table is warm (R - B ~ 50). Seen from
    close, the bright faces of the nut exceed a darkness threshold, which then covered only half of it (IoU 0.1-0.5, rejected);
    the greyness test gives IoU 0.88-0.90 and errors <= 0.5 mm."""
    im = np.asarray(img, dtype=int)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    return (np.abs(R - B) < chroma) & (np.abs(G - B) < chroma) & (im.max(axis=2) < top)


def perceive_nut_thread(ag, sim, k: int):
    from forceharness.backends.forcebench.vision import _components

    if k == 0:
        main = sim.images_raw["main"]
        cs = sorted([c for c in _components(steel_mask(main)) if c["area"] >= 30], key=lambda c: -c["area"])[:2]
        if len(cs) < 2:
            return _fail(ag, "the nut and the bolt")
        nut_c, bolt_c = sorted(cs, key=lambda c: c["v"])          # the nut is on the near post (top of the image = near the robot), the bolt further away
        mask = steel_mask(main)
        est = {}
        for key, c, z, outline in (("nut_main", nut_c, 0.5 * (NT_NUT_BOTTOM_Z + NT_NUT_TOP_Z), NT_NUT),
                                   ("bolt_main", bolt_c, 0.5 * NT_SHANK_TOP_Z, NT_BOLT)):
            q = sim.locate_on_plane("main", c["u"], c["v"], z)
            m = component_near(mask, (c["u"], c["v"]), min_area=10)
            if m is not None:                                     # the oblique view adds the post / sides (centroid 12-19 mm off): fit, then seed the wrist view
                xy, _ = fit_projection(sim, "main", _filled(m), outline, q[:2], span=0.03)
                q = np.array([xy[0], xy[1], z])
            est[key] = q
        n, b = est["nut_main"], est["bolt_main"]
        ag.est.update(nut_main=n, bolt_main=b)
        ag._log("nut", n, "main_fit")
        ag._log("target_bolt", b, "main_fit")
        # viewpoint straight above the bolt (checked offline from this viewpoint: error <= 1.5 mm)
        return [ag._tc("move_to", x=float(b[0]), y=float(b[1]), z=0.15)], None
    wrist = sim.images_raw["wrist"]
    if k == 1:
        b = ag.est["bolt_main"]
        xy = np.asarray(b[:2], dtype=float)
        m = component_near(grey_mask(wrist), sim.project("wrist", [b[0], b[1], NT_SHANK_TOP_Z]), min_area=30)
        if m is not None:
            fit, iou = fit_projection(sim, "wrist", _filled(m), NT_BOLT, xy, span=0.02)
            ok = iou > 0.5 and float(np.linalg.norm(fit - xy)) < 0.03
            ag._log("target_bolt", [fit[0], fit[1], 0.0], f"wrist_fit iou={iou:.2f} used={ok}")
            if ok:
                xy = fit
        ag.est["bolt"] = xy
        n = ag.est["nut_main"]
        return [ag._tc("move_to", x=float(n[0]), y=float(n[1]), z=0.127)], None
    n = ag.est["nut_main"]
    xy = np.asarray(n[:2], dtype=float)
    m = component_near(grey_mask(wrist), sim.project("wrist", [n[0], n[1], NT_NUT_TOP_Z]), min_area=30)
    if m is not None:
        fit, iou = fit_projection(sim, "wrist", _filled(m), NT_NUT, xy, span=0.02)   # fill the hole: from above, the nut is a hexagonal ring
        ok = iou > 0.5 and float(np.linalg.norm(fit - xy)) < 0.03
        ag._log("nut", [fit[0], fit[1], NT_NUT_BOTTOM_Z], f"wrist_fit iou={iou:.2f} used={ok}")
        if ok:
            xy = fit
    b = ag.est["bolt"]
    scene = [dict(name="nut", kind="object", pos=[float(xy[0]), float(xy[1]), NT_NUT_BOTTOM_Z], yaw_deg=0.0),
             dict(name="bolt", kind="fixture", pos=[float(b[0]), float(b[1]), 0.0], shank_top_z=NT_SHANK_TOP_Z,
                  target_nut_bottom_z=NT_TARGET_BOTTOM_Z),
             dict(name="table", kind="surface", top_z=0.0)]
    return None, scene
