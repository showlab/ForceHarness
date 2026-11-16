"""Image tools for RGB-only perception (backend layer):
(1) the images shown to the model (enlarged, with a pixel grid; coordinates stay in the original 224 x 224 pixels);
(2) colour detection and known-shape projection fitting for the scripted agent (classical vision);
(3) `locate_known_object`, the shape fit behind the `locate_object` tool, which needs only a rough click from the model.

Colour thresholds were measured on the initial main-view images of weight_sort seeds 300-303: the tan box covers ~600 pixels
with a bounding box of about 23 x 30; the blue container ~2900 pixels, about 57 x 61 (it does not move between episodes); the
dark-brown table is RGB ~ (52-68, 45-58, 39-51). The black container is close in colour to shadows and the dark background, so
it is picked by connected components + area / compactness, or by a sliding window (find_rect).
"""
from __future__ import annotations

import io
from typing import Any, Dict, Optional, Tuple

import numpy as np


def grid_png(img: np.ndarray, scale: int = 2, step: int = 16, label_every: int = 32) -> bytes:
    """Enlarge by `scale`, draw a faint line every `step` pixels and a coordinate label every `label_every` pixels (original
    pixel units); return PNG bytes."""
    from PIL import Image, ImageDraw

    h, w = img.shape[:2]
    im = Image.fromarray(np.asarray(img, dtype=np.uint8)).resize((w * scale, h * scale), Image.NEAREST)
    dr = ImageDraw.Draw(im, "RGBA")
    for k in range(0, max(w, h) + 1, step):
        a = 110 if k % label_every == 0 else 45
        if k <= w:
            dr.line([(k * scale, 0), (k * scale, h * scale)], fill=(255, 255, 0, a), width=1)
        if k <= h:
            dr.line([(0, k * scale), (w * scale, k * scale)], fill=(255, 255, 0, a), width=1)
    for k in range(0, max(w, h), label_every):
        if k < w:
            dr.text((k * scale + 2, 1), str(k), fill=(255, 255, 0, 230))
        if 0 < k < h:
            dr.text((1, k * scale + 1), str(k), fill=(255, 255, 0, 230))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _components(mask: np.ndarray):
    from scipy import ndimage

    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        vs, us = np.nonzero(lab == i)
        area = len(us)
        if area == 0:
            continue
        bw, bh = us.max() - us.min() + 1, vs.max() - vs.min() + 1
        out.append(dict(area=area, u=float(us.mean()), v=float(vs.mean()), bbox=(int(us.min()), int(us.max()), int(vs.min()), int(vs.max())),
                        fill=area / float(bw * bh), bw=int(bw), bh=int(bh)))
    return out


MASKS = {
    "tan": lambda R, G, B: (R > 120) & (R - B > 45) & (R - G > 12) & (G - B > 15),
    "blue": lambda R, G, B: (B > 110) & (B - R > 60) & (B - G > 25),
    "black": lambda R, G, B: (R < 30) & (G < 30) & (B < 36) & (np.abs(R - B) < 12),
}


def find_blob(img: np.ndarray, color: str, area: Tuple[int, int], min_fill: float = 0.35,
              size_hint: Optional[Tuple[int, int]] = None, open_iter: int = 0) -> Optional[Dict[str, float]]:
    """Colour mask -> (optional) opening to cut thin bridges -> connected components -> compact blobs with an area within
    `area`. With size_hint (bounding-box width, height) the blob with the closest box wins, otherwise the largest one."""
    from scipy import ndimage

    im = np.asarray(img, dtype=int)
    m = MASKS[color](im[..., 0], im[..., 1], im[..., 2])
    if open_iter:
        m = ndimage.binary_opening(m, structure=np.ones((3, 3), bool), iterations=open_iter)
    cands = [c for c in _components(m) if area[0] <= c["area"] <= area[1] and c["fill"] >= min_fill]
    if not cands:
        return None
    if size_hint is not None:
        sw, sh = size_hint
        return min(cands, key=lambda c: abs(c["bw"] - sw) + abs(c["bh"] - sh))
    return max(cands, key=lambda c: c["area"])


def find_rect(img: np.ndarray, color: str, size: Tuple[int, int], min_cover: float = 0.45) -> Optional[Dict[str, float]]:
    """Slide a window of size = (width, height) over the colour mask and return the centre of the window with the highest
    coverage (the black container often merges with shadows / the dark background, so connected components cannot separate it)."""
    from scipy import ndimage

    im = np.asarray(img, dtype=int)
    m = MASKS[color](im[..., 0], im[..., 1], im[..., 2]).astype(float)
    w, h = int(size[0]), int(size[1])
    cover = ndimage.uniform_filter(m, size=(h, w), mode="constant", cval=0.0)
    v, u = np.unravel_index(int(np.argmax(cover)), cover.shape)
    if cover[v, u] < min_cover:
        return None
    return dict(u=float(u), v=float(v), cover=float(cover[v, u]), bw=w, bh=h)


# ------------------------------------------------------------------ projection fit of a known shape (classical model-based localisation)
# The wrist camera looks obliquely (optical axis about 30 deg from vertical, toward the robot), so a blob always includes the side
# facing the camera, and its centroid intersected at the height of the top face is 7-14 mm off in x. With known dimensions, the
# object's outline (8 corners of a box, two rings of a cylinder) is projected with the calibration, its convex hull is compared
# with the blob (IoU), and the centre is grid-searched in the world frame; the side faces are then accounted for and there is no
# bias. Only the calibration and the dimensions are used, never ground truth.

def dark_mask(img: np.ndarray, thr: int = 75) -> np.ndarray:
    """Dark regions on a bright background (black basket, cooktop, sink): max over the three channels < thr."""
    return np.asarray(img, dtype=int).max(axis=2) < thr


def color_mask(img: np.ndarray, color: str) -> np.ndarray:
    im = np.asarray(img, dtype=int)
    if color == "dark":
        return dark_mask(im)
    return MASKS[color](im[..., 0], im[..., 1], im[..., 2])


def component_near(mask: np.ndarray, uv: Optional[Tuple[float, float]] = None, min_area: int = 20) -> Optional[np.ndarray]:
    """The connected component of `mask` closest to uv (the largest one if uv is None), as a boolean image containing only it."""
    from scipy import ndimage

    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    best, best_key = None, None
    for i in range(1, n + 1):
        vs, us = np.nonzero(lab == i)
        if len(us) < min_area:
            continue
        if uv is None:
            key = -len(us)
        else:
            key = float(np.min((us - uv[0]) ** 2 + (vs - uv[1]) ** 2))
        if best_key is None or key < best_key:
            best, best_key = i, key
    return None if best is None else (lab == best)


def box_outline(size: Tuple[float, float, float], z0: float = 0.0, yaw_deg: float = 0.0):
    """Box with size = (along x, along y, height) at yaw = 0 and its bottom at z0. Returns f(x, y) -> world coordinates of the
    8 corners."""
    L, W, H = size
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    loc = np.array([[sx * L / 2, sy * W / 2] for sx in (-1, 1) for sy in (-1, 1)])
    rot = loc @ np.array([[c, s], [-s, c]])

    def f(x, y):
        xy = rot + np.array([x, y])
        return np.concatenate([np.c_[xy, np.full(4, z0)], np.c_[xy, np.full(4, z0 + H)]])
    return f


def cylinder_outline(radius: float, z0: float, z1: float, n: int = 24):
    """Upright cylinder (or a hexagonal nut / knob approximated by one) from z0 to z1. Returns f(x, y) -> sample points on the
    bottom and top rings."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.c_[np.cos(a), np.sin(a)] * radius

    def f(x, y):
        xy = ring + np.array([x, y])
        return np.concatenate([np.c_[xy, np.full(n, z0)], np.c_[xy, np.full(n, z1)]])
    return f


def _hull_mask(shape, uv: np.ndarray) -> Optional[np.ndarray]:
    from PIL import Image, ImageDraw
    from scipy.spatial import ConvexHull

    if uv is None or len(uv) < 3 or not np.all(np.isfinite(uv)):
        return None
    try:
        hull = uv[ConvexHull(uv).vertices]
    except Exception:
        return None
    im = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(im).polygon([(float(u) + 0.5, float(v) + 0.5) for u, v in hull], fill=1)
    return np.asarray(im, dtype=bool)


def fit_projection(sim, cam: str, mask: np.ndarray, outline, init_xy, span: float = 0.03, step: float = 0.002,
                   fine_step: float = 0.0005):
    """Search a grid of +-span around init_xy for the centre that maximises IoU(projected hull, blob), then refine within
    +-step of the best point. Returns (xy, IoU). Hull parts outside the image are clipped, so objects cut off by the image
    border can still be fitted."""
    h, w = mask.shape

    def score(x, y):
        pts = outline(x, y)
        uv = []
        for p in pts:
            q = sim.project(cam, p)
            if q is None:
                return -1.0
            uv.append(q)
        hm = _hull_mask((h, w), np.asarray(uv))
        if hm is None:
            return -1.0
        inter = np.count_nonzero(hm & mask)
        union = np.count_nonzero(hm | mask)
        return inter / union if union else -1.0

    def grid(cx, cy, sp, st):
        best = (cx, cy, score(cx, cy))
        n = int(round(sp / st))
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                x, y = cx + i * st, cy + j * st
                sc = score(x, y)
                if sc > best[2]:
                    best = (x, y, sc)
        return best

    x, y, _ = grid(float(init_xy[0]), float(init_xy[1]), span, step)
    x, y, sc = grid(x, y, step, fine_step)
    return np.array([x, y]), float(sc)


# ------------------------------------------------------------------ refinement for the model: one rough click + the known shape (locate_object)
# Clicking the bolt, models were systematically +10-17 mm off in x (the wrist camera looks about 30 deg off vertical; they clicked
# the middle of the bolt but intersected at the height of its top), while Nut Thread needs centring within 2.5 mm. The scripted
# agent's projection fit gets within 2 mm, but its colour thresholds are hand-written per object. This version needs no colour
# prior: it crops a patch around the click, takes the median colour of the patch border as the background, and counts as
# foreground what differs from it in chromaticity or is clearly darker / brighter (a shadow keeps the chromaticity and is only
# slightly darker, so it is not counted); then it takes the component nearest to the click, fills holes and fits the shape.

def click_mask(img: np.ndarray, uv: Tuple[float, float], radius: int, chroma_thr: float = 0.05, dark: float = 0.62,
               bright: float = 1.6) -> Optional[np.ndarray]:
    from scipy import ndimage

    h, w = img.shape[:2]
    u, v, r = int(round(uv[0])), int(round(uv[1])), int(radius)
    u0, u1, v0, v1 = max(u - r, 0), min(u + r + 1, w), max(v - r, 0), min(v + r + 1, h)
    if u1 - u0 < 4 or v1 - v0 < 4:
        return None
    crop = np.asarray(img[v0:v1, u0:u1], dtype=float)
    ring = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    bg = np.median(ring, axis=0)
    i_bg = max(float(bg.sum()), 1.0)
    inten = crop.sum(axis=2)
    chroma = crop / np.maximum(inten[..., None], 1.0)
    d_chroma = np.abs(chroma - bg / i_bg).sum(axis=2)
    ratio = inten / i_bg
    fg = ((d_chroma > chroma_thr) & (inten > 60.0)) | (ratio < dark) | (ratio > bright)   # chromaticity of very dark pixels is unstable: use the brightness ratio
    fg = ndimage.binary_opening(fg, structure=np.ones((2, 2), bool))
    out = np.zeros((h, w), bool)
    out[v0:v1, u0:u1] = fg
    comp = component_near(out, (float(uv[0]), float(uv[1])), min_area=8)
    return None if comp is None else ndimage.binary_fill_holes(comp)


def stack_outline(parts):
    """Stack of coaxial upright cylinders (bolt = base + shank; nut + post). parts = [(radius, z0, z1), ...]."""
    fs = [cylinder_outline(r, z0, z1) for r, z0, z1 in parts]

    def f(x, y):
        return np.concatenate([g(x, y) for g in fs])
    return f


def locate_known_object(sim, cam: str, u: float, v: float, parts) -> Optional[Dict[str, Any]]:
    """Rough click (u, v) + known shape -> world xy of the object's vertical axis and the fit score (IoU). Uses only the
    current image, the camera calibration and the dimensions of the shape."""
    outline = stack_outline(parts)
    z_mid = 0.5 * (min(p[1] for p in parts) + max(p[2] for p in parts))
    q = sim.locate_on_plane(cam, u, v, z_mid)
    if q is None:
        return None
    uv = [sim.project(cam, p) for p in outline(q[0], q[1])]
    if any(x is None for x in uv):
        return None
    uv = np.asarray(uv)
    size = float(max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1])))
    mask = click_mask(np.asarray(sim.images_raw[cam]), (u, v), int(np.clip(1.5 * size, 14, 70)))
    if mask is None:
        return None
    xy, iou = fit_projection(sim, cam, mask, outline, q[:2], span=0.03, step=0.002)
    return dict(xy=xy, score=float(iou), mask_area=int(mask.sum()), rough=np.asarray(q[:2], dtype=float))
