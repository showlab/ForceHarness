"""HiddenPhys task registry: the seven MuJoCo (robosuite) tasks of the benchmark, their shifted variants, and the base
tasks the hard variants are built from.

This module defines the benchmark, not the harness: it is used by the MuJoCo backend (``forceharness.backends.mujoco``) and
by every reference (expert scripts, learned policies, Code-as-Policies), so all methods face exactly the same episodes.

Each task hides one physical variable that decides success and that a camera cannot see. The instruction states the range
of the hidden variable, never its value, which is drawn from the episode seed. A ``TaskSpec`` answers three questions:

- why force is needed: ``force_role`` (an internal note, never shown to the model);
- how success is checked: ``success``, computed from privileged simulator state on the final state;
- what the model is told: ``instruction`` and ``describe_scene`` (ground-truth poses plus a per-episode fixed offset,
  rendered as text).

Hidden variables are drawn per seed in ``reset_hook``, or in ``env_kwargs_fn`` when they must be fixed before the MuJoCo
model is built. ``step_hook`` runs on every control tick for task-specific process checks (pressure threshold, damage).

| task | scene | hidden variable | what force must do |
|---|---|---|---|
| weight_sort | Lift | cube mass, 0.15 or 0.60 kg | weigh the held cube |
| com_sort | ComSortBar (custom) | centre of mass of a bar, centred or 4 cm off | measure the held moment |
| wipe_hard | Wipe | table height, up to about 6 cm low | press 5-15 N on 12 spots |
| stack_gentle | Stack | top of the lower cube, known to +/-1 cm | feel contact, place gently |
| peg_in_hole_hard | PegInHoleRound (custom) | hole position, 15 mm error per axis | search until the peg drops |
| door_latch | Door with latch | unlock angle, 15-35 deg | press to release, then pull |
| lock_or_stiff | Door | locked, or stiff and freed at 6-11 N | pull below the 15 N damage line |

The ``*_ood`` variants change only the distribution of the hidden variable, and their instructions state the new range.
``wipe``, ``peg_in_hole`` and ``door`` are the base tasks; they stay importable for development but are not part of the
benchmark.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np


def _yaw_from_wxyz(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


@dataclasses.dataclass
class TaskSpec:
    name: str
    env_name: str
    instruction: str
    force_role: str
    env_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    env_kwargs_fn: Optional[Callable[[np.random.Generator], Dict[str, Any]]] = None
    objects: List[Dict[str, str]] = dataclasses.field(default_factory=list)   # {"label","kind","name"}
    reset_hook: Callable = lambda sim, rng: None
    success: Callable = lambda sim: False
    describe_scene: Callable = lambda sim: []
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)
    has_gripper: bool = True
    truth_fn: Optional[Callable] = None     # sim -> dict, stored in report.truth after each primitive (evaluation / debugging only; never shown to the model)
    step_hook: Optional[Callable] = None    # (sim, wrench), called once per control tick: task-specific process checks (e.g. the pressure threshold of wipe_hard)
    pregrasp_hook: Optional[Callable] = None  # (sim, prims) -> dict: set-up motions run with the primitives before the episode (e.g. lock_or_stiff grasps the handle); not counted as calls, results not shown to the model, not added to reports

    def object_body_substrs(self) -> List[str]:
        return [o["name"] for o in self.objects if o.get("kind", "body") == "body"]


# ----------------------------------------------------------------------------- weight_sort (Lift)
WS_LIGHT, WS_HEAVY = 0.15, 0.60   # kg; the two classes differ by 4.4 N at the wrist, far above the sensor noise
WS_ZONES = {"heavy_zone": np.array([0.0, 0.20]), "light_zone": np.array([0.0, -0.20])}


def _ws_reset(sim, rng):
    seed = getattr(sim, "seed", None)      # classes are balanced by seed parity (even = light, odd = heavy); a coin flip can give 8:2 at n = 10
    label = ("heavy" if seed % 2 else "light") if seed is not None else ("heavy" if rng.random() < 0.5 else "light")
    ex = getattr(getattr(sim, "task", None), "extra", None) or {}
    mass = float(ex.get("heavy_kg", WS_HEAVY)) if label == "heavy" else float(ex.get("light_kg", WS_LIGHT))      # shifted variants override the masses through extra
    sim.set_body_mass("cube_main", mass)
    sim.task_state.update(label=label, mass=mass)


def _table_top(sim) -> float:
    off = getattr(sim.env, "table_offset", None)
    return float(off[2]) if off is not None else 0.8


def _ws_success(sim) -> bool:
    cube = sim.body_pos("cube_main")
    zone = WS_ZONES[f"{sim.task_state['label']}_zone"]
    resting = cube[2] < _table_top(sim) + 0.06
    in_zone = abs(cube[0] - zone[0]) < 0.08 and abs(cube[1] - zone[1]) < 0.08
    released = (sim.gripper_opening or 0) > 0.03 and "cube_main" not in sim.gripper_contact_state()["touching"]
    return bool(resting and in_zone and released)


def _ws_scene(sim) -> List[Dict[str, Any]]:
    top = _table_top(sim)
    cube = sim.body_pos("cube_main")
    pos, yaw = sim.noised("cube", cube, _yaw_from_wxyz(sim.body_quat_wxyz("cube_main")))
    out = [dict(name="cube", kind="object", pos=pos.round(4).tolist(), yaw_deg=round(yaw, 1), size_m=0.04,
                note="one of two visually identical cubes types: light or heavy; appearance does not tell which")]
    for zn, zc in WS_ZONES.items():
        out.append(dict(name=zn, kind="zone", pos=[float(zc[0]), float(zc[1]), top], radius_m=0.08))
    out.append(dict(name="table", kind="surface", top_z=round(top, 3)))
    return out


WEIGHT_SORT = TaskSpec(
    name="weight_sort", env_name="Lift",
    instruction=("There is one cube on the table. It is either LIGHT (about {light} kg) or HEAVY (about {heavy} kg); the two "
                 "types look identical. Pick it up, decide which type it is, and place it inside the matching zone: "
                 "heavy_zone or light_zone (zone centers are given in the scene; a zone is a 16 cm square around its "
                 "center). Release it there and move the gripper up. Then call done with your verdict.").format(
                     light=WS_LIGHT, heavy=WS_HEAVY),
    force_role="the mass can only be read from the z component of the wrist force while lifting; a camera cannot see it",
    objects=[dict(label="cube", kind="body", name="cube_main")],
    reset_hook=_ws_reset, success=_ws_success, describe_scene=_ws_scene,
    extra=dict(light_kg=WS_LIGHT, heavy_kg=WS_HEAVY, zones={k: v.tolist() for k, v in WS_ZONES.items()}),
)


# ----------------------------------------------------------------------------- wipe (Wipe, a wiping tool without fingers)
WIPE_TABLE_NOMINAL = 0.90   # default table height of robosuite 1.5 Wipe (z of task_config table_offset)


def _wipe_env_kwargs(rng: np.random.Generator, seed: Optional[int] = None) -> Dict[str, Any]:
    """The table height is randomised by Wipe's own ``table_height_std`` (the dirt markers move with the table; changing
    table_offset directly would bury them in the table). Wipe draws from the global np.random, which ForceSim seeds."""
    from robosuite.environments.manipulation.wipe import DEFAULT_WIPE_CONFIG   # robosuite 1.5: Wipe's task parameters go through task_config

    task_config = {**DEFAULT_WIPE_CONFIG, "num_markers": 12, "line_width": 0.04, "use_condensed_obj_obs": True,
                   "early_terminations": False, "table_height": 0.0, "table_height_std": 0.012}
    return dict(task_config=task_config)


def _wipe_reset(sim, rng):
    zs = [sim.body_pos(b)[2] for b in sim.find_names("body", "contact") if b.endswith("_main")]
    sim.task_state.update(table_top=round(float(np.mean(zs)), 4) if zs else _table_top(sim))


def _wipe_success(sim) -> bool:
    return bool(sim.env._check_success())


def _wipe_scene(sim) -> List[Dict[str, Any]]:
    env = sim.env
    out = []
    pts = []
    # robosuite 1.5: the dirt markers are bodies contact{i}_main; wiped ones are kept in env.wiped_markers (objects whose .name looks like "contact3")
    wiped = set()
    for mk in getattr(env, "wiped_markers", []):
        wiped.add(getattr(mk, "name", str(mk)))
    for body in sim.find_names("body", "contact"):
        if not body.endswith("_main"):
            continue
        if body[:-5] in wiped:
            continue
        pts.append(sim.body_pos(body))
    if pts:
        pts = np.array(pts)
        xy = pts[:, :2]
        out.append(dict(name="dirt_markers", kind="targets", count=int(len(pts)),
                        xy_list=[[round(float(a), 3), round(float(b), 3)] for a, b in xy],
                        note="dirt spots to wipe; a spot is wiped when the wiper pad passes over it while touching the table"))
    out.append(dict(name="table", kind="surface", top_z_nominal=WIPE_TABLE_NOMINAL,
                    note="actual table height is within ±2 cm of nominal and is NOT known exactly"))
    return out


WIPE = TaskSpec(
    name="wipe", env_name="Wipe",
    instruction=("Wipe all the dirt spots off the table with the wiper pad on the tool. A spot counts as wiped only when the pad "
                 "sweeps over it while in contact with the table. The exact table height is unknown (within ±2 cm of 0.80 m). "
                 "Keep the contact gentle (avoid pressing harder than needed). Call done when all spots are gone."),
    force_role=("unknown table height: the surface must be found by contact and the normal force held while wiping; "
                "a camera cannot see the normal force"),
    env_kwargs_fn=_wipe_env_kwargs, objects=[], has_gripper=False,
    reset_hook=_wipe_reset, success=_wipe_success, describe_scene=_wipe_scene,
)


# ----------------------------------------------------------------------------- door (Door without latch)
def _door_reset(sim, rng):
    """Randomise hinge damping and friction per episode (the hinge joint is called Door_hinge in robosuite 1.5)."""
    hinges = sim.find_names("joint", "hinge")
    if hinges:
        jid = sim.mujoco.mj_name2id(sim.m, sim.mujoco.mjtObj.mjOBJ_JOINT, hinges[0])
        dof = int(sim.m.jnt_dofadr[jid])
        damping = float(rng.uniform(0.5, 4.0))
        friction = float(rng.uniform(0.0, 1.0))
        sim.m.dof_damping[dof] = damping
        sim.m.dof_frictionloss[dof] = friction
        sim.task_state.update(hinge_damping=damping, hinge_friction=friction)


def _door_success(sim) -> bool:
    return bool(sim.env._check_success())


def _door_geometry(sim) -> Optional[Dict[str, Any]]:
    """Geometry of the handle bar and the door panel. A camera would see it, so it is given as text when there are no
    images: bar axis and size, panel normal toward the robot, and the gap between the bar and the panel."""
    mj, m, d = sim.mujoco, sim.m, sim.d
    gh = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "Door_handle")
    gp = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "Door_panel")
    if gh < 0 or gp < 0:
        return None
    Rh, Rp = d.geom_xmat[gh].reshape(3, 3), d.geom_xmat[gp].reshape(3, 3)
    sh, sp = m.geom_size[gh], m.geom_size[gp]
    ch, cp = d.geom_xpos[gh].copy(), d.geom_xpos[gp].copy()
    n = Rp[:, int(np.argmin(sp))].copy()
    if float(n @ (-cp)) < 0:                       # point the normal toward the robot (its base is near the origin)
        n = -n
    axis = Rh[:, int(np.argmax(sh))].copy()
    half_n = float(sum(abs(Rh[:, i] @ n) * sh[i] for i in range(3)))
    half_z = float(sum(abs(Rh[:, i][2]) * sh[i] for i in range(3)))
    front = cp + n * float(np.min(sp))
    gap = float((ch - front) @ n) - half_n
    return dict(axis=axis, n=n, length=2 * float(np.max(sh)), thick=2 * half_n, height=2 * half_z, gap=gap, front=front)


def _door_scene(sim) -> List[Dict[str, Any]]:
    out = []
    handles = sim.find_names("site", "handle")
    geo = _door_geometry(sim)
    if handles:
        p = sim.site_pos(handles[0])
        pos, _ = sim.noised("handle", p)
        h = dict(name="door_handle", kind="object", pos=pos.round(4).tolist(),
                 note="grasp the handle, then pull; the door swings about a vertical hinge, resistance unknown")
        if geo is not None:
            h.update(shape="horizontal lever bar parallel to the door panel; pos is a graspable point on the bar",
                     bar_axis=np.round(geo["axis"], 3).tolist(), bar_length_m=round(geo["length"], 3),
                     bar_thickness_m=round(geo["thick"], 3), bar_height_m=round(geo["height"], 3),
                     gap_between_bar_and_panel_m=round(geo["gap"], 3))
        out.append(h)
    if geo is not None:
        out.append(dict(name="door_panel", kind="surface", front_face_point=np.round(geo["front"], 3).tolist(),
                        normal_toward_robot=np.round(geo["n"], 3).tolist(),
                        note="flat door panel; the handle bar sticks out from its front face"))
    hinges = sim.find_names("joint", "hinge")
    if hinges:
        jid = sim.mujoco.mj_name2id(sim.m, sim.mujoco.mjtObj.mjOBJ_JOINT, hinges[0])
        anchor = sim.d.xanchor[jid]
        out.append(dict(name="door_hinge", kind="axis", pos=anchor.round(4).tolist(),
                        note="vertical hinge axis; the handle moves on a circle around it (counter-clockwise seen from above opens the door)"))
    env = sim.env
    if hasattr(env, "hinge_qpos_addr"):
        out.append(dict(name="door_angle_deg", kind="state", value=round(math.degrees(float(sim.d.qpos[env.hinge_qpos_addr])), 1),
                        note="opening angle, success when > 17 deg"))
    return out


DOOR = TaskSpec(
    name="door", env_name="Door",
    instruction=("Open the door by at least 17 degrees. Grasp the handle and pull. The door rotates about a vertical hinge; "
                 "its resistance (damping / friction) is unknown and varies. Avoid excessive force. Call done when it is open."),
    force_role="constrained motion: the pull has to follow the hinge arc, and the resistance can only be felt",
    env_kwargs=dict(use_latch=False),
    objects=[dict(label="door", kind="body", name="Door")],
    # The handle is a bar close to the panel: grasping it from above makes the palm hit the panel, so door tasks also get
    # point_tool (level the tool and grasp from the side).
    extra=dict(extra_tools=["point_tool", "pull"]),        # pull (pull along the constraint) exists only on the force interface
    reset_hook=_door_reset, success=_door_success, describe_scene=_door_scene,
)


# ----------------------------------------------------------------------------- door_latch (lever handle with a spring latch)
DL_FDAMAGE = 15.0                       # N: while the gripper holds the handle, a wrist reaction above this damages the handle
DL_TH_LO, DL_TH_HI, DL_STOP_GAP = 15.0, 35.0, 8.0   # unlock angle theta_u ~ U[15, 35] deg; lever end stop at theta_u + 8 deg
DL_DETENT_NM, DL_DETENT_WIN = 0.45, 8.0             # detent ("click"): resisting torque rises 0 -> 0.45 N*m over [theta_u - 8, theta_u] deg, zero past theta_u
DL_LEVER_R = 0.125                      # m: from the free end of the handle to the lever axis (in the panel plane)


def _dl_ids(sim):
    mj, m = sim.mujoco, sim.m
    return (mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "Door_latch_joint"), mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "Door_hinge"))


def _dl_angles(sim):
    jl, jh = _dl_ids(sim)
    m, d = sim.m, sim.d
    return math.degrees(float(d.qpos[m.jnt_qposadr[jl]])), math.degrees(float(d.qpos[m.jnt_qposadr[jh]]))


def _door_latch_reset(sim, rng):
    _door_reset(sim, rng)
    mj, m = sim.mujoco, sim.m
    jl, jh = _dl_ids(sim)
    th_u = float(rng.uniform(DL_TH_LO, DL_TH_HI))
    th_stop = th_u + DL_STOP_GAP
    m.jnt_range[jl] = [0.0, math.radians(th_stop)]
    for gname in ("Door_latch", "Door_latch_tip"):        # disable the native latch geoms (they lock by hitting the frame); the lock is logical instead
        g = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, gname)
        if g >= 0:
            m.geom_contype[g] = 0
            m.geom_conaffinity[g] = 0
    m.jnt_range[jh] = [0.0, 0.0005]
    sim.task_state.update(theta_u_deg=round(th_u, 2), theta_stop_deg=round(th_stop, 2), locked=True, unlatched_ever=False,
                          damaged=False, peak_handle_force_N=0.0, f_damage=DL_FDAMAGE, lever_max_deg=0.0)


def _door_latch_step(sim, w):
    m, d = sim.m, sim.d
    st = sim.task_state
    jl, jh = _dl_ids(sim)
    th, door = _dl_angles(sim)
    th_u = float(st["theta_u_deg"])
    st["lever_max_deg"] = max(float(st.get("lever_max_deg", 0.0)), th)
    tau = 0.0
    if th_u - DL_DETENT_WIN <= th < th_u:                  # detent: harder to press the closer to theta_u, released right after it
        tau = -DL_DETENT_NM * (th - (th_u - DL_DETENT_WIN)) / DL_DETENT_WIN
    d.qfrc_applied[m.jnt_dofadr[jl]] = tau
    if st["locked"]:
        if th >= th_u:
            m.jnt_range[jh] = [0.0, 1.2]      # once unlocked, allow about 70 deg (robosuite's own 0.4 rad / 23 deg limit is hit while pulling and overshoots the 15 N damage line)
            st["locked"], st["unlatched_ever"] = False, True
    elif door < 1.5 and th < th_u - 2.0:                   # lever released before the door left the frame: the latch springs back
        m.jnt_range[jh] = [0.0, 0.0005]
        st["locked"] = True
    if getattr(sim, "grip_cmd", -1.0) > 0:                 # holding the handle: pressing or pulling past the damage line
        f = float(np.linalg.norm(sim.reaction_force(w)))
        st["peak_handle_force_N"] = max(float(st.get("peak_handle_force_N", 0.0)), f)
        if f > DL_FDAMAGE:
            st["damaged"] = True


def _door_latch_success(sim) -> bool:
    return bool(sim.env._check_success()) and not bool(sim.task_state.get("damaged", False))


def _door_latch_scene(sim) -> List[Dict[str, Any]]:
    out = _door_scene(sim)
    jl, _ = _dl_ids(sim)
    th, _ = _dl_angles(sim)
    out.append(dict(name="lever", kind="mechanism", pivot=np.round(sim.d.xanchor[jl], 4).tolist(), axis=np.round(sim.d.xaxis[jl], 3).tolist(),
                    radius_m=DL_LEVER_R, angle_deg=round(th, 1),
                    note="the door handle is a spring-loaded lever turning about this horizontal axis (through its base); a positive rotation "
                         "about this axis (right-hand rule) moves its free end DOWN, increases angle_deg and retracts the latch; it springs back up "
                         "when released"))
    return out


def _door_latch_truth(sim) -> Dict[str, Any]:
    th, door = _dl_angles(sim)
    st = sim.task_state
    return dict(lever_deg=round(th, 2), door_deg=round(door, 2), locked=st.get("locked"), damaged=st.get("damaged"),
                theta_u_deg=st.get("theta_u_deg"), peak_handle_force_N=round(float(st.get("peak_handle_force_N", 0.0)), 2))


DOOR_LATCH = TaskSpec(
    name="door_latch", env_name="Door",
    instruction=("Open the door by at least 17 degrees. The door has a lever handle with a spring latch: press the lever's free end DOWN until "
                 "the latch releases, keep it pressed while you start pulling the door open, then pull it open. Door hardware varies: the latch "
                 f"releases somewhere between {DL_TH_LO:g} and {DL_TH_HI:g} degrees of lever travel, and the lever hits its end stop about "
                 f"{DL_STOP_GAP:g} degrees after that. Any force above {DL_FDAMAGE:g} N on the handle (pressing or pulling) damages it and fails the "
                 "task. The door rotates about a vertical hinge; its resistance is unknown. Call done when it is open."),
    force_role=("the unlock point shows only in force (the detent: resistance peaks, then drops); probing pulls or pressing to "
                "the end stop without force exceed the 15 N damage line"),
    env_kwargs=dict(use_latch=True),
    objects=[dict(label="door", kind="body", name="Door")],
    reset_hook=_door_latch_reset, success=_door_latch_success, describe_scene=_door_latch_scene, step_hook=_door_latch_step,
    truth_fn=_door_latch_truth,
    extra=dict(extra_tools=["point_tool", "pull", "turn"]),     # turn (rotate the lever) on both interfaces; pull only on the force interface
)


# ----------------------------------------------------------------------------- stack_gentle (Stack: place the red cube gently on the green one)
def _stack_reset(sim, rng):
    dz = float(rng.uniform(-0.01, 0.01))     # the reported height of the green cube carries a fixed error within +/-1 cm (a camera cannot judge the landing height well)
    sim.task_state.update(cubeB_z_error=round(dz, 4))


def _stack_success(sim) -> bool:
    return bool(sim.env._check_success())


def _stack_scene(sim) -> List[Dict[str, Any]]:
    top = _table_top(sim)
    out = []
    a = sim.body_pos("cubeA_main")
    pa, yaw = sim.noised("cubeA", a, _yaw_from_wxyz(sim.body_quat_wxyz("cubeA_main")))
    out.append(dict(name="red_cube", kind="object", pos=pa.round(4).tolist(), yaw_deg=round(yaw, 1), size_m=0.04,
                    note="the cube to pick up"))
    b = sim.body_pos("cubeB_main")
    pb, _ = sim.noised("cubeB", b)
    pb = pb.copy(); pb[2] += sim.task_state.get("cubeB_z_error", 0.0)
    out.append(dict(name="green_cube", kind="object", pos=pb.round(4).tolist(), size_m=0.05,
                    note="place the red cube on top of this one; its height is uncertain by about ±1 cm"))
    out.append(dict(name="table", kind="surface", top_z=round(top, 3)))
    return out


STACK_GENTLE = TaskSpec(
    name="stack_gentle", env_name="Stack",
    instruction=("Pick up the red cube and place it on top of the green cube so that it rests there, then release and move up. "
                 "Do it gently: the height of the green cube's top is only known to about ±1 cm, so avoid pressing the red cube "
                 "into it or dropping it from high up. Call done when finished."),
    force_role=("the landing height is uncertain: contact has to tell when the cube has arrived; a camera cannot see how hard "
                "it presses"),
    objects=[dict(label="cubeA", kind="body", name="cubeA"), dict(label="cubeB", kind="body", name="cubeB")],
    reset_hook=_stack_reset, success=_stack_success, describe_scene=_stack_scene,
)


# ----------------------------------------------------------------------------- peg_in_hole (custom scene: a square peg, already in the gripper, into a round hole)
PEG_CLEARANCE = 0.0045


def _peg_reset(sim, rng):
    env = sim.env
    m, d = sim.m, sim.d
    # random position of the hole block (the block is welded to the table, so set body_pos directly)
    (x0, x1), (y0, y1) = env.hole_xy_range
    hx, hy = float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))
    m.body_pos[env.hole_body_id][:2] = [hx, hy]
    sim.mujoco.mj_forward(m, d)
    # put the peg between the fingers and close them (the task tests insertion only, not grasping)
    eef = sim.eef_pos
    adr = env.sim.model.get_joint_qpos_addr(env.peg.joints[0])
    if isinstance(adr, (tuple, list)):          # robosuite 1.5 returns (start, end) for a free joint
        adr = int(adr[0])
    dof = env.sim.model.get_joint_qvel_addr(env.peg.joints[0])
    if isinstance(dof, (tuple, list)):
        dof = int(dof[0])
    pose = [eef[0], eef[1], eef[2] - 0.01, 1, 0, 0, 0]              # peg centre 1 cm below the fingertip centre, held between the fingers
    for i in range(25):                                             # the peg would fall before the fingers close: put it back between them on every tick (kinematic support)
        d.qpos[adr:adr + 7] = pose
        d.qvel[dof:dof + 6] = 0
        sim.mujoco.mj_forward(m, d)
        sim.env.step(np.concatenate([np.zeros(6), [1.0]]))
    for _ in range(10):                                             # drop the kinematic support and let physics settle the grasp
        sim.env.step(np.concatenate([np.zeros(6), [1.0]]))
    sim.obs = sim.env._get_observations(force_update=True)
    sim.grip_cmd = 1.0
    sim.set_held_weld(True)                                         # weld the peg to the hand (see custom_envs); the release primitive removes the weld
    sim.task_state.update(hole_xy=[round(hx, 4), round(hy, 4)], clearance_m=PEG_CLEARANCE)


def _peg_success(sim) -> bool:
    return bool(sim.env._check_success())


def _peg_scene(sim) -> List[Dict[str, Any]]:
    env = sim.env
    top = env.hole_top()
    pos, _ = sim.noised("hole", top)
    out = [dict(name="hole", kind="target", pos=pos.round(4).tolist(), inner_radius_m=round(env.peg_radius + env.clearance, 4),
                note="round hole in a fixed block; position is approximate (up to about 1 cm error); pos is the centre of the hole opening (top face)"),
           dict(name="peg", kind="held_object", width_m=round(2 * env.peg_half_width, 4), length_m=round(2 * env.peg_half_length, 3),
                note="square peg (flat bottom), already held upright in the gripper; its bottom end is about 4 cm below the tool centre point"),
           dict(name="table", kind="surface", top_z=round(_table_top(sim), 3))]
    return out


def _peg_truth(sim) -> Dict[str, Any]:
    env = sim.env
    peg = sim.body_pos("peg_main"); top = env.hole_top()
    return dict(peg_minus_hole_xy_mm=[round(float((peg[0] - top[0]) * 1000), 1), round(float((peg[1] - top[1]) * 1000), 1)],
                depth_mm=round(float((top[2] - (peg[2] - env.peg_half_length)) * 1000), 1))


PEG_IN_HOLE = TaskSpec(
    name="peg_in_hole", env_name="PegInHoleRound",
    instruction=("A square peg is already held upright in the gripper. Insert it into the round hole in the fixed block on the table so that "
                 "it goes in by at least 2 cm, then release and move up. The clearance between the peg's corners and the hole wall is only about 4 mm, "
                 "and the hole position you are given has up to about 1 cm of error, so you will need to feel for the hole. "
                 "Do not force it: the safety layer aborts moves above the force limit. Call done when finished."),
    force_role=("4.5 mm from the peg corners to the hole wall and 6 mm (sigma) hole-position noise: alignment needs a contact "
                "search; only force tells jamming or tilt"),
    objects=[dict(label="peg", kind="body", name="peg"), dict(label="hole", kind="body", name="hole")],
    reset_hook=_peg_reset, success=_peg_success, describe_scene=_peg_scene, truth_fn=_peg_truth,
    extra=dict(min_scene_noise_pos=0.006, grip_force_scale=5.0),
)


# ----------------------------------------------------------------------------- com_sort (custom scene: bar with a centred or offset centre of mass; read the moment after lifting)
COM_OFFSET = 0.04
COM_ZONES = {"balanced_zone": np.array([0.0, 0.20]), "offset_zone": np.array([0.0, -0.20])}


def _com_env_kwargs(rng: np.random.Generator, seed: Optional[int] = None, offset: float = COM_OFFSET) -> Dict[str, Any]:
    """The centre-of-mass offset (where the hidden ballast sits) is fixed when the model is built. Classes are balanced by
    seed: even = offset, odd = balanced; the direction of the offset alternates with seed // 2."""
    if seed is not None:
        label = "offset" if seed % 2 == 0 else "balanced"
        sign = 1 if (seed // 2) % 2 == 0 else -1
    else:
        label = "offset" if rng.random() < 0.5 else "balanced"
        sign = 1 if rng.random() < 0.5 else -1
    d = offset * sign if label == "offset" else 0.0
    return dict(com_offset=d)


def _com_reset(sim, rng):
    env = sim.env
    d = float(getattr(env, "com_offset", 0.0))
    sim.task_state.update(label="offset" if abs(d) > 1e-6 else "balanced", com_offset_m=round(d, 3),
                          mass=round(float(sim.m.body_subtreemass[env.bar_body_id]), 3))


def _bar_body(sim) -> str:
    return getattr(sim.env.bar, "root_body", "bar_main")      # a CompositeObject names its root body differently from a BoxObject


def _com_success(sim) -> bool:
    bar = sim.body_pos(_bar_body(sim))
    zone = COM_ZONES[f"{sim.task_state['label']}_zone"]
    resting = bar[2] < _table_top(sim) + 0.06
    in_zone = abs(bar[0] - zone[0]) < 0.08 and abs(bar[1] - zone[1]) < 0.08
    released = (sim.gripper_opening or 0) > 0.05 and _bar_body(sim) not in sim.gripper_contact_state()["touching"]
    return bool(resting and in_zone and released)


def _com_scene(sim) -> List[Dict[str, Any]]:
    top = _table_top(sim)
    pos, yaw = sim.noised("bar", sim.body_pos(_bar_body(sim)), _yaw_from_wxyz(sim.body_quat_wxyz(_bar_body(sim))))
    out = [dict(name="bar", kind="object", pos=pos.round(4).tolist(), yaw_deg=round(yaw, 1), size_m=[0.12, 0.04, 0.04],
                note="12 cm bar, 0.8 kg; its centre of mass is either at the geometric centre or shifted 4 cm along its long axis; "
                     "appearance does not tell which. yaw_deg is the direction of the long axis")]
    for zn, zc in COM_ZONES.items():
        out.append(dict(name=zn, kind="zone", pos=[float(zc[0]), float(zc[1]), top], radius_m=0.08))
    out.append(dict(name="table", kind="surface", top_z=round(top, 3)))
    return out


def _com_truth(sim) -> Dict[str, Any]:
    out = dict(label=sim.task_state.get("label"), com_offset_m=sim.task_state.get("com_offset_m"))
    try:
        bid = sim.env.bar_body_id
        q = sim.d.xquat[bid]; w, x, y, z = q
        # world z component of the long axis (local x) -> tilt angle
        lx = 1 - 2 * (y * y + z * z); lz = 2 * (x * z - w * y)
        com = sim.d.xipos[bid]; ssite = sim.d.site_xpos[sim._ft_site]
        out.update(bar_tilt_deg=round(math.degrees(math.asin(max(-1.0, min(1.0, lz)))), 1),
                   com_lever_xy_mm=[round(float((com[0] - ssite[0]) * 1000), 1), round(float((com[1] - ssite[1]) * 1000), 1)],
                   bar_minus_eef_mm=[round(float((sim.d.xpos[bid][i] - sim.eef_pos[i]) * 1000), 1) for i in range(3)])
    except Exception as e:
        out["error"] = str(e)
    return out


COM_SORT = TaskSpec(
    name="com_sort", env_name="ComSortBar",
    instruction=("There is a 12 cm bar on the table (0.8 kg). Its centre of mass is either BALANCED (at the geometric centre) or OFFSET "
                 "(shifted 4 cm along the long axis); the two look identical. Pick the bar up at its geometric centre, decide which type "
                 "it is, and place it inside the matching zone: balanced_zone or offset_zone (16 cm squares around the given centres). "
                 "Release it there and move the gripper up, then call done with your verdict."),
    force_role=("the offset shows only in the wrist moment while lifting (0.3 N*m in theory, less once the bar tilts between "
                "the fingers); a camera cannot see it"),
    objects=[dict(label="bar", kind="body", name="bar")],
    env_kwargs_fn=_com_env_kwargs,
    reset_hook=_com_reset, success=_com_success, describe_scene=_com_scene, truth_fn=_com_truth,
    extra=dict(zones={k: v.tolist() for k, v in COM_ZONES.items()}, grip_force_scale=2.5),   # grip harder, or the offset bar tilts between the fingers and the moment is lost
)


def env_kwargs_for(task: "TaskSpec", seed: int) -> Dict[str, Any]:
    """Environment parameters that must be fixed when the environment is built, derived from the seed (class balance
    follows the seed)."""
    if task.env_kwargs_fn is None:
        return {}
    rng = np.random.default_rng(seed + 7)
    try:
        return task.env_kwargs_fn(rng, seed)
    except TypeError:
        return task.env_kwargs_fn(rng)


def _register_custom_envs():
    try:
        from forceharness.backends.mujoco.custom_envs import PegInHoleRound, ComSortBar   # noqa: F401  (importing registers the environments)
    except Exception:
        pass


# ----------------------------------------------------------------------------- lock_or_stiff
LS_FDAMAGE, LS_FB, LS_KINETIC, LS_ARM_STEPS = 15.0, (6.0, 11.0), 0.5, 20


def _ls_hinge(sim):
    mj, m = sim.mujoco, sim.m
    jh = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, sim.find_names("joint", "hinge")[0])
    return jh, int(m.jnt_dofadr[jh])


def _ls_limit_torque(sim, jh) -> float:
    """Torque (N*m) with which the hinge limit currently pushes back: only this joint's LIMIT_JOINT constraint rows."""
    mj, d = sim.mujoco, sim.d
    n = int(d.nefc)
    if n == 0:
        return 0.0
    sel = (d.efc_type[:n] == int(mj.mjtConstraint.mjCNSTR_LIMIT_JOINT)) & (d.efc_id[:n] == jh)
    return float(np.abs(d.efc_force[:n][sel]).sum())


def _ls_reset(sim, rng):
    """Locked and stiff doors are held by the same joint limit until breakaway, so before the door moves they look
    identical in position. (MuJoCo frictionloss is a soft constraint: a stiff door built with it creeps about 0.3 deg
    under a sub-breakaway force while a locked door stays still, a cue that a position-only strategy could read.)
    A stiff door is freed on the tick where the tangential force at the handle reaches F_b; after that its kinetic
    friction is 0.5 F_b."""
    _door_reset(sim, rng)
    mj, m, d = sim.mujoco, sim.m, sim.d
    jh, dof = _ls_hinge(sim)
    label = "locked" if int(getattr(sim, "seed", 0)) % 2 == 0 else "stiff"
    sid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, sim.find_names("site", "handle")[0])
    r = float(np.linalg.norm((d.site_xpos[sid] - d.xanchor[jh])[:2]))
    f_b = float(rng.uniform(*LS_FB))
    m.jnt_range[jh] = [0.0, 0.0005]
    sim.task_state.update(label=label, breakaway_N=round(f_b, 2) if label == "stiff" else None, handle_radius_m=round(r, 3),
                          broke_free=False, hold_N_max=0.0, damaged=False, peak_handle_force_N=0.0, f_damage=LS_FDAMAGE)
    sim._ls = dict(jh=jh, dof=dof, r=r, f_b=f_b)


def _ls_step(sim, w):
    st, k = sim.task_state, sim._ls
    m, d = sim.m, sim.d
    if not st["broke_free"]:
        q = float(d.qpos[m.jnt_qposadr[k["jh"]]])
        if q > 0.00025:                                       # pressing against the limit on the opening side
            f_hold = _ls_limit_torque(sim, k["jh"]) / k["r"]
            st["hold_N_max"] = round(max(float(st["hold_N_max"]), f_hold), 2)
            if st["label"] == "stiff" and f_hold >= k["f_b"]:
                m.jnt_range[k["jh"]] = [0.0, 1.2]
                m.dof_frictionloss[k["dof"]] = LS_KINETIC * k["f_b"] * k["r"]
                m.dof_damping[k["dof"]] = 0.5
                st["broke_free"] = True
    if getattr(sim, "grip_cmd", -1.0) > 0:                 # damage counts from 1 s after the gripper closes: on a stiff arm, closing drags the hand toward the handle centre for about 0.9 s at a transient 12-20 N (about 2 N on the soft arm), unrelated to locked / stiff
        k["closed_steps"] = k.get("closed_steps", 0) + 1
        if k["closed_steps"] > LS_ARM_STEPS:
            f = float(np.linalg.norm(sim.reaction_force(w)))
            st["peak_handle_force_N"] = max(float(st.get("peak_handle_force_N", 0.0)), f)
            if f > LS_FDAMAGE:
                st["damaged"] = True
    else:
        k["closed_steps"] = 0


def _ls_pregrasp(sim, pr) -> Dict[str, Any]:
    """Grasp the handle for the model before the episode starts, so that the episode tests the locked / stiff decision
    rather than the grasp. The expert scripts (scripts/hiddenphys/lock_or_stiff_expert.py) call this same function:
    level the tool 12 cm out along the door tangent and 10 cm above the handle, descend to the handle height, move in
    along the tangent, close the gripper."""
    sc = sim.scene()
    h = np.array(next(o for o in sc if o.get("name") == "door_handle")["pos"])
    hg = np.array(next(o for o in sc if o.get("name") == "door_hinge")["pos"])
    r = (h - hg)[:2]; r /= np.linalg.norm(r); t = np.array([-r[1], r[0]])
    if t @ (-h[:2]) < 0:
        t = -t
    pre = h.copy(); pre[:2] += 0.12 * t
    pr.call("move_to", x=pre[0], y=pre[1], z=pre[2] + 0.10, stop_on_contact=False)
    pr.call("point_tool", direction=[-t[0], -t[1], 0.0])
    pr.call("move_to", x=pre[0], y=pre[1], z=pre[2], stop_on_contact=False)
    g = h.copy(); g[:2] -= 0.01 * t
    pr.call("move_to", x=g[0], y=g[1], z=g[2], stop_on_contact=False)
    ok = bool(pr.call("grasp").derived.get("closed_on_something"))
    sim.task_state["pregrasped"] = ok
    return dict(grasped=ok, door_deg=round(_ls_door_deg(sim), 3), broke_in_grasp=bool(sim.task_state.get("broke_free")))


def _ls_scene(sim) -> List[Dict[str, Any]]:
    out = _door_scene(sim)
    for o in out:
        if o.get("name") == "door_handle":
            o["note"] = ("the gripper is already closed around this handle bar (grasped for you before the episode started); "
                         "the door swings about a vertical hinge")
    return out


def _ls_door_deg(sim) -> float:
    mj, m, d = sim.mujoco, sim.m, sim.d
    jh = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, sim.find_names("joint", "hinge")[0])
    return math.degrees(float(d.qpos[m.jnt_qposadr[jh]]))


def _ls_success(sim) -> bool:
    """The door ends in the right state (stiff: open beyond 17 deg; locked: below 2 deg) and the handle is not damaged.
    Whether the verdict is right is recorded separately (``verdict_correct`` in the episode record); the expert scripts
    require both."""
    st = sim.task_state
    if st.get("damaged"):
        return False
    return _ls_door_deg(sim) > 17.0 if st.get("label") == "stiff" else _ls_door_deg(sim) < 2.0


LOCK_OR_STIFF = TaskSpec(
    name="lock_or_stiff", env_name="Door",
    instruction=("Your gripper is already holding the door handle. "
                 "This door is either LOCKED or just STIFF (if stiff, it takes 6-11 N at the handle to get it moving). If it is stiff, open it by "
                 "at least 17 degrees and call done with verdict 'stiff'. If it is locked, do not force it: call done with verdict 'locked'. "
                 f"Any force above {LS_FDAMAGE:g} N on the handle damages it and fails the task. The door rotates about a vertical hinge."),
    force_role=("before the door moves, locked and stiff look the same in position (one joint limit holds both); only force "
                "separates the 6-11 N breakaway from the 15 N damage line"),
    env_kwargs=dict(use_latch=False),
    objects=[dict(label="door", kind="body", name="Door")],
    reset_hook=_ls_reset, success=_ls_success, describe_scene=_ls_scene, step_hook=_ls_step, pregrasp_hook=_ls_pregrasp,
    extra=dict(extra_tools=["point_tool", "pull"], pos_tol=0.0001, slow_in_contact=True),   # two settings for the position interface: sub-millimetre steps (0.1 mm arrival tolerance) and genuinely slow motion in contact
)


# ============================================================================= Hard variants (strong models solve the base tasks even without force, so the base versions cannot measure the ceiling)
# ----------------------------------------------------------------------------- wipe_hard: more uncertain table height + a spot comes off only at >= 5 N + more than 15 N is damage
WIPE_HARD_FMIN, WIPE_HARD_FMAX = 5.0, 15.0


def _wipe_hard_env_kwargs(rng: np.random.Generator, seed: Optional[int] = None, table_height: float = 0.0) -> Dict[str, Any]:
    from robosuite.environments.manipulation.wipe import DEFAULT_WIPE_CONFIG

    task_config = {**DEFAULT_WIPE_CONFIG, "num_markers": 12, "line_width": 0.04, "use_condensed_obj_obs": True,
                   "early_terminations": False, "table_height": table_height, "table_height_std": 0.03}
    return dict(task_config=task_config)


def _wipe_hard_reset(sim, rng):
    _wipe_reset(sim, rng)
    sim.task_state.update(good_wiped=[], light_wipes=0, peak_normal_N=0.0, f_min=WIPE_HARD_FMIN, f_max=WIPE_HARD_FMAX)


def _wipe_hard_step(sim, w):
    """robosuite 1.5 counts a spot as wiped whenever the pad sweeps within 2 cm above it (with any contact in the
    simulation), regardless of pressure. This adds the pressure threshold: a spot wiped with a normal force below f_min
    does not count; it is taken out of env.wiped_markers and made visible again, so it can be wiped again. Also records
    the peak normal force."""
    env = sim.env
    st = sim.task_state
    fz = float(sim.reaction_force(w)[2])            # pressing on the table -> reaction along +z
    st["peak_normal_N"] = max(float(st.get("peak_normal_N", 0.0)), fz)
    good = st.setdefault("good_wiped", [])
    for mk in list(getattr(env, "wiped_markers", [])):
        nm = getattr(mk, "name", str(mk))
        if nm in good:
            continue
        if fz >= WIPE_HARD_FMIN:
            good.append(nm)
        else:
            env.wiped_markers.remove(mk)
            st["light_wipes"] = int(st.get("light_wipes", 0)) + 1
            try:
                gid = sim.mujoco.mj_name2id(sim.m, sim.mujoco.mjtObj.mjOBJ_GEOM, mk.visual_geoms[0])
                sim.m.geom_rgba[gid][3] = 1.0
            except Exception:
                pass


def _wipe_hard_success(sim) -> bool:
    st = sim.task_state
    n = int(getattr(sim.env, "num_markers", 0) or 0)
    return bool(n > 0 and len(st.get("good_wiped", [])) >= n and float(st.get("peak_normal_N", 0.0)) < WIPE_HARD_FMAX)


def _wipe_hard_scene(sim) -> List[Dict[str, Any]]:
    out = _wipe_scene(sim)
    out.append(dict(name="wiping_rule", kind="rule", min_normal_force_N=WIPE_HARD_FMIN, max_normal_force_N=WIPE_HARD_FMAX,
                    note="a spot only comes off while the pad presses the table with at least min_normal_force_N; "
                         "exceeding max_normal_force_N at any time damages the table (task failure)"))
    return out


WIPE_HARD = TaskSpec(
    name="wipe_hard", env_name="Wipe",
    instruction=("Wipe all the dirt spots off the table with the wiper pad on the tool. The table may sit up to about 6 cm BELOW its nominal "
                 "height (unknown), so you must find the surface. A spot only comes off when the pad sweeps over it while pressing on the "
                 f"table with at least {WIPE_HARD_FMIN:g} N; a lighter touch leaves the spot in place. Pressing harder than {WIPE_HARD_FMAX:g} N "
                 "at any moment damages the table and fails the task. Call done when all spots are gone."),
    force_role=("table height unknown by up to about 6 cm; a spot comes off only with the normal force inside [5, 15] N: "
                "without force there is no way to know whether the press is enough or too much"),
    env_kwargs_fn=_wipe_hard_env_kwargs, objects=[], has_gripper=False,
    reset_hook=_wipe_hard_reset, success=_wipe_hard_success, describe_scene=_wipe_hard_scene, step_hook=_wipe_hard_step,
    extra=dict(wipe_force=9.0),
)

# ----------------------------------------------------------------------------- peg_in_hole_hard: hole-position error sigma 1.5 cm; inserting straight down fails, a contact search is required
PEG_IN_HOLE_HARD = dataclasses.replace(
    PEG_IN_HOLE, name="peg_in_hole_hard",
    instruction=("A square peg is already held upright in the gripper. Insert it into the round hole in the fixed block on the table so that "
                 "it goes in by at least 2 cm, then release and move up. The clearance between the peg's corners and the hole wall is only about 4 mm, "
                 "and the hole position you are given can be off by up to about 3 cm, so you will have to feel for the hole. "
                 "Do not force it: the safety layer aborts moves above the force limit. Call done when finished."),
    force_role="hole-position error sigma 15 mm: alignment needs a contact search (radius up to about 4 cm)",
    extra=dict(min_scene_noise_pos=0.015, grip_force_scale=5.0, search_radius=0.05),
)

# ----------------------------------------------------------------------------- shifted variants (distribution-shift column)
# Only the distribution of the hidden variable changes; object names, scene format and success checks stay the same, so
# trained policies run without retraining. The instruction given to the model states the new range.
WEIGHT_SORT_OOD = dataclasses.replace(
    WEIGHT_SORT, name="weight_sort_ood",
    instruction=WEIGHT_SORT.instruction.replace(f"about {WS_LIGHT} kg", "about 0.3 kg").replace(f"about {WS_HEAVY} kg", "about 0.45 kg"),
    extra={**WEIGHT_SORT.extra, "light_kg": 0.30, "heavy_kg": 0.45, "ood_of": "weight_sort"},
)
COM_SORT_OOD = dataclasses.replace(
    COM_SORT, name="com_sort_ood", instruction=COM_SORT.instruction.replace("shifted 4 cm", "shifted 2.5 cm"),
    env_kwargs_fn=lambda rng, seed=None: _com_env_kwargs(rng, seed, offset=0.025),
    extra={**COM_SORT.extra, "ood_of": "com_sort"},
)
WIPE_HARD_OOD = dataclasses.replace(
    WIPE_HARD, name="wipe_hard_ood", instruction=WIPE_HARD.instruction.replace("up to about 6 cm BELOW", "up to about 9 cm BELOW"),
    env_kwargs_fn=lambda rng, seed=None: _wipe_hard_env_kwargs(rng, seed, table_height=-0.03),
    extra={**WIPE_HARD.extra, "ood_of": "wipe_hard"},
)
PEG_IN_HOLE_HARD_OOD = dataclasses.replace(
    PEG_IN_HOLE_HARD, name="peg_in_hole_hard_ood",
    instruction=PEG_IN_HOLE_HARD.instruction.replace("off by up to about 3 cm", "off by up to about 5 cm"),
    extra={**PEG_IN_HOLE_HARD.extra, "min_scene_noise_pos": 0.025, "ood_of": "peg_in_hole_hard"},
)


#: The seven HiddenPhys tasks, in the order of the paper's table.
HIDDENPHYS_TASKS = ["weight_sort", "com_sort", "wipe_hard", "stack_gentle", "peg_in_hole_hard", "door_latch", "lock_or_stiff"]
#: Shifted variants used for the distribution-shift column (hidden variables drawn from a different range).
SHIFT_TASKS = ["weight_sort_ood", "com_sort_ood", "wipe_hard_ood", "peg_in_hole_hard_ood"]

TASKS: Dict[str, TaskSpec] = {t.name: t for t in (WEIGHT_SORT, COM_SORT, WIPE_HARD, STACK_GENTLE, PEG_IN_HOLE_HARD, DOOR_LATCH,
                                                  LOCK_OR_STIFF, WEIGHT_SORT_OOD, COM_SORT_OOD, WIPE_HARD_OOD, PEG_IN_HOLE_HARD_OOD)}
#: Base tasks the hard variants are built from; kept importable for development, not part of the benchmark.
BASE_TASKS: Dict[str, TaskSpec] = {t.name: t for t in (WIPE, PEG_IN_HOLE, DOOR)}
DEFAULT_TASKS = list(HIDDENPHYS_TASKS)


def get_task(name: str) -> TaskSpec:
    spec = TASKS.get(name) or BASE_TASKS.get(name)
    if spec is None:
        raise KeyError(f"unknown task {name}; available: {list(TASKS)}")
    if spec.env_name not in ("Lift", "Wipe", "Door", "Stack"):
        _register_custom_envs()
    return spec
