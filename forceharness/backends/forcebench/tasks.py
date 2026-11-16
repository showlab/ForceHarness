"""Harness-side descriptions of the eight ForceBench tasks (backend layer): what the agent is told about each task.

Each task has an instruction (ForceBench's own prompt + a "Task details" paragraph in the format of the HiddenPhys tasks) and a
scene description. Two variants exist:
- privileged (`TASKS`): the scene gives the true geometry from the evaluator's "privileged" field plus noise that is fixed per
  episode; fixed fixtures use the constants of ForceBench's task specifications.
- RGB only (`TASKS_RGB`, the paper's setting): no object or fixture coordinates. The model finds things in the two camera
  images and turns pixels into positions with `locate` / `locate_object`; the instruction gives dimensions and key heights.
Tasks: weight_sort and rattle_sort (CoM Sort), faucet_turn and faucet_force_turn (Faucet Direction), stove_turn_on and
stove_press_turn, radmanso_drawer_sort_table (Drawer Pull), nut_thread.

All numbers come from ForceBench at commit 31acf73, for example:
- weight_sort: `forcebench/tasks/weight_sort/spec.py` (box 0.1506 x 0.2118 x 0.0836 scaled by 0.36 = 54 x 76 x 30 mm,
  0.06 / 0.18 kg; black container at (0.52, -0.16), blue at (0.52, +0.16), opening 0.16 x 0.10 m, walls 75 mm; success = box
  centre within +-0.062 / +-0.032 m of the opening centre, z <= 0.06, gripper opening >= 0.65, box speed <= 0.15 m/s).
  Prompt: `forcebench/tasks/weight_sort/__init__.py`.
- rattle_sort: `forcebench/tasks/rattle_sort/spec.py` (parcel 54 x 120 x 23.3 mm, 0.18 kg, centre of mass centred or shifted
  20 mm along the long axis, about 0.035 N*m; black basket at (0.64, -0.22), opening 0.235 x 0.145, floor at z 0.0327, walls
  105 mm; usable conveyor area centred at (0.6909, 0.22), 0.24 x 0.22, surface at z 0.018).
Never given to the model: the hidden values (mass, centre of mass, detent angle, easier direction, resistance profile) and the
seed (ForceBench assigns classes by seed parity; the seed is part of rollout_id, which the harness does not pass on).
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List

import numpy as np

from forceharness.backends.forcebench.fbsim import FBSim, FBTask

TABLE_TOP_Z = 0.0          # the support surface of every ForceBench scene is at z = 0 (forcebench/scene_profiles.py)

# ------------------------------------------------------------------ weight_sort
WS_PROMPT = ("Pick up the box to determine whether it is light or heavy. Put a light box into the black container or a heavy box "
             "into the blue container at the side of the table, then release it.")
WS_LIGHT, WS_HEAVY = 0.06, 0.18
WS_BOX = (0.0762, 0.0542, 0.0301)      # world frame at yaw = pi/2: 76 mm along x, 54 mm along y (finger closing direction), 30 mm tall
WS_CONTAINERS = {"black_container": (0.52, -0.16), "blue_container": (0.52, 0.16)}


def _entity(sim: FBSim, role: str) -> Dict[str, Any]:
    ents = (sim.privileged or {}).get("entities") or {}
    if role not in ents:
        raise KeyError(f"privileged geometry has no entity {role!r} (run scripts/forcebench/eval_forcebench.py with --privileged_geometry)")
    return ents[role]


def _yaw_deg(e: Dict[str, Any]) -> float:
    return math.degrees(float(e["rpy"][2]))


def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _ws_scene(sim: FBSim) -> List[Dict[str, Any]]:
    box = _entity(sim, "box")
    root = np.asarray(box["pos"], dtype=float)
    centre = root + np.array([0.0, 0.0, WS_BOX[2] / 2])            # the root is at the bottom of the box (spawn z = 0.003)
    long_axis = _wrap180(_yaw_deg(box) - 90.0)                       # direction of the long side (76 mm, local y)
    pos, yaw = sim.noised("box", centre, long_axis)
    out = [dict(name="box", kind="object", pos=np.round(pos, 4).tolist(), yaw_deg=round(yaw, 1),
                size_m=[WS_BOX[0], WS_BOX[1], WS_BOX[2]],
                note="flat box: 76 mm along its long side (direction yaw_deg), 54 mm across, 30 mm tall; one of two visually "
                     "identical box types, light or heavy; appearance does not tell which")]
    for name, (cx, cy) in WS_CONTAINERS.items():
        out.append(dict(name=name, kind="container", pos=[cx, cy, TABLE_TOP_Z], opening_m=[0.16, 0.10], wall_height_m=0.075,
                        note=("for a LIGHT box" if name.startswith("black") else "for a HEAVY box")
                        + "; opening is 16 cm along x and 10 cm along y around pos; floor at the table surface"))
    out.append(dict(name="table", kind="surface", top_z=TABLE_TOP_Z))
    return out


WEIGHT_SORT = FBTask(
    name="weight_sort",
    instruction=(WS_PROMPT + f"\n\nTask details: the box is either LIGHT (about {WS_LIGHT} kg) or HEAVY (about {WS_HEAVY} kg); the two "
                 "types look identical. Container centres and openings are given in the scene. Lower the box into the matching "
                 "container (its walls are 7.5 cm tall), release it there and move the gripper up, then call done with your "
                 "verdict (light or heavy)."),
    describe_scene=_ws_scene,
    extra=dict(table_top_z=TABLE_TOP_Z, light_kg=WS_LIGHT, heavy_kg=WS_HEAVY, fb_prompt=WS_PROMPT,
               containers={k: list(v) for k, v in WS_CONTAINERS.items()}),
)

# ------------------------------------------------------------------ rattle_sort (CoM Sort in the paper)
RS_PROMPT = ("Lift and inspect the parcel without shaking it. Put an imbalanced parcel in the left black basket or a uniform parcel "
             "on the right conveyor, then release it.")
RS_BOX = (0.120, 0.054, 0.02328)       # world frame: long axis (120 mm) along x, 54 mm along y, 23 mm tall
RS_MASS, RS_OFFSET = 0.18, 0.020
RS_BASKET = dict(pos=(0.64, -0.22, 0.0327), opening=(0.235, 0.145), wall=0.105)
RS_CONVEYOR = dict(pos=(0.69093371, 0.22, 0.018), usable=(0.24, 0.22))


def _rs_scene(sim: FBSim) -> List[Dict[str, Any]]:
    box = _entity(sim, "box")
    root = np.asarray(box["pos"], dtype=float)
    centre = root + np.array([0.0, 0.0, RS_BOX[2] / 2])
    long_axis = _wrap180(_yaw_deg(box) - 90.0)                       # local +y (long axis) points to world -x at yaw = pi/2
    pos, yaw = sim.noised("parcel", centre, long_axis)
    out = [dict(name="parcel", kind="object", pos=np.round(pos, 4).tolist(), yaw_deg=round(yaw, 1),
                size_m=[RS_BOX[0], RS_BOX[1], RS_BOX[2]],
                note=f"12 cm parcel, {RS_MASS} kg; its centre of mass is either at the geometric centre (UNIFORM) or shifted "
                     f"{RS_OFFSET * 100:.0f} cm along its long axis (IMBALANCED); appearance does not tell which. "
                     "yaw_deg is the direction of the long axis")]
    out.append(dict(name="black_basket", kind="container", pos=list(RS_BASKET["pos"]), opening_m=list(RS_BASKET["opening"]),
                    wall_height_m=RS_BASKET["wall"],
                    note="for an IMBALANCED parcel; pos is the centre of the basket floor (z = floor top); opening 23.5 cm along x, "
                         "14.5 cm along y; walls 10.5 cm tall"))
    out.append(dict(name="conveyor", kind="zone", pos=list(RS_CONVEYOR["pos"]), size_m=list(RS_CONVEYOR["usable"]),
                    note="for a UNIFORM parcel; pos is the centre of the usable belt area (z = belt surface); 24 cm along x, 22 cm along y"))
    out.append(dict(name="table", kind="surface", top_z=TABLE_TOP_Z))
    return out


RATTLE_SORT = FBTask(
    name="rattle_sort",
    instruction=(RS_PROMPT + f"\n\nTask details: the parcel weighs {RS_MASS} kg. Its centre of mass is either UNIFORM (at the geometric "
                 f"centre) or IMBALANCED (shifted {RS_OFFSET * 100:.0f} cm along the long axis); the two look identical. Pick the parcel "
                 "up at its geometric centre, decide which type it is, and put it in the matching place (black_basket or conveyor, "
                 "given in the scene). Release it there and move the gripper up, then call done with your verdict "
                 "(imbalanced or uniform)."),
    describe_scene=_rs_scene,
    extra=dict(table_top_z=TABLE_TOP_Z, mass_kg=RS_MASS, com_offset_m=RS_OFFSET, fb_prompt=RS_PROMPT),
)

def _joint(e: Dict[str, Any], name: str) -> float:
    names = list(e.get("joint_names") or [])
    return float(e["joint_pos"][names.index(name)]) if name in names else 0.0


# ------------------------------------------------------------------ faucet_turn / faucet_force_turn (faucet lever)
# spec: valve root at (0.653, 0.015, -0.037); lever hinge = root + (0.01, 0, -0.01), about world y. The lever is a thin upright rod
# (radius 5.2 mm, 66 mm long, centred 59 mm above the hinge) whose centre line is offset (-0.09 mm, -13.9 mm) from the hinge.
# Joint reading = task angle (opening positive); physical rotation = -task angle (the rod top moves to -x, toward the robot).
FT_PIVOT_OFFSET = np.array([0.01, 0.0, -0.01])
FT_ROD_XY = (-0.00009, -0.013866420774313262)


def _faucet_scene(sim: FBSim, radius: float, two_way: bool) -> List[Dict[str, Any]]:
    v = _entity(sim, "faucet_valve")
    pivot = np.asarray(v["pos"], dtype=float) + FT_PIVOT_OFFSET
    phys = -_joint(v, "valve_joint")
    c, s_ = math.cos(phys), math.sin(phys)
    cx, cy = FT_ROD_XY
    grasp = pivot + np.array([cx * c + radius * s_, cy, -cx * s_ + radius * c])
    gpos, _ = sim.noised("faucet_grasp", grasp)
    ppos, _ = sim.noised("faucet_pivot", pivot)
    if two_way:
        dirs = ("it can turn either way, up to about 45 degrees: a NEGATIVE angle about +y moves the top of the lever toward the "
                "robot (-x), a POSITIVE angle moves it away (+x)")
    else:
        dirs = ("the water comes on when the top of the lever moves toward the robot (-x), i.e. a NEGATIVE rotation about +y "
                "(right-hand rule); it turns 0 to 110 degrees that way")
    return [dict(name="faucet_lever", kind="lever", pivot=np.round(ppos, 4).tolist(), axis=[0.0, 1.0, 0.0],
                 grasp_point=np.round(gpos, 4).tolist(), rod_diameter_m=0.0104,
                 note=("a thin rod standing upright on a hinge whose axis runs along world y through pivot; grasp_point is on the rod "
                       f"{radius * 100:.1f} cm above the pivot (close the fingers across the rod there); " + dirs)),
            dict(name="countertop", kind="surface", top_z=0.0)]


FT_PROMPT = "Rotate the faucet knob past the detent to turn on the water."
FAUCET_TURN = FBTask(
    name="faucet_turn",
    instruction=(FT_PROMPT + "\n\nTask details: the knob is the lever in the scene (faucet_lever). It has a detent - the resistance "
                 "rises, then suddenly drops - somewhere between 20 and 30 degrees from upright; the water turns on once the lever is "
                 "just past the detent. Grasp the lever at grasp_point and turn it about its hinge (turn tool; pivot and axis are in "
                 "the scene) toward the robot past the detent, then call done."),
    describe_scene=lambda sim: _faucet_scene(sim, 0.090, two_way=False),
    extra=dict(table_top_z=0.0, fb_prompt=FT_PROMPT, extra_tools=("turn",)),
)

FFT_PROMPT = ("Use force feedback to identify the lower-resistance direction, rotate the faucet lever past the randomized detent, "
              "and leave the water on.")
FAUCET_FORCE_TURN = FBTask(
    name="faucet_force_turn",
    instruction=(FFT_PROMPT + "\n\nTask details: the lever (scene: faucet_lever) can be turned either way about its hinge (world y "
                 "axis through pivot), up to about 45 degrees each way. One direction is clearly easier (lower resistance) than the "
                 "other. The water turns on only when the lever is turned past its detent - the resistance rises, then suddenly drops - "
                 "somewhere between 20 and 30 degrees in the EASIER direction. Grasp the lever at grasp_point, find the easier "
                 "direction, turn past the detent, then release the lever and move the gripper up to at least z = 0.11 m, leaving "
                 "the lever still. Call done with the easier direction as verdict ('toward_robot' or 'away_from_robot')."),
    describe_scene=lambda sim: _faucet_scene(sim, 0.065, two_way=True),
    extra=dict(table_top_z=0.0, fb_prompt=FFT_PROMPT, extra_tools=("turn",), min_final_tcp_z=0.108),
)

# ------------------------------------------------------------------ stove_turn_on / stove_press_turn (centre knob of the cooktop)
# spec: knob root at (0.4433, -0.0001, 0.0205), a cylinder of radius 19 mm and height 19.2 mm turning about the vertical axis,
# counter-clockwise from above positive; ForceBench's expert grasps 2 mm above the root (-1 mm on Stove Press).
ST_PROMPT = "Turn the center stove knob past the detent to ignite the burner."
SPT_PROMPT = "Press and hold the center stove knob, then turn it in the lower-resistance direction past the detent."


def _stove_scene(sim: FBSim, press: bool) -> List[Dict[str, Any]]:
    k = _entity(sim, "stove_knob")
    root = np.asarray(k["pos"], dtype=float)
    pos, _ = sim.noised("stove_knob", root)
    grasp = pos + np.array([0.0, 0.0, -0.001 if press else 0.002])
    turn_note = ("it only turns while pressed down by at least 7 mm (about 7 N; it bottoms out at 8 mm); it can turn either way, "
                 "up to 120 degrees" if press else
                 "it turns about the vertical axis through its centre; counter-clockwise seen from above (positive angle) turns it "
                 "on; range 0 to 120 degrees")
    return [dict(name="stove_knob", kind="knob", pos=np.round(pos, 4).tolist(), diameter_m=0.038, height_m=0.0192,
                 grasp_point=np.round(grasp, 4).tolist(),
                 note=("the centre knob of the cooktop: a 38 mm cylinder whose base is at pos; grasp it from above at grasp_point with "
                       "the fingers closing across it; " + turn_note)),
            dict(name="countertop", kind="surface", top_z=0.0)]


STOVE_TURN_ON = FBTask(
    name="stove_turn_on",
    instruction=(ST_PROMPT + "\n\nTask details: grasp the knob (scene: stove_knob) at grasp_point and turn it counter-clockwise seen "
                 "from above. It has an ignition detent - the resistance rises gradually, then suddenly drops - somewhere between 60 "
                 "and 90 degrees; the burner ignites just past it (a few degrees past is fine, far past is not needed). Then call done."),
    describe_scene=lambda sim: _stove_scene(sim, press=False),
    extra=dict(table_top_z=0.0, fb_prompt=ST_PROMPT),
)

STOVE_PRESS_TURN = FBTask(
    name="stove_press_turn",
    instruction=(SPT_PROMPT + "\n\nTask details: the knob (scene: stove_knob) only turns while it is pressed down by at least 7 mm "
                 "(about 7 N against its spring; it bottoms out at 8 mm) - use press_turn, which keeps pressing while it turns. It "
                 "can turn either way (up to 120 degrees), but one direction is clearly easier (lower resistance). The burner lights "
                 "only when the knob is turned past its detent - the resistance rises, then suddenly drops - somewhere between 60 and "
                 "90 degrees in the EASIER direction while pressed. Grasp the knob at grasp_point, find the easier direction, turn "
                 "past the detent, then call done with the direction as verdict ('ccw' or 'cw', seen from above)."),
    describe_scene=lambda sim: _stove_scene(sim, press=True),
    extra=dict(table_top_z=0.0, fb_prompt=SPT_PROMPT, fb_extra_tools=("press_turn",)),
)

# ------------------------------------------------------------------ radmanso_drawer_sort_table (Drawer Pull)
# spec: cabinet at (0.620, -0.275, 0), yaw pi, scale 0.45; middle drawer handle (closed) at about (0.6202, -0.1611, 0.2188);
# the drawer opens toward +y (opening = -joint reading).
DR_HANDLE_CLOSED = np.array([0.620 + 0.45 * 0.0005263, -0.275 + 0.45 * 0.2530647, 0.45 * 0.4863018])
DR_PROMPT = ("Grip the protruding round handle on the middle drawer of the walnut cabinet, pull toward the robot until resistance "
             "clearly rises, then stop and hold the drawer open without pulling it out.")


def _drawer_scene(sim: FBSim) -> List[Dict[str, Any]]:
    cab = _entity(sim, "cabinet")
    opening = -_joint(cab, "PrismaticJoint_middle")
    handle = DR_HANDLE_CLOSED + np.array([0.0, opening, 0.0])
    pos, _ = sim.noised("drawer_handle", handle)
    return [dict(name="drawer_handle", kind="handle", pos=np.round(pos, 4).tolist(), opens_along=[0.0, 1.0, 0.0],
                 note=("round knob handle on the front of the middle drawer, sticking out toward +y; grip it from the +y side with "
                       "the tool lying flat and the fingers pointing toward -y (point_tool [0, -1, 0]), so that the fingers close "
                       "up and down on it; the drawer slides out toward +y")),
            dict(name="cabinet", kind="fixture", pos=[0.620, -0.275, 0.0], note="walnut cabinet; its front faces +y"),
            dict(name="table", kind="surface", top_z=0.0)]


DRAWER = FBTask(
    name="radmanso_drawer_sort_table",
    instruction=(DR_PROMPT + "\n\nTask details: the drawer slides with only about 0.35 N of resistance for the first few "
                 "centimetres, then the resistance rises steeply over 1 to 3 cm. Stop where it first reaches about 2.4 N and hold "
                 "the drawer still there for at least 1 second (the target is within 2.5 mm of that point). Pulling through the steep "
                 "part (a peak of several newtons) breaks the latch and fails the task, and the drawer closes again if you let go. "
                 "Then call done."),
    describe_scene=_drawer_scene,
    extra=dict(table_top_z=0.0, fb_prompt=DR_PROMPT, extra_tools=("point_tool", "pull", "push_click")),
)

# ------------------------------------------------------------------ nut_thread (an M16 nut onto a bolt)
# spec: nut 0.03 kg, 24 mm across flats, about 10 mm tall, starting on a small post at (0.49, 0, 0.010) (root = bottom face of the
# nut); bolt root on the table, 10 mm base + 25 mm threaded shank (top at z 0.035), 2 mm pitch. Goal: nut bottom = bolt root +
# 0.032 (within -2 / +1 mm), centred within 2.5 mm, tilt <= 10 deg, released, hand away (5.5 cm above or 7 cm to the side).
NT_PROMPT = "Pick up the M16 nut, thread it onto the bolt, release it, and retract the gripper."


def _nut_scene(sim: FBSim) -> List[Dict[str, Any]]:
    nut = _entity(sim, "nut")
    bolt = _entity(sim, "target_bolt")
    npos, nyaw = sim.noised("nut", np.asarray(nut["pos"], dtype=float), math.degrees(float(nut["rpy"][2])))
    broot = np.asarray(bolt["pos"], dtype=float)
    bpos, _ = sim.noised("bolt", broot)
    return [dict(name="nut", kind="object", pos=np.round(npos, 4).tolist(), yaw_deg=round(nyaw, 1), across_flats_m=0.024, height_m=0.010,
                 note="hexagonal M16 nut (0.03 kg); pos is the centre of its bottom face; it rests on a small 18 mm post"),
            dict(name="bolt", kind="fixture", pos=np.round(bpos, 4).tolist(), shank_top_z=round(float(broot[2]) + 0.035, 4),
                 target_nut_bottom_z=round(float(broot[2]) + 0.032, 4), thread_pitch_m=0.002, shank_diameter_m=0.016,
                 note=("M16 bolt fixed upright on the table: a 10 mm base with a 25 mm threaded shank on top; pos is the bolt axis at "
                       "its base; right-handed thread: turning the nut clockwise seen from above screws it down")),
            dict(name="table", kind="surface", top_z=0.0)]


NUT_THREAD = FBTask(
    name="nut_thread",
    instruction=(NT_PROMPT + "\n\nTask details: positions are approximate (a few millimetres), so find the bolt by touch. Success "
                 "needs the nut centred on the bolt (within 2.5 mm), screwed down so that its bottom face is at target_nut_bottom_z "
                 "(between 2 mm below and 1 mm above), not tilted, released, and the gripper moved away (at least 5.5 cm above the "
                 "nut or 7 cm to the side). The nut sits about 8 mm below the tool centre point when grasped from above at mid-height. "
                 "press_turn presses down while turning (negative angle = clockwise seen from above). Then call done."),
    describe_scene=_nut_scene,
    extra=dict(table_top_z=0.0, fb_prompt=NT_PROMPT, fb_extra_tools=("press_turn",)),
)

TASKS = {t.name: t for t in (WEIGHT_SORT, RATTLE_SORT, FAUCET_TURN, FAUCET_FORCE_TURN, STOVE_TURN_ON, STOVE_PRESS_TURN, DRAWER, NUT_THREAD)}

# ------------------------------------------------------------------ RGB only (the paper's setting)
# No object or container coordinates: every turn the model gets the two camera images (enlarged, with a pixel grid), finds things
# itself and turns pixels into positions with locate (line of sight intersected with a plane of known height). Given are only the
# table height, object and container dimensions (object knowledge) and the rough camera orientation: the same sensing as
# ForceBench's trained policies (RGB, no depth), plus the camera calibration.
CAMERA_NOTE = ("main: a fixed camera beyond the far end of the table looking back at the robot (the robot is at the top of the image; "
               "the robot's left, +y, is on the image's right); wrist: a camera on the hand looking down past the fingers")


def _ws_scene_rgb(sim: FBSim) -> List[Dict[str, Any]]:
    return [dict(name="table", kind="surface", top_z=TABLE_TOP_Z),
            dict(name="cameras", kind="info", note=CAMERA_NOTE + ". There are no object coordinates: find objects in the images "
                 "and use locate(camera, u, v, z) to turn a pixel into a world position on the horizontal plane at height z.")]


WEIGHT_SORT_RGB = FBTask(
    name="weight_sort",
    instruction=(WS_PROMPT + f"\n\nTask details: the box is either LIGHT (about {WS_LIGHT} kg) or HEAVY (about {WS_HEAVY} kg); the two "
                 "types look identical. The box is a flat cardboard box 76 mm long, 54 mm wide and 30 mm tall, lying on the table "
                 "(its top face is at z = 0.03 m) with its long side along world x. The black and the blue container are open "
                 "boxes with 7.5 cm tall walls and an opening 16 cm along x and 10 cm along y; their floor is at the table surface. "
                 "No positions are given: find the box and the containers in the camera images and use locate to get their world "
                 "positions (choose z = the height of the surface under the pixel, e.g. 0.03 for the top of the box). The wrist "
                 "camera gives a much finer view once the hand is above the box. Lower the box into the matching container, release "
                 "it there and move the gripper up, then call done with your verdict (light or heavy)."),
    describe_scene=_ws_scene_rgb,
    extra=dict(table_top_z=TABLE_TOP_Z, light_kg=WS_LIGHT, heavy_kg=WS_HEAVY, fb_prompt=WS_PROMPT, perception="rgb",
               fb_extra_tools=("locate",), box_top_z=0.0331),
)

# RGB-only versions of the other seven tasks, on the same principle: no horizontal position of any object or fixture; only
# dimensions, the heights of key parts above the table (needed to turn a pixel into a position without a depth camera; the
# equivalent of a data sheet), the type of mechanism, and where its axis lies relative to the visible parts.
# CAMERA_NOTE_V2 adds the orientation of the wrist camera (beyond the fingertips, looking back toward the robot at about 30 deg)
# to CAMERA_NOTE; weight_sort keeps the older CAMERA_NOTE because its paper runs used it.
CAMERA_NOTE_V2 = ("main: a fixed camera above the far side of the table looking back at the robot (the robot is at the top of the image; "
                  "the robot's left, +y, is on the image's right). wrist: a camera on the hand, about 6 cm beyond the fingertips (+x) and "
                  "9 cm above them, looking down and back toward the robot (about 30 degrees from vertical) when the gripper points "
                  "down; in its image up is +x (away from the robot) and right is -y (the robot's right). The point straight below the "
                  "fingertips appears in the upper half of the wrist image; to see an object in the middle of the wrist image, stop "
                  "about 7 cm beyond it (larger x) and 10-15 cm above it. The wrist camera turns with the hand (e.g. after point_tool).")


def _rgb_scene(sim: FBSim) -> List[Dict[str, Any]]:
    return [dict(name="table", kind="surface", top_z=TABLE_TOP_Z),
            dict(name="cameras", kind="info", note=CAMERA_NOTE_V2 + " There are no object coordinates: find objects in the images "
                 "and use locate(camera, u, v, z) to turn a pixel into a world position on the horizontal plane at height z "
                 "(for a vertical surface use plane = 'x' or 'y' with value = that coordinate).")]


_NO_POS = ("No positions are given: find {what} in the camera images and use locate to get world positions (choose the height of the "
           "surface under the pixel); the wrist camera gives a much finer view once the hand is near.")

RS_TOP_Z, RS_BASKET_FLOOR_Z, RS_BASKET_RIM_Z, RS_BELT_Z = 0.0233, 0.0327, 0.105, 0.018
RATTLE_SORT_RGB = FBTask(
    name="rattle_sort",
    instruction=(RS_PROMPT + f"\n\nTask details: the parcel weighs {RS_MASS} kg. Its centre of mass is either UNIFORM (at the geometric "
                 f"centre) or IMBALANCED (shifted {RS_OFFSET * 100:.0f} cm along the long axis); the two look identical. The parcel is a "
                 "flat cardboard box 120 mm long, 54 mm wide and 23 mm tall lying on the table (top face at z = 0.023 m) with its long "
                 "side along world x; pick it up at its geometric centre (an off-centre grasp of 1 cm along the long axis produces about "
                 "as much torque as half the imbalance). The black basket is an open box (walls about 10.5 cm tall, opening 23.5 cm "
                 "along x and 14.5 cm along y, inner floor at z = 0.033 m). The conveyor is a long blue belt (surface at z = 0.018 m, "
                 "28 cm wide along x) that starts beside the table and runs away along +y; put a uniform parcel on its near end: within "
                 "about 20 cm of the belt end closest to the robot's centre line, and at least 3 cm from the belt's side edges. "
                 + _NO_POS.format(what="the parcel, the basket and the conveyor") + " Decide which type it is, put it in the matching "
                 "place, release it there and move the gripper up, then call done with your verdict (imbalanced or uniform)."),
    describe_scene=_rgb_scene,
    extra=dict(table_top_z=TABLE_TOP_Z, mass_kg=RS_MASS, com_offset_m=RS_OFFSET, fb_prompt=RS_PROMPT, perception="rgb",
               fb_extra_tools=("locate",)),
)

# faucet: rod top 92 mm above the hinge (rod centre 59 mm above it, rod 66 mm long); the hinge is below the countertop (z = -0.047); rod top at z ~ 0.045.
FT_ROD_TOP_Z, FT_ROD_TOP_ABOVE_PIVOT = 0.045, 0.092
_FT_LEVER_RGB = ("The lever is a thin upright metal rod (10 mm thick, rounded top end) sticking up out of the faucet's square valve head, "
                 "at the end of a short horizontal pipe in the sink; its top end is about 4.5 cm above the countertop (z = 0.045 m). "
                 "It is hinged about a horizontal axis along world y that passes 9.2 cm below the rod's top end (directly below it "
                 "while the rod is upright; the hinge itself is hidden inside the valve head), so the turn pivot is (rod x, rod y, "
                 "0.045 - 0.092).")
FAUCET_TURN_RGB = FBTask(
    name="faucet_turn",
    instruction=(FT_PROMPT + "\n\nTask details: " + _FT_LEVER_RGB + " The water comes on when the top of the lever moves toward the "
                 "robot (-x), i.e. a NEGATIVE rotation about +y (right-hand rule); it turns 0 to 110 degrees that way. It has a detent - "
                 "the resistance rises, then suddenly drops - somewhere between 20 and 30 degrees from upright; the water turns on once "
                 "the lever is just past the detent. " + _NO_POS.format(what="the lever") + " Grasp the rod near its top (fingertips "
                 "about 2 mm below the top end, fingers closing across the rod), turn it about its hinge (turn tool, axis [0, 1, 0]) "
                 "toward the robot past the detent, then call done."),
    describe_scene=_rgb_scene,
    extra=dict(table_top_z=0.0, fb_prompt=FT_PROMPT, extra_tools=("turn",), perception="rgb", fb_extra_tools=("locate",),
               rod_top_z=FT_ROD_TOP_Z, grasp_below_top=0.002),
)
FAUCET_FORCE_TURN_RGB = FBTask(
    name="faucet_force_turn",
    instruction=(FFT_PROMPT + "\n\nTask details: " + _FT_LEVER_RGB + " It can be turned either way about its hinge, up to about 45 "
                 "degrees each way: a NEGATIVE angle about +y moves the top of the lever toward the robot (-x), a POSITIVE angle moves "
                 "it away (+x). One direction is clearly easier (lower resistance) than the other. The water turns on only when the "
                 "lever is turned past its detent - the resistance rises, then suddenly drops - somewhere between 20 and 30 degrees in "
                 "the EASIER direction. " + _NO_POS.format(what="the lever") + " Grasp the rod 2.7 cm below its top end (6.5 cm above "
                 "the hinge, fingers closing across the rod), find the easier direction, turn past the detent, then release the lever "
                 "and move the gripper up to at least z = 0.11 m, leaving the lever still. Call done with the easier direction as "
                 "verdict ('toward_robot' or 'away_from_robot')."),
    describe_scene=_rgb_scene,
    extra=dict(table_top_z=0.0, fb_prompt=FFT_PROMPT, extra_tools=("turn",), min_final_tcp_z=0.108, perception="rgb",
               fb_extra_tools=("locate",), rod_top_z=FT_ROD_TOP_Z, grasp_below_top=0.027),
)

# stove knob: root (base) at z 0.0205, 19.2 mm tall -> top face at z ~ 0.040; ForceBench's expert grasps 2 mm above the base (1 mm below it on Stove Press).
ST_KNOB_TOP_Z, ST_KNOB_BASE_Z = 0.0397, 0.0205
_ST_KNOB_RGB = ("The centre knob is a 38 mm wide, 19 mm tall silver cylinder standing on the dark cooktop, with a white line on its top; "
                "its top face is at about z = 0.040 m and its base at z = 0.0205 m. ")
STOVE_TURN_ON_RGB = FBTask(
    name="stove_turn_on",
    instruction=(ST_PROMPT + "\n\nTask details: " + _ST_KNOB_RGB + "It turns about the vertical axis through its centre; "
                 "counter-clockwise seen from above (positive angle) turns it on; range 0 to 120 degrees. It has an ignition detent - "
                 "the resistance rises gradually, then suddenly drops - somewhere between 60 and 90 degrees; the burner ignites just past "
                 "it (a few degrees past is fine, far past is not needed). " + _NO_POS.format(what="the knob") + " Grasp it from above, "
                 "centred, with the fingertips about 2 mm above its base (fingers closing across it), turn it counter-clockwise past "
                 "the detent, then call done."),
    describe_scene=_rgb_scene,
    extra=dict(table_top_z=0.0, fb_prompt=ST_PROMPT, perception="rgb", fb_extra_tools=("locate",), grasp_above_base=0.002),
)
# Facts about the force reading only, so this note appears only with the force interface (_NO_FORCE_EDITS removes it); the
# no-force instruction is otherwise word-for-word the same.
_SPT_READING_NOTE = ("The wrist torque reading is noisy, so each press_turn report lists the resistance measured so far on each side of the knob's "
                     "start angle (only while moving away from the start, 8 to 25 degrees out): turn past 25 degrees on both sides, and the side "
                     "with the lower value is the easier direction. Do not expect to see the sudden drop in the reading. A force_cap stop during "
                     "a turn is usually a brief spike in the reading, not a mechanical limit: repeat the turn. ")
STOVE_PRESS_TURN_RGB = FBTask(
    name="stove_press_turn",
    instruction=(SPT_PROMPT + "\n\nTask details: " + _ST_KNOB_RGB + "It only turns while it is pressed down by at least 7 mm (about "
                 "7 N against its spring; it bottoms out at 8 mm) - use press_turn, which keeps pressing while it turns. It can turn "
                 "either way (up to 120 degrees), but one direction is clearly easier (lower resistance). The burner lights only when "
                 "the knob is turned past its detent - the resistance rises, then suddenly drops - somewhere between 60 and 90 degrees "
                 "in the EASIER direction while pressed. " + _SPT_READING_NOTE + _NO_POS.format(what="the knob") + " Grasp it from above, centred, with "
                 "the fingertips level with its base (fingers closing across it), find the easier direction, turn past the detent, "
                 "then call done with the direction as verdict ('ccw' or 'cw', seen from above)."),
    describe_scene=_rgb_scene,
    extra=dict(table_top_z=0.0, fb_prompt=SPT_PROMPT, perception="rgb", fb_extra_tools=("locate", "press_turn"), grasp_above_base=-0.001, turn_sides=True),
)

# drawer: centre of the middle handle (drawer closed) at z ~ 0.219; the cabinet front faces +y.
DR_HANDLE_Z = float(DR_HANDLE_CLOSED[2])
DRAWER_RGB = FBTask(
    name="radmanso_drawer_sort_table",
    instruction=(DR_PROMPT + "\n\nTask details: the walnut cabinet stands at the robot's right side of the table with its front facing "
                 "+y (toward the robot's centre line); it has three drawers, each with a round brass knob handle sticking out about 2 cm "
                 "toward +y. The target is the MIDDLE drawer's handle, whose centre is about 22 cm above the table (z = 0.219 m). Grip "
                 "it from the +y side with the tool lying flat and the fingers pointing toward -y (point_tool [0, -1, 0]), so that the "
                 "fingers close up and down on it, with the fingertips at the handle's centre; the drawer slides out toward +y. "
                 + _NO_POS.format(what="the handle") + " After point_tool the wrist camera looks along -y at the cabinet front: locate "
                 "a pixel there with plane = 'y' and value = the handle's y (instead of a height z). The drawer slides with only about 0.35 N of resistance for the "
                 "first few centimetres, then the resistance rises steeply over 1 to 3 cm. Stop where it first reaches about 2.4 N and "
                 "hold the drawer still there for at least 1 second (the target is within 2.5 mm of that point). Pulling through the "
                 "steep part (a peak of several newtons) breaks the latch and fails the task, and the drawer closes again if you let go. "
                 "Then call done."),
    describe_scene=_rgb_scene,
    extra=dict(table_top_z=0.0, fb_prompt=DR_PROMPT, extra_tools=("point_tool", "pull", "push_click"), perception="rgb",
               fb_extra_tools=("locate",), handle_z=DR_HANDLE_Z),
)

# nut: bottom face at z 0.010 (on the post), 10 mm tall -> top face at z 0.020; bolt root on the table, shank top at z 0.035,
# goal for the nut bottom at z 0.032.
NT_NUT_TOP_Z, NT_NUT_BOTTOM_Z, NT_SHANK_TOP_Z, NT_TARGET_BOTTOM_Z = 0.020, 0.010, 0.035, 0.032
NUT_THREAD_RGB = FBTask(
    name="nut_thread",
    instruction=(NT_PROMPT + "\n\nTask details: the M16 nut (hexagonal, 24 mm across flats, 10 mm tall, 0.03 kg) rests on a small post "
                 "near the robot, its top face at z = 0.020 m (bottom face at z = 0.010 m). The bolt stands upright on the table, "
                 "somewhere further from the robot (its place changes from episode to episode): a 10 mm base with a 25 mm threaded "
                 "shank (16 mm thick) whose top is at z = 0.035 m; right-handed thread, 2 mm pitch: turning the nut clockwise seen from "
                 "above screws it down. " + _NO_POS.format(what="the nut and the bolt") + " Because the cameras look at an angle, locate on "
                 "these small tall objects is often a centimetre off; for the nut and the bolt use locate_object (click roughly on the "
                 "object, the harness fits its known shape). The nut you hold hides the middle of the wrist image, so while holding it, look at "
                 "the bolt with the hand directly above the bolt (the bolt then shows in the upper half of the image). Success needs the nut "
                 "centred on the bolt (within 2.5 mm), "
                 "screwed down until its bottom face is at z = 0.032 m (between 2 mm below and 1 mm above), not tilted, released, and "
                 "the gripper moved away (at least 5.5 cm above the nut or 7 cm to the side). Heights: close the fingers with the tool centre "
                 "point at about z = 0.018 m (from 0.021 m up they catch only the top edge or miss it). The nut then stays fixed in the fingers, so with G = the tool "
                 "height at which you grasped, the nut's bottom touches the top of the shank when the tool is at G + 0.025 m and is at the goal "
                 "z = 0.032 m when the tool is at G + 0.022 m. Stop there and release; going deeper fails. Sideways: fast moves end a few mm "
                 "off the command, so make the last move slowly (speed at most 0.02 m/s) and read the end position it reports. The fingers close "
                 "along y, so in x the nut keeps whatever offset it had from the tool when they closed: line up the nut with the bolt, not the "
                 "tool. Centred within about 1 mm, the nut drops 2-3 mm past the top of the shank when pressed lightly, onto the start of "
                 "the thread; if the tool stops at G + 0.025 m the nut is sitting on the shank top off-centre, so lift and correct instead of "
                 "pressing harder. Turning: press_turn presses down with a set force while turning; move_turn holds a commanded height while "
                 "turning (negative angle = clockwise seen from above; the nut goes down 2 mm per full turn). The wrist can only turn about "
                 "150 degrees clockwise from its starting orientation and turning further pushes the hand sideways, so turn at most -150 degrees "
                 "in total, at about 30 deg/s. Then call done."),
    describe_scene=_rgb_scene,
    # locate_object is added for this task, and the force interface also gets move_turn so that its motions are a superset of the
    # no-force control's. The shapes match the dimensions in the instruction (the hexagon as a cylinder of radius 13 mm, between
    # its 12 mm inner and 13.9 mm outer radius). Heights, offsets, chamfer and wrist limit in the instruction are facts of the
    # task, the same in every episode, with no per-episode truth. (An earlier wording put the nut bottom 8 mm below the TCP for a
    # mid-height grasp; the true value is 5 mm, and a model following it screwed 5 mm too deep.)
    extra=dict(table_top_z=0.0, fb_prompt=NT_PROMPT, perception="rgb", fb_extra_tools=("locate", "locate_object", "press_turn", "move_turn"),
               object_models={"nut": [(0.013, NT_NUT_BOTTOM_Z, NT_NUT_TOP_Z), (0.009, 0.0, NT_NUT_BOTTOM_Z)],
                              "bolt": [(0.012, 0.0, 0.010), (0.008, 0.010, NT_SHANK_TOP_Z)]}),
)

TASKS_RGB = {t.name: t for t in (WEIGHT_SORT_RGB, RATTLE_SORT_RGB, FAUCET_TURN_RGB, FAUCET_FORCE_TURN_RGB, STOVE_TURN_ON_RGB,
                                 STOVE_PRESS_TURN_RGB, DRAWER_RGB, NUT_THREAD_RGB)}

# No-force control ("none|pose"): tools named in the instruction must match its tool menu; everything else stays word-for-word the
# same (both conditions get the same task knowledge).
_NO_FORCE_EDITS = {
    "stove_press_turn": [("use press_turn, which keeps pressing while it turns", "use move_turn, which keeps the hand pressed down while it turns"),
                         (_SPT_READING_NOTE, "")],
    "nut_thread": [("press_turn presses down with a set force while turning; move_turn holds a commanded height while turning",
                    "move_turn holds a commanded height while turning")],
}


def for_interface(task: FBTask, interface: str) -> FBTask:
    """Return the task unchanged for the force interface; for other interfaces, replace the mentions of force-only tools in the
    instruction (the tool menu itself is set by FBEpisode)."""
    if interface == "force" or task.extra.get("perception") != "rgb" or task.name not in _NO_FORCE_EDITS:
        return task
    text = task.instruction
    for old, new in _NO_FORCE_EDITS[task.name]:
        if old not in text:
            raise ValueError(f"{task.name}: instruction no longer contains {old!r}")
        text = text.replace(old, new)
    return dataclasses.replace(task, instruction=text)


def get_fb_task(name: str, perception: str = "privileged") -> FBTask:
    if perception == "rgb":
        if name not in TASKS_RGB:
            raise KeyError(f"task {name!r} has no RGB-only adapter yet (have: {sorted(TASKS_RGB)})")
        return TASKS_RGB[name]
    return TASKS[name]
