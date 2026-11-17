"""JSON schemas of the tools (primitives) offered to the model, filtered by action interface.

The descriptions do not oversell: they state each primitive's real termination conditions.  They are part of what the
model sees, so their text is kept exactly as used in the experiments.
"""
from __future__ import annotations

from typing import Any, Dict, List

from forceharness.primitives.library import FORCE_ONLY_EXTRA, INTERFACES

_num = {"type": "number"}
_vec3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}

_ALL: Dict[str, Dict[str, Any]] = {
    "scene": dict(description="Get the current scene description (object/zone poses in the world frame, metres; poses are approximate) "
                              "and the robot state.", parameters={"type": "object", "properties": {}}),
    "move_to": dict(description="Move the tool centre point to an absolute world position (x, y, z in metres), orientation unchanged. "
                                "{guard_note}",
                    parameters={"type": "object", "properties": {"x": _num, "y": _num, "z": _num,
                                                                 "speed": {"type": "number", "description": "m/s, optional (default 0.10 free / 0.03 near contact)"}},
                                "required": ["x", "y", "z"]}),
    "move_by": dict(description="Move the tool centre point by a relative offset (dx, dy, dz in metres, world frame). {guard_note}",
                    parameters={"type": "object", "properties": {"dx": _num, "dy": _num, "dz": _num, "speed": _num}, "required": ["dx", "dy", "dz"]}),
    "rotate_z": dict(description="Rotate the tool about the vertical axis by angle_deg (positive = counter-clockwise seen from above). "
                                 "{rotate_note}",
                     parameters={"type": "object", "properties": {"angle_deg": _num,
                                                                  "torque_cap": {"type": "number", "description": "Nm; stop if exceeded (force interface only)"},
                                                                  "stop_on_resistance_drop": {"type": "boolean", "description": "stop when resistance peaks then drops (force interface only)"}},
                                 "required": ["angle_deg"]}),
    "move_until_contact": dict(description="Move slowly along a direction vector until the wrist senses contact (|F| >= f_threshold N) or max_dist metres "
                                           "is travelled. Returns where contact happened.",
                               parameters={"type": "object", "properties": {"direction": _vec3, "max_dist": _num,
                                                                            "f_threshold": {"type": "number", "description": "N, default 2"}},
                                           "required": ["direction", "max_dist"]}),
    "push": dict(description="Press along a direction and regulate the contact force to target_force newtons for duration seconds "
                             "(admittance control; stays in place if already at the target force).",
                 parameters={"type": "object", "properties": {"direction": _vec3, "target_force": _num, "duration": _num}, "required": ["direction", "target_force"]}),
    "slide": dict(description="Slide along a surface: keep normal_force newtons against the surface (normal defaults to -z, i.e. pressing down) "
                              "while moving distance metres along direction (tangential).",
                  parameters={"type": "object", "properties": {"direction": _vec3, "distance": _num, "normal_force": _num, "normal": _vec3},
                              "required": ["direction", "distance"]}),
    "search": dict(description="Insertion search around the current position (start with the held part on or just above the surface: the "
                               "primitive first descends up to 3 cm to find support, result no_surface if nothing is there). mode='hop' (default): visit spiral sample points "
                               "(spacing pitch, out to radius_max) and at each one lift by hop, move sideways, and probe gently along "
                               "normal (default -z) until contact; mode='slide': press with normal_force and spiral while sliding. Stops "
                               "with result 'dropped' as soon as the tool advances more than drop metres along the normal (the part fell "
                               "into the hole); 'completed' means the whole radius was searched without a drop. mode='slide' is force-guided: when a corner catches on "
                               "the rim it lifts and moves along the wall's reaction toward the hole centre; results also include stuck (sideways motion blocked) "
                               "and out_of_range. After dropped or stuck, call insert to seat the part.",
                   parameters={"type": "object", "properties": {"normal": _vec3, "radius_max": _num, "pitch": _num, "normal_force": _num,
                                                                "drop": _num, "mode": {"type": "string", "enum": ["hop", "slide"]}, "hop": _num,
                                                                "speed": {"type": "number", "description": "m/s lateral speed, default 0.02"}}}),
    "insert": dict(description="Compliant insertion along normal (default downward): the tool advances slowly, slows down as the reaction force "
                               "approaches max_force (newtons) and lets the held part self-centre sideways. Stops when it has advanced by depth metres "
                               "(result seated), when it jams (force at the limit with no progress; result jammed), or at the safety limit. Use after the "
                               "part has found the hole (e.g. search reported dropped).",
                   parameters={"type": "object", "properties": {"depth": _num, "normal": _vec3,
                                                                "max_force": {"type": "number", "description": "N, default 12"}}}),
    # kept purely geometric: the wording "use it to grasp a horizontal bar from the side" was ambiguous (models pointed
    # the fingers along the bar)
    "point_tool": dict(description="Re-orient the tool so the fingers point along direction (a vector; the tool centre point stays where it is). "
                                   "[0, 0, -1] is the default straight-down orientation. A horizontal direction lays the tool flat with the "
                                   "fingers pointing that way and opening vertically (up/down). To grasp a bar, point the fingers "
                                   "perpendicular to the bar so that the bar ends up between the fingers. "
                                   "{guard_note}",
                       parameters={"type": "object", "properties": {"direction": _vec3}, "required": ["direction"]}),
    "turn": dict(description="Rotate the hand rigidly by angle_deg about the axis that passes through pivot with direction axis (right-hand "
                             "rule): the tool centre point moves on the circle around the axis and the tool turns with it - use it to turn a lever "
                             "or knob you are holding about its own axis. {rotate_note}",
                 parameters={"type": "object", "properties": {"axis": _vec3, "pivot": _vec3, "angle_deg": _num,
                                                              "rate_deg_s": {"type": "number", "description": "deg/s, default 10"},
                                                              "torque_cap": {"type": "number", "description": "Nm; stop if exceeded (force interface only)"},
                                                              "stop_on_resistance_drop": {"type": "boolean", "description": "stop when resistance peaks and then drops - a 'click' (force interface only)"}},
                             "required": ["axis", "pivot", "angle_deg"]}),
    "pull": dict(description="Pull something you are holding that is attached to a mechanism (door handle, drawer, lever). The pulling force along "
                             "direction is regulated to force newtons while the hand gives way sideways and turns about the vertical axis, and the "
                             "pulling direction adapts to the way the mechanism actually moves - so the hand follows the path the mechanism allows "
                             "(an arc for a hinged door, a straight line for a drawer) even if direction is only roughly right. Stops after distance "
                             "metres along the path (completed), when the mechanism stops moving (stalled: at its limit, locked or slipping), or at the "
                             "safety limit. Reports how far the hand moved, how much the path turned, the pivot it turned about, and the forces.",
                 parameters={"type": "object", "properties": {"direction": _vec3, "force": {"type": "number", "description": "N, default 8"},
                                                              "distance": {"type": "number", "description": "m along the path, default 0.10"},
                                                              "speed": {"type": "number", "description": "m/s, default 0.03"}},
                             "required": ["direction"]}),
    "lift": dict(description="Raise the tool vertically by height metres, then hold still for hold_time seconds while measuring the wrist "
                             "force (use after grasping to weigh an object).",
                 parameters={"type": "object", "properties": {"height": _num, "hold_time": _num}}),
    "probe": dict(description="Tap: move along direction until contact (at most amplitude metres) and return to the start. Reports whether "
                              "and where contact was found.",
                  parameters={"type": "object", "properties": {"direction": _vec3, "amplitude": _num, "f_threshold": _num}}),
    "grasp": dict(description="Close the gripper. Reports the final opening (whether something is between the fingers).",
                  parameters={"type": "object", "properties": {}}),
    "release": dict(description="Open the gripper.", parameters={"type": "object", "properties": {}}),
    "hold": dict(description="Stay still for seconds seconds (measures the wrist signal while holding).",
                 parameters={"type": "object", "properties": {"seconds": _num}}),
    "done": dict(description="Declare the episode finished. success=true only if you believe the task is complete. verdict is the task-specific "
                             "answer when one is asked for (e.g. 'heavy' / 'light'), otherwise empty.",
                 parameters={"type": "object", "properties": {"success": {"type": "boolean"}, "verdict": {"type": "string"},
                                                              "note": {"type": "string"}}, "required": ["success"]}),
}

GUARD_NOTES = {
    "pose": "The motion completes the full commanded displacement unless a safety force limit is hit.",
    "guard": "The motion STOPS EARLY if the tool makes contact with anything (contact guard) and reports it.",
    "force": "The motion STOPS EARLY if the tool makes contact with anything (contact guard) and reports it.",
}
ROTATE_NOTES = {
    "pose": "Rotation completes fully unless a safety torque limit is hit.",
    "guard": "Rotation completes fully unless a safety torque limit is hit.",
    "force": "Optionally stop when torque exceeds torque_cap or when resistance peaks and then drops.",
}


# task switch "push_click" (used by ForceBench's Drawer Pull; first written for an earlier MuJoCo snap-fit task): the push
# description and parameters gain the constant-speed advance, max_dist and the stop on a "click"; for every other task the
# push description stays word for word the same
PUSH_CLICK_DESC = (" While the resistance is below target_force it advances at about 2 cm/s (at most max_dist metres, default 0.05). With "
                   "stop_on_resistance_drop=true it stops as soon as the resistance peaks and then clearly drops (a 'click', e.g. a latch giving way).")


def tool_schemas(interface: str, has_gripper: bool = True, extra_tools=()) -> List[Dict[str, Any]]:
    """Tool schemas for an action interface.  extra_tools: primitives a task opens in addition (inserted before done);
    empty by default, which gives the standard tool list.  "push_click" is a description switch, not a tool."""
    flags = {e for e in extra_tools if e == "push_click"}
    extra_tools = [e for e in extra_tools if e not in flags]
    names = list(INTERFACES[interface])
    for e in extra_tools:
        if e in FORCE_ONLY_EXTRA and interface != "force":      # primitives that close their loop on force readings exist only under the force interface
            continue
        if e not in names:
            names.insert(names.index("done") if "done" in names else len(names), e)
    out = []
    for n in names:
        if not has_gripper and n in ("grasp", "release", "lift"):
            continue
        t = _ALL[n]
        desc = t["description"].format(guard_note=GUARD_NOTES[interface], rotate_note=ROTATE_NOTES[interface])
        params = t["parameters"]
        if n == "push" and "push_click" in flags:
            desc += PUSH_CLICK_DESC
            params = dict(params, properties={**params["properties"], "max_dist": _num, "stop_on_resistance_drop": {"type": "boolean"}})
        out.append(dict(name=n, description=desc, parameters=params))
    return out
