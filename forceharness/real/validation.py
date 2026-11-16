"""Validation of every model call before anything moves (layer 2): menu, arguments, ranges, and reach.

validate() returns the cleaned arguments or raises ValueError with a reason the model reads in its tool result
("Invalid; not executed: ..."); nothing has moved then. Besides the argument types and the ranges stated in the
menu, it walks the planned path with the FR3 kinematic model (checks.py) and refuses motions the arm cannot
follow, giving the largest feasible amount. In condition B, force-only arguments are dropped and recorded.
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation as R

from forceharness.real import scene_objects
from forceharness.real.checks import (TURN_FLANGE_REACH_M, rotation_reach, translation_reach, turn_flange_chord,
                                      vector)
from forceharness.real.geometry import check_flange_workspace
from forceharness.real.menu import (BOOL, CONDITIONS, CONTACT_SEARCH_MAX_M, FORCE_LIMITS, FORCE_ONLY_ARGS, MODE,
                                    MOVE_MAX_M, PRIMITIVES, PULL_FORCE_DEFAULT_N, PULL_FORCE_MAX_N, PUSH_FORCE_MAX_N,
                                    ROT_MAX_DEG, ROTATE_Z_MAX_DEG, SLIDE_FORCE_DEFAULT_N, TEXT, VEC2, VEC3, signature)
from forceharness.real.primitives import (SCREW_PUSH_MAX_N, WIGGLE_DEPTH_MAX_M, WIGGLE_SHIFT_MAX_M, WIGGLE_TIME_MAX_S,
                                          WIGGLE_YAW_MAX_DEG)


def grasp_reach(backend, found):
    """Whether the arm can make each object's suggested grasp rotate_z from the current hand pose,
    and the equivalent turn the other way (+-180 deg, the gripper is symmetric) when it cannot."""
    for o in (found or {}).get('objects', []):
        g = o.get('grasp') or {}
        turn = g.get('rotate_z_deg')
        if turn is None or abs(turn) <= scene_objects.ALIGNED_DEG:
            continue
        pos = np.asarray(backend.eef_pos, dtype=float)
        feasible, why = rotation_reach(backend, [0, 0, 1], pos, turn, ROT_MAX_DEG)
        if why is None or feasible >= abs(turn)-.5:
            g['reach'] = dict(ok=True)
            continue
        other = turn-math.copysign(180., turn)
        other_ok, why_other = rotation_reach(backend, [0, 0, 1], pos, other, ROT_MAX_DEG)
        g['reach'] = dict(ok=False, max_deg=round(feasible, 1), why=why, alternative_deg=round(other, 1),
                          alternative_ok=bool(why_other is None or other_ok >= abs(other)-.5))
    return found


def unit(v, name='direction'):
    a = vector(v)
    n = float(np.linalg.norm(a))
    if n < 1e-9:
        raise ValueError(name+' cannot be a zero vector')
    return a/n


def _range(args, key, lo, hi, default=None, *, closed_lo=False):
    value = args.get(key, default)
    if value is None:
        return None
    ok = (lo <= value <= hi) if closed_lo else (lo < value <= hi)
    if not ok:
        raise ValueError(f'{key} must be in {"[" if closed_lo else "("}{lo:g}, {hi:g}] (this time {value:g})')
    return value


def _reach_check(backend, delta, what, segment_m=None):
    feasible, why = translation_reach(backend, delta, segment_m=segment_m)
    if why and feasible < float(np.linalg.norm(delta))-.002:
        raise ValueError(f'{why}: at most about {feasible*1000:.0f} mm more is possible in this direction; {what}')


def validate(name, args, backend, condition, limits=None, *, conditions=None, primitives=None):
    """(clean args, ignored arg names) or ValueError with the reason; nothing has moved yet."""
    limits = limits or FORCE_LIMITS
    primitives = primitives or PRIMITIVES
    menu = (conditions or CONDITIONS)[condition]['menu']
    if not isinstance(name, str) or name not in menu or not isinstance(args, dict):
        raise ValueError('Action not on the menu. Menu: '+', '.join(menu))
    spec = primitives[name]
    ignored = []
    if condition == 'B':
        ignored = sorted(set(args) & FORCE_ONLY_ARGS)
        args = {k: v for k, v in args.items() if k not in FORCE_ONLY_ARGS}
    unknown, missing = set(args)-set(spec['args']), set(spec['req'])-set(args)
    if unknown or missing:
        raise ValueError(f'Wrong arguments: {signature(name, condition, primitives)}'
                         + (f'; unexpected {sorted(unknown)}' if unknown else '')
                         + (f'; missing {sorted(missing)}' if missing else ''))
    for key, value in args.items():
        kind = spec['args'][key]
        if kind == VEC3:
            vector(value)
        elif kind == VEC2:
            if (not isinstance(value, (list, tuple)) or len(value) != 2
                    or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value)):
                raise ValueError(key+' must be two finite numbers [dx, dy] (metres)')
        elif kind == BOOL:
            if type(value) is not bool:
                raise ValueError(key+' must be true/false')
        elif kind == TEXT:
            if not isinstance(value, str) or len(value) > 2000:
                raise ValueError(key+' must be text')
        elif kind == MODE:
            if value not in ('hop', 'slide'):
                raise ValueError('mode must be "hop" or "slide"')
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(key+' must be a finite number')
    pos = np.asarray(backend.eef_pos, dtype=float)
    if name == 'locate':
        if any(type(args[k]) is not int or args[k] < 0 for k in ('frame', 'u', 'v')):
            raise ValueError('frame, u and v must be non-negative integers')
    elif name in ('move_to', 'move_by'):
        goal = (np.array([args['x'], args['y'], args['z']], dtype=float) if name == 'move_to' else
                pos+np.array([args['dx'], args['dy'], args['dz']], dtype=float))
        dist = float(np.linalg.norm(goal-pos))
        if dist > MOVE_MAX_M:
            raise ValueError(f'at most {MOVE_MAX_M:.2f} m per move (this time {dist:.3f} m): split it into several moves')
        _range(args, 'speed', 0, .008)
        try:
            check_flange_workspace(np.asarray(backend._state['flange_pos'], dtype=float)+goal-pos)
        except ValueError:
            raise ValueError('the hand flange position for this target is outside the workspace box (x 0.20–0.65, '
                             '|y| < 0.30, z 0.12–0.55 m; with the fingers pointing down the flange is about 0.12 m '
                             'above the TCP)') from None
        _reach_check(backend, goal-pos, 'choose another target or split the move')
    elif name == 'rotate_z':
        _range({'|angle_deg|': abs(args['angle_deg'])}, '|angle_deg|', 0, ROTATE_Z_MAX_DEG)
        feasible, why = rotation_reach(backend, [0, 0, 1], pos, args['angle_deg'], ROT_MAX_DEG)
        if why and feasible < abs(args['angle_deg'])-.5:
            raise ValueError(f'{why}: at most about {feasible:.0f}° more is possible in this direction')
    elif name == 'screw':
        _range({'|angle_deg|': abs(args['angle_deg'])}, '|angle_deg|', 0, ROTATE_Z_MAX_DEG)
        _range(args, 'push', 0, SCREW_PUSH_MAX_N)
        feasible, why = rotation_reach(backend, [0, 0, 1], pos, args['angle_deg'], ROT_MAX_DEG)
        if why and feasible < abs(args['angle_deg'])-.5:
            raise ValueError(f'{why}: at most about {feasible:.0f}° more is possible in this direction')
    elif name == 'point_tool':
        a = unit(args['direction'])
        z = R.from_quat(backend.eef_quat).apply([0, 0, 1])
        angle = math.degrees(math.acos(float(np.clip(z@a, -1, 1))))
        if angle > ROT_MAX_DEG+.5:
            raise ValueError(f'point_tool turns at most {ROT_MAX_DEG:.0f}° per call (this time {angle:.0f}°): '
                             'split it into several turns')
        if angle > .5:
            feasible, why = rotation_reach(backend, np.cross(z, a), pos, angle)
            if why and feasible < angle-.5:
                raise ValueError(f'{why}: at most about {feasible:.0f}° more is possible toward this direction')
    elif name == 'turn':
        unit(args['axis'], 'axis')
        _range({'|angle_deg|': abs(args['angle_deg'])}, '|angle_deg|', 0, ROT_MAX_DEG)
        chord, limit_deg = turn_flange_chord(backend, args)
        if chord > TURN_FLANGE_REACH_M:
            raise ValueError(f'this turn would move the hand flange by {chord*1000:.0f} mm (≤ '
                             f'{TURN_FLANGE_REACH_M*1000:.0f} mm per call): with this pivot turn at most '
                             f'{limit_deg:.0f}° per call, in several calls')
        feasible, why = rotation_reach(backend, args['axis'], args['pivot'], args['angle_deg'])
        if why and feasible < abs(args['angle_deg'])-.5:
            raise ValueError(f'{why}: from the current pose at most about {feasible:.0f}° more is possible in this '
                             'direction' + (' (none: already at the limit)' if feasible < 1
                                            else f'; use angle_deg ≤ {feasible:.0f}'))
    elif name in ('move_until_contact', 'probe'):
        u = unit(args['direction'])
        search = name == 'move_until_contact'
        dist = _range(args, 'max_dist' if search else 'amplitude', 0, CONTACT_SEARCH_MAX_M if search else MOVE_MAX_M, .02)
        _range(args, 'f_threshold', 0, limits['contact'])
        _reach_check(backend, u*dist, 'make max_dist / amplitude smaller', MOVE_MAX_M if search else None)
    elif name == 'push':
        u = unit(args['direction'])
        _range(args, 'target_force', 0, max(limits['contact'], PUSH_FORCE_MAX_N))
        _range(args, 'duration', 0, 30)
        _reach_check(backend, u*_range(args, 'max_dist', 0, MOVE_MAX_M, .05), 'make max_dist smaller')
    elif name == 'slide':
        u = unit(args['direction'])
        n = unit(args.get('normal', [0, 0, -1]), 'normal')
        if abs(float(u@n)) > .9:
            raise ValueError("slide's direction must run along the surface (roughly perpendicular to normal)")
        if args.get('normal_force') is None:
            args = dict(args, normal_force=limits.get('slide_default', SLIDE_FORCE_DEFAULT_N))
        _range(args, 'normal_force', 0, limits['contact'])
        _reach_check(backend, u*_range(args, 'distance', 0, MOVE_MAX_M), 'make distance smaller')
    elif name == 'search':
        n = unit(args.get('normal', [0, 0, -1]), 'normal')
        for key, hi in (('radius_max', .015), ('pitch', .01), ('hop', .01), ('drop', .07)):
            _range(args, key, 0, hi)
        _range(args, 'normal_force', 0, limits['contact'])
        if 'guess' in args:
            if args.get('mode') != 'hop':
                raise ValueError('guess is used only with mode="hop"')
            if float(np.hypot(*args['guess'])) > .02:
                raise ValueError('guess is at most 0.02 m; if it is farther off, move_to there first')
            _reach_check(backend, [args['guess'][0], args['guess'][1], 0.], 'make guess smaller')
        _reach_check(backend, n*(.03+args.get('drop', .004)), 'choose another start')
    elif name == 'insert':
        n = unit(args.get('normal', [0, 0, -1]), 'normal')
        _range(args, 'max_force', 0, limits['insert'])
        _range(args, 'wiggle', 0, .005, closed_lo=True)
        _reach_check(backend, n*_range(args, 'depth', 0, MOVE_MAX_M, .02), 'make depth smaller')
    elif name == 'wiggle':
        _range(args, 'force', 0, limits['insert'])
        _range(args, 'yaw_deg', 0, WIGGLE_YAW_MAX_DEG, closed_lo=True)
        _range(args, 'shift', 0, WIGGLE_SHIFT_MAX_M, closed_lo=True)
        _range(args, 'duration', 0, WIGGLE_TIME_MAX_S)
        _reach_check(backend, [0, 0, -_range(args, 'depth', 0, WIGGLE_DEPTH_MAX_M, .008)], 'make depth smaller')
    elif name == 'lift':
        _range(args, 'hold_time', 0, 30)
        _reach_check(backend, [0, 0, _range(args, 'height', 0, MOVE_MAX_M, .02)], 'make height smaller')
    elif name == 'pull':
        u = unit(args['direction'])
        _range(args, 'force', 0, PULL_FORCE_MAX_N, PULL_FORCE_DEFAULT_N)
        _range(args, 'speed', 0, .004)
        _reach_check(backend, u*_range(args, 'distance', 0, MOVE_MAX_M, .02), 'make distance smaller')
    elif name == 'grasp':
        if backend._closed is not False:
            raise ValueError('the gripper is already closed' if backend._closed
                             else 'the gripper state is unknown; the gripper cannot be actuated')
        left = backend.grasps_left() if hasattr(backend, 'grasps_left') else None
        if left is not None and left <= 0:
            raise ValueError(f'the gripper has already closed {backend.port.max_trials} times in this episode (the most '
                             'the gripper allows per episode); no more grasps')
    elif name == 'release':
        if backend._closed is not True:
            raise ValueError('the gripper is already open' if backend._closed is False
                             else 'the gripper state is unknown; the gripper cannot be actuated')
    elif name == 'hold':
        _range(args, 'seconds', 0, 120)
    return args, ignored
