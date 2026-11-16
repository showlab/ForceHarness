"""The real robot's tool menu: what each condition may call, with the text the model reads (layers 2-3 interface).

PRIMITIVES maps each tool to its arguments, the required ones, whether it needs force (force=True tools exist
only in condition A), and its description. Condition A (ForceHarness: phys2 tool results) gets every tool except
the task-only ones; condition B (the no-force control: same robot, same guards, tool results without force)
gets the tools that need no force. A task may change the menus (tasks.py); menu_text() and validate() take the
per-episode menus, so a task never changes the module-level ones. The numbers in the descriptions are the
ranges validate() enforces before anything moves.
"""
from forceharness.real.primitives import (SCREW_PUSH_MAX_N, SCREW_PUSH_N, SCREW_TURN_START_N, WIGGLE_DEPTH_MAX_M,
                                          WIGGLE_SHIFT_MAX_M, WIGGLE_TIME_MAX_S, WIGGLE_YAW_MAX_DEG)
from forceharness.real import receipt_text as T

MAX_ACTIONS, MAX_MINUTES = 40, 40           # episode length only; --max-actions / --max-minutes
ACTION_RANGE, MINUTE_RANGE = (1, 80), (1, 60)
MOVE_MAX_M = .10          # the robot side's envelope is 120 mm including the 8 mm lead
ROT_MAX_DEG = 30.         # the robot side allows 0.60 rad per motion; turn/rotate_z allow 0.55 rad
ROTATE_Z_MAX_DEG = 180.   # one rotate_z or screw call, run as segments of <= ROT_MAX_DEG
FORCE_MAX_N = 10.         # contact targets (move_until_contact, push, slide, search) unless a task raises them
PULL_FORCE_MAX_N = 20.
PULL_FORCE_DEFAULT_N = 8.
RAW_FORCE_CAP_N = 25.     # the raw cap ends the episode
INSERT_FORCE_MAX_N = 12.
# push may ask for up to 15 N (with the contact lead it reaches about 15-19 N on the FR3); other contact
# targets keep --force-max
PUSH_FORCE_MAX_N = 15.
# A task may raise the contact targets (--force-max) and insert (--insert-force-max) up to 0.8 x the raw cap;
# the cap itself never changes.
TARGET_FORCE_CEILING_N = .8*RAW_FORCE_CAP_N
SLIDE_FORCE_DEFAULT_N = 1.  # slide's own default; a task may start higher (--slide-force-default)
FORCE_LIMITS = dict(contact=FORCE_MAX_N, insert=INSERT_FORCE_MAX_N, slide_default=SLIDE_FORCE_DEFAULT_N)
# One move_until_contact call may search this far, run as segments of at most MOVE_MAX_M, each its own
# robot-side motion with every guard and a fresh still zero.
CONTACT_SEARCH_MAX_M = .30

NUM, VEC3, BOOL, TEXT, MODE, VEC2 = 'number', 'vec3', 'bool', 'text', 'mode', 'vec2'
# name -> arguments (JSON names the model writes), required ones, force-only?, menu text.
# Ranges are checked in validation.validate() before any motion.
PRIMITIVES = {
    'scene': dict(args={}, req=(), force=False,
                  text='Take new wrist and exterior photos and detect the objects on the table again; the robot does '
                       'not move. For an overview, first raise the hand (z ≈ 0.25–0.30), then call scene.'),
    'locate': dict(args=dict(frame=NUM, u=NUM, v=NUM), req=('frame', 'u', 'v'), force=False,
                   text='Read the local depth at a given pixel of the latest wrist image; the robot does not move. frame '
                        'is the frame number from the observation; u runs left to right and v top to bottom, in integer '
                        'pixel coordinates of the original image (not coordinates normalized to 0–1000). Parts such '
                        'as a handle can be selected even without their own detection box; returns a surface reference '
                        'point in the base frame, not an automatic grasp pose. If depth is missing, the reason is given.'),
    'move_to': dict(args=dict(x=NUM, y=NUM, z=NUM, speed=NUM), req=('x', 'y', 'z'), force=False,
                    text='Move the TCP (the center of the two finger pads) in a straight line to (x, y, z), keeping the '
                         'hand orientation. At most 0.10 m per call; split longer moves; speed ≤ 0.008 m/s '
                         '(default 0.008).'),
    'move_by': dict(args=dict(dx=NUM, dy=NUM, dz=NUM, speed=NUM), req=('dx', 'dy', 'dz'), force=False,
                    text='Move by (dx, dy, dz) relative to the current position; same rules as move_to.'),
    'rotate_z': dict(args=dict(angle_deg=NUM), req=('angle_deg',), force=False,
                     text='Turn the hand about the vertical axis by angle_deg (counterclockwise seen from above is '
                          'positive); the TCP stays put. Use it to change the direction in which the two fingers close '
                          '(the "grasp angle" in the object list says how far to turn). |angle_deg| ≤ 180 per call; '
                          'above 30° the program splits the turn into segments (≤ 30° each) and runs them '
                          'in sequence, about 8°/s.'),
    'screw': dict(args=dict(angle_deg=NUM, push=NUM), req=('angle_deg',), force=True,
                  text='Drive a thread: with the screw (knob) in the hand, first press down along the vertical axis to '
                       f'about push N (default {SCREW_PUSH_N:g}, ≤ {SCREW_PUSH_MAX_N:g}; the lead is set from the '
                       'arm\'s stiffness of about 0.75 N/mm; when the total force reading reaches 23 N it stops pressing '
                       'harder, and above 24 N it eases off a little, without stopping the action) to press the screw '
                       'in, holding the press at least 1 s until the screw stops sinking (at most about 6 s; tool result '
                       'press_seated), then turn about the vertical axis by angle_deg (counterclockwise seen from above '
                       'is positive; tighten a right-hand thread with a negative value). While turning, the press starts '
                       f'at about {SCREW_TURN_START_N:g} N and rises linearly to push as it turns '
                       '(later segments of the same call are already engaged and only press quickly to the starting '
                       'force before turning); it follows the thread down, and follows the screw up if it pushes back. '
                       'Sideways: while pressing the hand holds still (keeping the screw from tipping); while turning it '
                       'is soft (about 70% stiffness), letting the hand follow the screw a little instead of bending it '
                       'back, at most 10 mm off; the sideways displacement in the tool result is how far the hand '
                       'followed the screw and needs no correction. This differs from rotate_z, which locks the hand '
                       'position, so that the hand fights the thread when the thread pulls down. The tool result gives '
                       'separately how '
                       'far it sank while pressing and while turning, and the equivalent millimetres per turn (from the '
                       'turning part only; with the thread engaged it is roughly the pitch; near 0 = not engaged, '
                       'slipping or fully tightened; negative = backing out), and the change of the sideways force. The '
                       'harder it presses, the harder it is to turn; when the wrist cannot keep up, the turn slows down. '
                       '|angle_deg| ≤ 180 per call; above 30° it is split into segments automatically (each '
                       'segment presses first, then turns at about 5°/s); it stops if it sinks more than 10 mm in '
                       'one call.'),
    'point_tool': dict(args=dict(direction=VEC3), req=('direction',), force=False,
                       text='Point the fingers along direction ([0,0,-1] = straight down); the TCP stays put. At most '
                            '30° per call; split larger angles.'),
    'turn': dict(args=dict(axis=VEC3, pivot=VEC3, angle_deg=NUM, stop_on_resistance_drop=BOOL),
                 req=('axis', 'pivot', 'angle_deg'), force=False,
                 text='Rotate the hand (with whatever it holds) rigidly by angle_deg about the axis along axis through '
                      'pivot (right-hand rule); the TCP moves on an arc. |angle_deg| ≤ 30 per call; use it to '
                      'flip, set upright or twist things (when holding something, choose its support point as the '
                      'pivot).',
                 text_a=' stop_on_resistance_drop=true: stop when the tangential resistance along the arc first rises '
                        'and then clearly drops (relative force signal).'),
    'move_until_contact': dict(args=dict(direction=VEC3, max_dist=NUM, f_threshold=NUM), req=('direction', 'max_dist'),
                               force=True,
                               text='Move slowly along direction (4 mm/s) and stop when the contact force reaches '
                                    'f_threshold (default 5 N, ≤ {force_max}), or after max_dist (≤ 0.30 m; '
                                    'above 0.10 m the program splits it into segments run in sequence, settling and '
                                    're-zeroing before each segment, and stopping on contact). The tool result gives '
                                    'the contact point and force.',
                               # for a task that puts it on B's menu: B's tool results carry no force
                               text_b='Move slowly along direction (4 mm/s) and stop when it touches something, or '
                                      'after max_dist (≤ 0.30 m). The tool result only says where it stopped and '
                                      'whether it went the full distance; it gives no force reading.'),
    'probe': dict(args=dict(direction=VEC3, amplitude=NUM, f_threshold=NUM), req=('direction',), force=True,
                  text='Tap: move along direction until contact (at most amplitude ≤ 0.10 m, default 0.02), then '
                       'return to the start; reports the contact distance and force.'),
    'push': dict(args=dict(direction=VEC3, target_force=NUM, duration=NUM, max_dist=NUM, stop_on_resistance_drop=BOOL),
                 req=('direction', 'target_force'), force=True,
                 text='Press along direction and regulate the contact force to target_force (≤ {push_max} N), '
                      'holding it for duration seconds (≤ 30, default 2), travelling at most max_dist '
                      '(≤ 0.10 m, default 0.05). stop_on_resistance_drop=true: stop when the resistance passes a '
                      'peak and drops (a "click").'),
    'slide': dict(args=dict(direction=VEC3, distance=NUM, normal_force=NUM, normal=VEC3), req=('direction', 'distance'),
                  force=True, text='Slide along a surface: press along normal (default [0,0,-1], downward) with '
                                   'normal_force (≤ {force_max} N, default {slide_default}) while moving distance '
                                   '(≤ 0.10 m) along direction.'),
    'search': dict(args=dict(normal=VEC3, radius_max=NUM, pitch=NUM, normal_force=NUM, drop=NUM, mode=MODE, hop=NUM,
                             guess=VEC2),
                   req=(), force=True,
                   text='Search for a hole near the current position while holding a part: first probe down along '
                        'normal (default downward) by at most 3 cm to find support, then search in a spiral out to '
                        'radius_max (≤ 0.015 m; slide defaults 0.012 and pitch 0.002, about 1 minute; friction '
                        'keeps the hand a few millimetres behind the spiral target, and the tool result note gives the '
                        'radius actually covered); it reports dropped when the part falls along normal by more than '
                        'drop (default 0.004). '
                        'mode="slide" (default, sliding on the surface) or "hop" (at each point: lift by hop ≤ '
                        '0.01, move over in the air, re-zero in the air, probe straight down until the force rises by '
                        'normal_force (at least 6 N), then keep pressing for 1 s; no sideways dragging, so a long thin '
                        'part in the fingers is not dragged askew; the hop defaults are pitch 1.5 mm, hop 2 mm, drop '
                        '1 mm, radius_max 4 mm, 6 N, and usually need no change; guess=[dx, dy] (metres, ≤ 0.02): '
                        'your estimate of the hole\'s offset from the hand\'s current position; it first probes once in '
                        'place as the surface reference, then starts from the guessed point and probes outward around '
                        'it, and the tool result gives how far the hole actually was from guess (found_from_guess_m); '
                        'it stops at once and reports dropped when a point is deeper than the typical height of the '
                        'previous points by more than drop; for the first probe, which has nothing to compare with '
                        'yet, it compares with the surface touched by move_until_contact nearby just before (within '
                        '1.5 cm), and also reports dropped when deeper by more than 2 mm; when the hand cannot move '
                        'sideways and several probes in a row press at the same spot (often it is already caught in '
                        'the hole), it reports stuck, and you can try insert right away; each point takes about '
                        '3–5 s, the number of points ≈ 3×(radius_max/pitch)², at most about '
                        '195 s). Then use insert. '
                        'slide slides while pressing with normal_force (default 1 N); use 1–2 N to find a hole: '
                        'above 4–5 N friction often stops the hand from sliding, and the action stops with the '
                        'tool result stalled. While sliding, as soon as the tip is 0.25 mm below the surface it just '
                        'slid over (its height over the previous 0.5–2 s), it stops, holds sideways, immediately '
                        'thrusts down with about 15 N for 2 s (no more once the total force reading reaches 23 N), and '
                        'judges only after the full press: only if it is then more than 1.5 mm below that surface does '
                        'it count as in the hole (pressing on a flat surface also sinks about 0.9 mm because the '
                        'fingers and gripper compress under load), reported as dropped; the tool result note says how '
                        'far the found position is from the start; otherwise it keeps sliding. If it starts in the '
                        'air, it first moves down to find the surface: it counts as touching only once the surface '
                        'stops the hand from moving down (force readings are not used).'),
    'insert': dict(args=dict(depth=NUM, normal=VEC3, max_force=NUM, wiggle=NUM), req=(), force=True,
                   text='Compliant insertion: advance by depth (≤ 0.10 m, default 0.02) along normal (default '
                        'downward), slowing down as the normal force approaches max_force (≤ {insert_max} N, '
                        'default 12), soft sideways so the part centers itself. Result seated / jammed / stalled.'),
    'wiggle': dict(args=dict(depth=NUM, force=NUM, yaw_deg=NUM, shift=NUM, duration=NUM), req=(), force=True,
                   text='Rocking insertion (correction): use it in place when the plug is already caught on the port and '
                        'insert cannot push it in. Press straight down, raising the press from 8 N to force within 2 s '
                        '(default 18, ≤ {insert_max}; no more once the total force reading reaches 23 N), while '
                        'rocking back and forth about the vertical axis by ±yaw_deg'
                        f' (default 3°, ≤ {WIGGLE_YAW_MAX_DEG:g}°, about 3 s per cycle) and drawing a '
                        'small circle of radius shift in the plane'
                        f' (default 0.0005 m, ≤ {WIGGLE_SHIFT_MAX_M:g}), so that the chamfer of the port mouth '
                        'guides the plug in. '
                        'When the hand can no longer follow the small circle sideways (the port walls already hold the '
                        'plug), it stops rocking and only presses down until it stops sinking, then judges: '
                        'deeper than depth below the surface touched by the last search / move_until_contact (from the '
                        'start position if nothing nearby was touched)'
                        f' (default 0.008 m, ≤ {WIGGLE_DEPTH_MAX_M:g}) reports seated, shallower reports jammed '
                        '(caught on the port rim); '
                        f'not held by duration seconds (default 15, ≤ {WIGGLE_TIME_MAX_S:g}) reports not_seated. '
                        'The tool result gives the depth below the surface. '
                        'It can only correct errors within 1–3° and 1 mm; a plug inserted the wrong way '
                        'round (USB-A fits only one way) will not wiggle in: lift it, turn it 180° and align '
                        'again.'),
    'lift': dict(args=dict(height=NUM, hold_time=NUM), req=(), force=True,
                 text='Lift straight up by height (≤ 0.10 m, default 0.02), then hold still for hold_time seconds '
                      '(≤ 30, default 1). When holding something, the still reading in the tool result is its '
                      'weighing.'),
    'pull': dict(args=dict(direction=VEC3, force=NUM, distance=NUM, speed=NUM), req=('direction',), force=True,
                 text='Pull along direction while holding something attached to a mechanism (a drawer, a door handle): '
                      f'regulate the pulling force to force (≤ {PULL_FORCE_MAX_N:g} N, default '
                      f'{PULL_FORCE_DEFAULT_N:g}), over at most distance'
                      ' (≤ 0.10 m, default 0.02), speed ≤ 0.004 m/s (default 0.004). In a horizontal pull '
                      'the hand gives way sideways to follow the mechanism; '
                      'in a vertical or inclined direction it pulls in a straight line along the 3-D direction, '
                      'keeping the wrist orientation; distance is the actual TCP travel; '
                      'the control target may keep building up the pulling force within the existing 6 mm lead, and it '
                      'unloads and stops once the actual travel is reached. If it cannot pull, it returns stalled or '
                      'timeout. '
                      'Once the grip is confirmed firm, the direction correct and the cabinet not moving, you may raise '
                      'the force step by step (e.g. 8→12→16→20 N); '
                      'failing to pull at 3–5 N does not mean the mechanism cannot open. Decide whether to '
                      'continue from the actual force and displacement; there is no need to go to the limit every '
                      'time.'),
    'grasp': dict(args={}, req=(), force=False, text='{grasp_rule}'),
    'release': dict(args={}, req=(), force=False, text='{release_rule}'),
    'hold': dict(args=dict(seconds=NUM), req=('seconds',), force=False,
                 text='Hold in place for seconds seconds (≤ 120).'),
    'done': dict(args=dict(success=BOOL, note=TEXT), req=('success',), force=False,
                 text='End the episode: success=true/false; in note write the final result as the task asks; if no '
                      'output format is given, write the basis of your judgement.'),
}
FORCE_ONLY_ARGS = {'stop_on_resistance_drop', 'f_threshold'}
TASK_ONLY = ('wiggle',)     # on a menu only where a task adds it (plug insertion, screw driving)
CONDITIONS = dict(
    A=dict(label='phys2_force', interface='force', injector='phys2', prompt='free_task_force.md',
           menu=[n for n in PRIMITIVES if n not in TASK_ONLY]),
    B=dict(label='none_guard', interface='guard', injector='none', prompt='free_task_baseline.md',
           menu=[n for n, p in PRIMITIVES.items() if not p['force']]))
GRIP_RULES = dict(
    auto=('Close the gripper (fixed grip force); the two fingers close along the hand\'s y axis (closing_axis in the '
          'state, the line joining the two red boxes in the wrist image). **Nobody confirms the alignment for you**: '
          'before closing, check the wrist image and the object list to make sure the two fingers straddle the part '
          'to be grasped and that the closing direction crosses its narrow side (if the grasp angle is off by more '
          'than 10°, rotate_z first). An empty grasp (the two fingers meet) is reported to you; nothing is in the '
          'hand, and you can try again after release; after every release that is confirmed open you may grasp '
          'again, with no limit on the number of grasps.',
          'Open the gripper (to 80 mm). First put the object down where it is supported (on the table, in a stable '
          'position), then release; releasing in the air drops it.'),
    operator=('Close the gripper (fixed grip force); the two fingers close along the hand\'s y axis. The operator first '
              'confirms the alignment before letting it run; if not approved, the tool result carries the operator\'s '
              'hint: adjust as hinted, then ask again.',
              'Open the gripper. The operator first confirms that the object is supported before letting it run.'))
# Plug insertion: search with the slide mode only (no mode, hop or guess arguments), since a hop search tends to
# lift a plug that has already entered the port back out.
SEARCH_SLIDE_ONLY = dict(
    args={k: v for k, v in PRIMITIVES['search']['args'].items() if k not in ('mode', 'hop', 'guess')},
    req=(), force=True,
    text='Search for a hole near the current position while holding a part: first probe down along normal (default '
         'downward) by at most 3 cm to find support, then slide on the surface in a spiral out to radius_max '
         '(≤ 0.015 m; default 0.012, pitch default 0.002, about 1 minute; friction keeps the hand a few '
         'millimetres behind the spiral target, and the tool result note gives the radius actually covered); '
         'it reports dropped when the part falls along normal by more than drop (default 0.004). Then use insert. '
         'It slides while pressing with normal_force (default 1 N); use 1–2 N to find a hole: above 4–5 N '
         'friction often stops the hand from sliding, and the action stops with the tool result stalled. '
         'As soon as the tip is 0.25 mm below the surface it just slid over (its height over the previous 0.5–2 s), '
         'it stops, holds sideways, immediately thrusts down with about 15 N for 2 s'
         ' (no more once the total force reading reaches 23 N), and judges only after the full press: only if it is '
         'then more than 1.5 mm below that surface does it count as in the hole (pressing on a flat surface also sinks '
         'about 0.9 mm because the fingers and gripper compress under load), '
         'reported as dropped; the tool result note says how far the found position is from the start; otherwise it '
         'keeps sliding. If it starts in the air, it first moves down to find the surface: it counts as touching only '
         'once the surface stops the hand from moving down (force readings are not used).')


def signature(name, condition, primitives=None):
    spec = (primitives or PRIMITIVES)[name]
    args = [k for k in spec['args'] if condition == 'A' or k not in FORCE_ONLY_ARGS]
    return name+'('+', '.join(k if k in spec['req'] else k+'?' for k in args)+')'


def menu_text(condition, grip, limits=None, *, conditions=None, primitives=None):
    limits = limits or FORCE_LIMITS
    primitives = primitives or PRIMITIVES
    lines = []
    for name in (conditions or CONDITIONS)[condition]['menu']:
        spec = primitives[name]
        text = (spec['text'].replace('{grasp_rule}', grip[0]).replace('{release_rule}', grip[1])
                .replace('{force_max}', f"{limits['contact']:g}").replace('{insert_max}', f"{limits['insert']:g}")
                .replace('{push_max}', f"{max(limits['contact'], PUSH_FORCE_MAX_N):g}")
                .replace('{slide_default}', f"{limits.get('slide_default', SLIDE_FORCE_DEFAULT_N):g}"))
        if condition == 'A' and spec.get('text_a'):
            text += spec['text_a']
        if condition == 'B' and spec.get('text_b'):
            text = spec['text_b']
        lines.append(f'- `{signature(name, condition, primitives)}`: {text}')
    return '\n'.join(lines)


def action_prompt(condition, conditions=None):
    return (T.YOUR_MOVE+'\n'+T.SUBMIT_ONE+', '.join((conditions or CONDITIONS)[condition]['menu'])+
            '\n'+T.JSON_EXAMPLE+'{"calls":[{"name":"move_to","args":{"x":0.40,"y":0.0,"z":0.25}}],"note":"'
            + T.NOTE_EXAMPLE+'"}')
