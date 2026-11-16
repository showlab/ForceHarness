"""Fixed English phrases of the real-robot observations and tool results (layer 3 rendering on the FR3).

The runner (runner.py) and the object list (scene_objects.py) build their model-facing text from these
constants, so a GUI that splits a tool result into its motion, force and scene parts can match exactly the same
strings. Unless noted, a constant is the start of a line. The force lines of the physics translation layer
("contact events", "contact state at end", "measurements (") come from forceharness/translation and are listed
in FORCE_HEADS as well. Changing any phrase here changes what the model sees.
"""
import re

# ---- relay files written per turn (obs_<n>.md) ---------------------------------------------------------
TURN_HEAD = '# turn '                       # first line of obs_<n>.md, followed by the turn number
SYSTEM_HEAD = '## SYSTEM'                   # turn 0 only: the system prompt follows
TOOL_RESULT_HEAD = '## TOOL RESULT'         # turns >= 1: one JSON object {"text": ..., "states": ...}
YOUR_MOVE = '## YOUR MOVE'                  # the action request that ends every observation
SUBMIT_ONE = 'Submit one action per turn. Menu: '      # second line of the action request, then the menu names
JSON_EXAMPLE = 'JSON: '                     # third line: an example reply
NOTE_EXAMPLE = 'a one-sentence reason'      # the "note" value in that example
EPISODE_OVER = 'Episode over. Please stop.'  # done.json "text" when the runner has ended

# ---- tool-result lines written by the runner ----------------------------------------------------------
INVALID = 'Invalid; not executed: '                 # a refused call; the robot did not move
LOCATE_OK = 'locate: the robot did not move. '      # followed by the measured point as JSON
LOCATE_FAILED = 'locate returned no coordinates; the robot did not move: '
SCENE_OK = 'scene: the robot did not move.'
OPERATOR_DECLINED = 'The operator did not approve '  # chat confirmation mode only
EMPTY_GRASP = 'Empty grasp: '
HOLDING = 'Holding something: closed width '
# Force lines (condition A only). The still reading is one line; the two later phrases continue it.
STILL_READING = 'still reading after relaxing the controller: force on the hand '
CHANGE_SINCE_GRASP = '; change since the grasp '
WEIGHT_ESTIMATE = ' -> if the grasped object has fully left its support (hanging in the air), this change is about '
WEIGHT_CAVEAT = ('This is an estimate of absolute weight, with an error of up to ±{error} g: the hand\'s height, posture and '
                 'direction of motion differ between the grasp and this reading, the force the FR3 estimates from joint '
                 'torques is off by 2–4 N, and objects below a few hundred grams cannot be weighed. To compare the '
                 'weights of two objects, measure each once with exactly the same motion (same height, same posture, '
                 'arriving from the same direction); only their difference is accurate to tens of grams')
HELD_CONTACT_ZERO = 'Searching for contact while holding: '
CLOSED_WIDTH_NOTE = '(Closed width '
NET_FORCE_PEAK = 'net force peak '
TURNED = 'turned '
TORQUE_STOP = ('The torque estimate reached the 3 N·m limit and the action stopped (the safety guard works as usual); '
               'the episode continues. When moving while holding something, especially when lifting the hand after '
               'pressing on a surface, the torque that the FR3 estimates from joint torques drifts by 2–3 N·m, so it is '
               'not necessarily really stuck or loaded: check the photos. If the same action keeps stopping here, change '
               'the approach: smaller segments, slower, or first put the object down on a support surface and release '
               '(opening re-zeroes the force), then grasp again.')
TORQUE_STOP_B = ('A safety guard stopped this action (torque limit); the episode continues. If the same action keeps '
                 'stopping here, change the approach: smaller segments, slower, or first put the object down on a '
                 'support surface and release, then grasp again.')
RESUMED = 'At the start the gripper is already closed'

# ---- scene lines (appended to every tool result and to the first observation) --------------------------
PHOTOS = '[Photos this turn, in the relay folder, please open: '
SYNTHETIC_NO_IMAGES = '[Synthetic demo: no camera images]'
CAMERA_FAILED = '[Camera images could not be captured this turn: '
DETECTION_FAILED = '[Object detection failed: '
SCENE_OBJECTS = 'Scene objects'
OBJECT_LINE = re.compile(r'#\d+ center ')
TOP_OFFSET = '  Top relative to TCP'
GRASP_ANGLE = '  Grasp angle: '
NO_OBJECTS = '(No reliable object coordinates in this frame'
NO_HOLES = 'This list does not locate holes'
WRIST_AXES = 'Wrist image axes: '
TABLE_Z = 'Table z ≈ '
WRIST_FRAME = 'Wrist image frame='
NO_WRIST_IMAGE = 'The wrist image is not available this turn'

# Line starts a GUI can use to classify the lines of one tool result (stripped of leading spaces for the scene).
FORCE_HEADS = ('contact events', 'contact state at end', 'measurements (', NET_FORCE_PEAK, STILL_READING.strip(),
               HELD_CONTACT_ZERO, CLOSED_WIDTH_NOTE)
SCENE_HEADS = (PHOTOS, SYNTHETIC_NO_IMAGES, CAMERA_FAILED, DETECTION_FAILED, SCENE_OBJECTS, TOP_OFFSET.strip(),
               GRASP_ANGLE.strip(), NO_OBJECTS, NO_HOLES, WRIST_AXES, TABLE_Z, WRIST_FRAME, NO_WRIST_IMAGE)
