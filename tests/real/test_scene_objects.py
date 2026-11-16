"""Objects on the table from one wrist RGB-D capture: geometry, grasp angle, image directions, and the text the
model reads. Synthetic captures only (a known camera over a flat table); no device and no robot."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from forceharness.real import runner as free
from forceharness.real import scene_objects as so
from forceharness.real.config import BackendConfig
from forceharness.real.wrist_overlay import PAD_CENTRE_TCP_Z
from real_helpers import make_scene

# A logged still state of the FR3: the joints reproduce the flange pose, so the kinematic walk runs.
LOGGED = dict(joints=[-0.0008282056078314781, -0.5011396408081055, 0.013265701942145824, -2.6334707736968994,
                      0.007745702285319567, 2.1053194999694824, 0.8444894552230835],
              flange_pos=[0.37455424666404724, 0.004877594765275717, 0.40578076243400574],
              flange_quat=[0.9134523868560791, -0.4067210257053375, -0.012349549680948257, 0.005497400648891926])


def obj(length, width, axis_deg):
    return dict(length=length, width=width, axis_deg=axis_deg)


def test_grasp_angle_straddles_the_narrow_side():
    lying = so.grasp_angle(obj(.20, .062, -4.), 87.)               # a lying bottle: already across it
    assert lying['kind'] == 'narrow_side' and lying['target_closing_deg'] == 86 and lying['rotate_z_deg'] == -1
    bar = so.grasp_angle(obj(.10, .02, 89.), 87.)                   # a bar along y, fingers closing along y
    assert bar['target_closing_deg'] == pytest.approx(-1) and bar['rotate_z_deg'] == pytest.approx(-88)
    wrap = so.grasp_angle(obj(.10, .02, -2.), -88.)                   # closing -88 is the same line as +92
    assert wrap['target_closing_deg'] == 88 and wrap['rotate_z_deg'] == pytest.approx(-4)
    assert all(abs(so.grasp_angle(obj(.1, .02, a), c)['rotate_z_deg']) <= 90
               for a in range(-89, 91, 7) for c in range(-90, 90, 11))
    assert so.grasp_angle(obj(.018, .012, 40.), 87.)['kind'] == 'any'      # small: no reliable direction
    assert so.grasp_angle(obj(.07, .065, 40.), 87.)['kind'] == 'any'       # round from above
    assert so.grasp_angle(obj(.14, .11, 79.), 87.)['kind'] == 'too_wide'
    assert so.grasp_angle(obj(.10, .02, 0.), None).get('rotate_z_deg') is None   # hand on its side


def test_a_downward_view_names_the_image_directions_and_the_angles(tmp_path):
    frame = make_scene(tmp_path/'scene', boxes=[dict(centre=(.45, .05), size=(.12, .02), height=.03, yaw_deg=30.)])
    found = so.detect(frame, free.TCP_TRANSLATION)
    assert found['version'] == 'scene_objects/angle1'
    assert found['image_axes']['x_word'] == 'down' and found['image_axes']['y_word'] == 'right'
    assert found['closing_deg'] == pytest.approx(-90, abs=3) or found['closing_deg'] == pytest.approx(90, abs=3)
    assert all('grasp' in o for o in found['objects']) and not any('grasp' in o for o in found['candidates'])
    text = so.describe(found)
    assert "+x (away from the base) points down in the image, +y (the robot's left) points right" in text
    assert '  Grasp angle: ' in text and '  Top relative to TCP [dx, dy, dz]' in text
    out = tmp_path/'wrist.png'
    marks = so.annotate(frame, out, found, (0., 0., PAD_CENTRE_TCP_Z), BackendConfig().tcp_quaternion)
    assert out.is_file() and marks['objects_drawn'] >= 1


def test_rotation_reach_counts_the_joint_travel_per_segment():
    state = LOGGED
    b = SimpleNamespace(_state=state)
    pivot = state['flange_pos']
    feasible, why = free.rotation_reach(b, [0, 0, 1], pivot, 90)            # one robot-side motion: joint 7 > 0.85 rad
    assert '0.85 rad' in why and feasible < 60
    assert free.rotation_reach(b, [0, 0, 1], pivot, 90, free.ROT_MAX_DEG) == (pytest.approx(90), None)
    assert free.rotation_reach(b, [0, 0, 1], pivot, -90, free.ROT_MAX_DEG) == (pytest.approx(90), None)


def test_grasp_reach_adds_the_joint_check_to_the_hint():
    state = LOGGED
    b = SimpleNamespace(_state=state, eef_pos=np.asarray(state['flange_pos']))
    found = dict(objects=[dict(grasp=dict(kind='narrow_side', rotate_z_deg=-88.)),
                          dict(grasp=dict(kind='narrow_side', rotate_z_deg=4.)), dict(grasp=dict(kind='any'))])
    free.grasp_reach(b, found)
    assert found['objects'][0]['grasp']['reach'] == dict(ok=True)
    assert 'reach' not in found['objects'][1]['grasp'] and 'reach' not in found['objects'][2]['grasp']
    line = so._grasp_text(dict(width=.02, partial=False, grasp=found['objects'][0]['grasp']
                               | dict(target_closing_deg=-1., closing_now_deg=87.)))
    assert 'rotate_z(-88)' in line and 'the joints can turn that far' in line


def test_rotate_z_takes_up_to_90_deg_as_30_deg_segments():
    b, _ = free.demo()[:2]
    with b:
        assert free.validate('rotate_z', dict(angle_deg=90), b, 'A')[0] == dict(angle_deg=90)
        with pytest.raises(ValueError, match='angle_deg'):
            free.validate('rotate_z', dict(angle_deg=181), b, 'A')
        p = free.FreePrimitives(b, confirm=lambda kind, state: True, tick=lambda: None, stop_protocol='r9', interface='force')
        p.target_quat = np.asarray(b.eef_quat, dtype=float).copy()
        rep = p.call('rotate_z', angle_deg=-75)
        assert rep.termination == 'completed' and not b.fault
        assert [s['requested_deg'] for s in rep.derived['segments']] == [-25, -25, -25]
        assert rep.yaw_deg == pytest.approx(-75, abs=4) and rep.derived['rotated_deg'] == rep.yaw_deg
        assert rep.params['angle_deg'] == -75 and 'static_after_relax' in rep.derived
        assert rep.wrench_rows[-1][0] == pytest.approx(rep.duration, abs=.5)
        assert p.call('rotate_z', angle_deg=20).derived.get('segments') is None       # one segment: unchanged


def test_a_stalled_segment_ends_the_whole_rotation():
    b, _ = free.demo()[:2]
    with b:
        p = free.FreePrimitives(b, confirm=lambda kind, state: True, tick=lambda: None, stop_protocol='r9', interface='force')
        p.target_quat = np.asarray(b.eef_quat, dtype=float).copy()
        b.port.driver.frozen = True                                  # the plant ignores every new target
        rep = p.call('rotate_z', angle_deg=90)
        assert rep.termination == 'stalled' and len(rep.derived['segments']) == 1 and not b.fault


def synthetic_scene(tmp_path, monkeypatch, patches):
    """Known camera looking down at a flat table; patches give visible object top surfaces."""
    depth = np.full((256, 256), .35, dtype=np.float32)
    for (v0, v1, u0, u1), height in patches:
        depth[v0:v1, u0:u1] = .35-height
    intr = dict(fx=300., fy=300., ppx=128., ppy=128., coeffs=[0.]*5)
    (tmp_path/'scene.json').write_text(json.dumps(dict(wrist=intr)))
    np.savez_compressed(tmp_path/'scene.npz', wrist_depth=depth)
    camera = np.diag([1., -1., -1., 1.])
    camera[:3, 3] = [.45, 0., .35]
    tcp = np.eye(4)
    tcp[:3, 3] = [.7, .3, .3]             # outside this synthetic view, so no hand occlusion
    monkeypatch.setattr(so, 'scene_geometry', lambda *a, **k: (camera, tcp, .08))
    return so.detect(tmp_path)


def test_eight_mm_peg_survives_fine_grid_and_grid_boundary(tmp_path, monkeypatch):
    found = synthetic_scene(tmp_path, monkeypatch, [((75, 86, 70, 81), .12)])
    assert len(found['objects']) == 1
    peg = found['objects'][0]
    assert peg['height'] == pytest.approx(.12, abs=.001)
    assert .006 < peg['width'] < .011


@pytest.mark.parametrize('patch,height', [((80, 98, 70, 88), .008), ((80, 132, 70, 80), .02)])
def test_thin_flat_objects_are_reported_as_uncertain_instead_of_silently_lost(tmp_path, monkeypatch, patch, height):
    found = synthetic_scene(tmp_path, monkeypatch, [(patch, height)])
    all_objects = found['objects']+found['candidates']
    assert len(all_objects) == 1
    assert all_objects[0]['height'] == pytest.approx(height, abs=.001)
    assert all_objects[0]['detection'] == 'candidate'
    assert 'confirm with RGB or another view first' in so.describe(found)


def test_nearby_small_objects_stay_separate(tmp_path, monkeypatch):
    found = synthetic_scene(tmp_path, monkeypatch, [((75, 86, 70, 81), .12), ((75, 86, 98, 109), .12)])
    assert len(found['objects']) == 2
    assert abs(found['objects'][0]['top'][0]-found['objects'][1]['top'][0]) > .015


def test_image_edge_object_is_retained_and_marked_partial(tmp_path, monkeypatch):
    found = synthetic_scene(tmp_path, monkeypatch, [((0, 18, 70, 85), .12)])
    assert len(found['objects']) == 1 and found['objects'][0]['partial']


def test_isolated_bad_depth_pixels_do_not_become_objects(tmp_path, monkeypatch):
    patches = [((v, v+1, u, u+1), .1) for v in range(10, 240, 20) for u in range(10, 240, 20)]
    found = synthetic_scene(tmp_path, monkeypatch, patches)
    assert found['objects'] == found['candidates'] == []
    assert 'which does not mean nothing is in the image' in so.describe(found)


def test_a_peg_and_a_bottle_with_the_hand_above(tmp_path):
    frame = make_scene(tmp_path/'scene', boxes=[dict(centre=(.40, .03), size=(.009, .009), height=.135),
                                                 dict(centre=(.47, -.05), size=(.065, .065), height=.20)])
    found = so.detect(frame, free.TCP_TRANSLATION)
    pegs = [o for o in found['objects'] if np.linalg.norm(np.array(o['top'][:2])-[.40, .03]) < .01]
    assert len(pegs) == 1 and pegs[0]['height'] == pytest.approx(.135, abs=.006) and pegs[0]['width'] < .025
    assert pegs[0]['top_minus_tcp_mm'] == pytest.approx((np.array(pegs[0]['top'])-found['tcp'])*1000, abs=1)
    ids = [o['id'] for o in found['objects']+found['candidates']]
    assert len(ids) == len(set(ids))
    text = so.describe(found)
    assert '#1 center' in text and 'not an automatic grasp point' in text
    marks = so.annotate(frame, tmp_path/'objects.png', found, (0, 0, PAD_CENTRE_TCP_Z), BackendConfig().tcp_quaternion)
    assert marks['objects_drawn'] == len(found['objects'])+len(found['candidates'])


def test_locate_reads_the_surface_under_a_pixel_and_refuses_bad_ones(tmp_path):
    from forceharness.real import pixel_target
    from forceharness.real.wrist_overlay import deproject, scene_geometry
    frame = make_scene(tmp_path/'scene', boxes=[dict(centre=(.45, -.03), size=(.04, .04), height=.05)])
    scene = json.loads((frame/'scene.json').read_text())
    T_cam, _, _ = scene_geometry(scene, free.TCP_TRANSLATION, BackendConfig().tcp_quaternion)
    # the pixel that sees the centre of the box top
    p_cam = np.linalg.inv(T_cam)@np.array([.45, -.03, .055, 1.])
    u = int(round(scene['wrist']['fx']*p_cam[0]/p_cam[2]+scene['wrist']['ppx']))
    v = int(round(scene['wrist']['fy']*p_cam[1]/p_cam[2]+scene['wrist']['ppy']))
    got = pixel_target.locate(frame, u, v, free.TCP_TRANSLATION)
    assert got['surface_point_m'] == pytest.approx([.45, -.03, .055], abs=.003)
    assert got['note'].startswith('This is the surface reference point at the chosen pixel')
    for bad, match in (((1, 1), 'measurable range'), ((u+.5, v), 'integer pixel')):
        with pytest.raises(ValueError, match=match):
            pixel_target.locate(frame, *bad, free.TCP_TRANSLATION)
