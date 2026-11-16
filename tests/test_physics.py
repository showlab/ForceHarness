import numpy as np

from forceharness.translation.estimators import EstimatorContext, estimate, G
from forceharness.primitives.report import ContactEvent, ContactReport
from forceharness.translation import make_injector, FORMATS


def _rep(primitive, rows, pos_rows=None, params=None, events=None, termination="completed", start=(0, 0, 0.9), end=(0, 0, 0.9)):
    return ContactReport(primitive=primitive, params=params or {}, termination=termination, t_start=0.0, t_end=rows[-1][0],
                         start_pos=list(start), end_pos=list(end), commanded_disp=[0, 0, 0],
                         actual_disp=[float(e - s) for s, e in zip(start, end)], events=events or [], wrench_rows=rows,
                         final_wrench=list(rows[-1][1:]), peak_force=0.0, peak_torque=0.0, mean_force_in_contact=0.0, n_steps=len(rows),
                         gripper_opening=0.04, yaw_deg=0.0, pos_rows=pos_rows or [])


def _lift_rows(fz=-7.83, tx=-0.035, ty=0.345, n=110, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    return [[i * 0.05, 0.1, -0.1, fz + rng.normal(0, noise), tx, ty, 0.0] for i in range(n)]


def test_payload_and_com_offset():
    rows = _lift_rows()
    rep = _rep("lift", rows, [[0, 0, 0.909]] * len(rows), params=dict(height=0.08, hold_time=1.0))
    qs = {q.name: q for q in estimate(rep, EstimatorContext(grip_closed=True))}
    assert abs(qs["payload_mass"].value - 0.798) < 0.03
    assert 0.01 < qs["payload_mass"].sigma < 0.06
    assert abs(qs["com_offset"].value - 4.4) < 0.3          # 0.345 / (0.798 * 9.81) = 4.4 cm
    assert "within_noise" not in qs["com_offset"].flags
    assert qs["com_offset"].extra["balanced_floor_cm"] < 1.5


def test_no_load_and_not_closed():
    rows = _lift_rows(fz=0.0, tx=0.0, ty=0.0)
    rep = _rep("lift", rows, params=dict(hold_time=1.0))
    qs = {q.name: q for q in estimate(rep, EstimatorContext(grip_closed=True))}
    assert "no_load" in qs["payload_mass"].flags and "com_offset" not in qs
    assert estimate(rep, EstimatorContext(grip_closed=False)) == []


def test_contact_height_and_impact():
    rows, pos = [], []
    for i in range(30):
        t = i * 0.05
        if i < 10:
            rows.append([t, 0.1, -0.2, 0.0, 0.0, 0.0, 0.0]); pos.append([0, 0, 1.0 - 0.01 * i])
        else:
            rows.append([t, 0.1, -0.2, 6.0 + 0.5 * (i - 10), 0.0, 0.0, 0.0]); pos.append([0, 0, 0.9 - 0.0005 * (i - 10)])
    ev = [ContactEvent(0.5, "contact_start", 6.0, "reaction force mainly along +z", [0, 0, 0.9])]
    rep = _rep("move_until_contact", rows, pos, params=dict(direction=[0, 0, -1], max_dist=0.1), events=ev, termination="contact",
               start=(0, 0, 1.0), end=(0, 0, 0.89))
    qs = {q.name: q for q in estimate(rep, EstimatorContext())}
    assert abs(qs["contact_height"].value - 0.904) < 0.003       # event position 0.9, moved back 4 mm along -z for the detection lag
    assert qs["contact_normal"].value < 5 and qs["contact_normal"].extra["opposes_motion"]
    assert qs["impact_peak"].value >= 6.0 and qs["impact_peak"].flags[0] in ("gentle", "moderate", "hard")
    assert "stiffness" in qs and qs["stiffness"].value > 0


def test_friction_from_slide():
    rows = [[i * 0.05, 2.0, 0.0, 5.0, 0.0, 0.0, 0.0] for i in range(40)]
    rep = _rep("slide", rows, params=dict(direction=[1, 0, 0], distance=0.1, normal_force=5.0))
    qs = {q.name: q for q in estimate(rep, EstimatorContext())}
    assert abs(qs["normal_force"].value - 5.0) < 0.01
    assert abs(qs["friction_coefficient"].value - 0.4) < 0.01 and "stick_slip" not in qs["friction_coefficient"].flags


def test_render_levels_and_registry():
    assert "phys3" in FORMATS
    rows = _lift_rows()
    rep = _rep("lift", rows, [[0, 0, 0.909]] * len(rows), params=dict(height=0.08, hold_time=1.0))
    rep.quantities = [q.to_dict() for q in estimate(rep, EstimatorContext(grip_closed=True))]
    log = [dict(turn=1, primitive="lift", name="payload_mass", value=0.17, unit="kg", sigma=0.03, flags=[], extra={})]
    texts = {}
    for lvl in (1, 2, 3, 4):
        inj = make_injector(f"phys{lvl}")
        assert inj.name == f"phys{lvl}"
        parts = inj.render(rep, {"log": log, "proprio": "minimal"})
        texts[lvl] = "\n".join(p["text"] for p in parts if p["type"] == "text")
        assert "payload mass" in texts[lvl] and "centre-of-mass offset" in texts[lvl]
        assert "duration" not in texts[lvl]                      # proprio minimal: no durations
    assert "±" not in texts[1] and "±" in texts[2]
    assert "ref:" not in texts[2] and "ref:" in texts[3] and "earlier in this episode" in texts[3]
    assert "~ about the weight of" in texts[4]
    assert "MEASUREMENTS" in make_injector("phys3").describe()


def test_stopped_short_note():
    rows = [[i * 0.05, 0, 0, -7.8, 0, 0, 0] for i in range(10)]
    rep = _rep("move_to", rows, params=dict(x=0, y=-0.2, z=0.825), termination="contact", start=(0, -0.2, 0.944), end=(0, -0.2, 0.941))
    txt = make_injector("signals").render(rep, {"proprio": "minimal"})[0]["text"]
    assert "NOT reached" in txt and "0.116" in txt
