import numpy as np
import pytest

from forceharness.primitives.report import ContactEvent, ContactReport
from forceharness.translation import make_injector, FORMATS


def fake_report():
    rows = [[i * 0.05, 0.1, -0.2, 0.0 if i < 10 else 6.0 + 0.1 * i, 0.01, 0.02, 0.0] for i in range(30)]
    return ContactReport(primitive="move_until_contact", params={"direction": [0, 0, -1], "max_dist": 0.1}, termination="contact",
                         t_start=0.0, t_end=1.5, start_pos=[0, 0, 1.0], end_pos=[0, 0, 0.9], commanded_disp=[0, 0, -0.1],
                         actual_disp=[0, 0, -0.1], events=[ContactEvent(0.5, "contact_start", 6.0, "reaction force mainly along +z", [0, 0, 0.9])],
                         wrench_rows=rows, final_wrench=[0.1, -0.2, 8.9, 0.01, 0.02, 0.0], peak_force=8.9, peak_torque=0.02,
                         mean_force_in_contact=7.5, n_steps=30, gripper_opening=0.08, yaw_deg=0.0, derived={"payload_estimate_kg": 0.3})


@pytest.mark.parametrize("fmt", [f for f in FORMATS if f != "encoder"])
def test_each_format_renders(fmt):
    inj = make_injector(fmt)
    parts = inj.render(fake_report(), {})
    assert parts and parts[0]["type"] == "text" and "move_until_contact" in parts[0]["text"]
    texts = "\n".join(p["text"] for p in parts if p["type"] == "text")
    if fmt == "none":
        assert "N" not in texts.split("primitive=")[1].split("\n")[1] and "8.9" not in texts
        assert "blocked" in texts
    if fmt == "raw":
        assert "Fz=+8.90" in texts
    if fmt == "events":
        assert "contact_start" in texts
    if fmt == "signals":
        assert "payload_estimate_kg" in texts and "contact_start" in texts
    if fmt == "plot":
        imgs = [p for p in parts if p["type"] == "image"]
        assert imgs and imgs[0]["png"][:4] == b"\x89PNG"


def test_combo_and_describe():
    inj = make_injector("events+plot")
    assert inj.name == "events+plot"
    parts = inj.render(fake_report(), {})
    assert any(p["type"] == "image" for p in parts) and "EVENTS" in inj.describe()


