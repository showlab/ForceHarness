"""Fixtures: small run folders in the harness's formats, a fake repository root, and a fake streaming model API.

The records mirror what the real-robot runner writes (episode.json free_task/1, the primitive report
action_NN.json, relay obs/cmd files, the object list in each tool result); the monitor never imports the runner.
Text marked "Runner wording" copies the runners' tool-result wording and must follow it.
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

MONITOR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MONITOR))

PNG = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000"
                    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082")

TASK = "There are two water bottles on the table; find the heavier one, then lay it on its side. {menu} stays as is"
# Runner wording (forceharness/real/receipt_text.py): the object list of the free-task runner (scene_objects.describe)
OBJECTS = ("Scene objects (detected automatically by the wrist depth camera, base frame, meters; about ±1–2 cm; "
           "the side facing away from the camera is not seen):\n"
           "#1 center [0.45, 0.12], highest point [0.45, 0.12, 0.205] (0.200 above the table), from above length 0.065 "
           "× width 0.065, long axis [1.0, 0.0] (+0°)\n"
           "Table z ≈ 0.005.")
# Runner wording (forceharness/real/receipt_text.py): the runner's camera note and the tail of its still-reading line
CAMERA_NOTE = "[Photos this turn, in the relay folder, please open: obs_1_wrist.png] The wrist camera is on the hand."
RECEIPT_A = ("primitive=move_by params={'dx': 0, 'dy': 0, 'dz': 0.03}\n"
             "result: completed; duration 3.80 s\n"
             "end position [+0.450, +0.120, +0.190] m; commanded displacement [+0.000, +0.000, +0.030], actual [+0.000, +0.000, +0.030]\n"
             "contact events (t relative to primitive start):\n"
             "  t=1.50s resistance_drop: Relative tangential force dropped after a peak (|F|=2.9 N)\n"
             "contact state at end: IN CONTACT, |F|=3.1 N, mainly along -z\n"
             "still reading after relaxing the controller: force on the hand [+0.00, +0.00, -3.14] N (|F| 3.14 N) relative to the "
             "empty hand at the start; change since the grasp [+0.00, +0.00, -3.14] N -> if the gripped object has fully left "
             "its support (in the air), it weighs about 320 g\n"
             "net force peak 3.139 N\n" + CAMERA_NOTE + "\n" + OBJECTS)
RECEIPT_B = ("primitive=move_by params={'dx': 0, 'dy': 0, 'dz': 0.03}\n"
             "result: stopped early (motion blocked); duration 3.10 s\n"
             "end position [+0.450, +0.120, +0.170] m; commanded displacement [+0.000, +0.000, +0.030], actual [+0.000, +0.000, +0.010]\n"
             + CAMERA_NOTE + "\n" + OBJECTS)


def obs_text(turn, body, example='{"calls":[{"name":"hold","args":{"seconds":0.5}}]}'):
    # Runner wording (forceharness/real/receipt_text.py): the runners' YOUR MOVE section
    return f"# turn {turn}\n\n{body}\n\n## YOUR MOVE\nSubmit one action per turn.\nJSON: {example}.\n"


def make_free(runs: Path, name="free_test_A", condition="phys2_force", task=TASK, scored=None, task_id=None,
              ended=True, mode="demo", started="2026-10-26T13:00:00+00:00"):
    """One free-task record (task_id=None or "free") or a fixed task on the same runner (task_id=<task>)."""
    ep, relay = runs / "free" / name, runs / "free" / f"{name}_relay"
    ep.mkdir(parents=True)
    relay.mkdir()
    receipt = RECEIPT_A if condition == "phys2_force" else RECEIPT_B
    obj = dict(id=1, centre=[0.45, 0.12, 0.105], top=[0.45, 0.12, 0.205], height=0.2, length=0.065, width=0.065,
               axis=[1.0, 0.0], axis_deg=0.0, partial=False, under_hand=True)
    record = dict(version="free_task/1", mode=mode, condition=condition,
                  interface="force" if condition == "phys2_force" else "guard", instruction=task,
                  menu=["scene", "move_by", "grasp", "done"], model="relay", max_actions=40, max_minutes=40,
                  started=started,
                  actions=[dict(name="move_by", args=dict(dx=0, dy=0, dz=0.03), termination="completed",
                                note="lift to weigh", weight_since_grasp_g=320.0,
                                receipt=dict(text=receipt, states=dict(position=[0.45, 0.12, 0.19])))]
                  + ([dict(name="done", args=dict(success=True, note="the left one is heavier"))] if ended else []),
                  scene_updates=[dict(turn=0, objects=[obj], table_z=0.005, synthetic=True),
                                 dict(turn=1, objects=[obj], table_z=0.005, synthetic=True)],
                  config=dict(f_cap=15, t_cap=3.0, speed_m_s=0.008, tcp_translation=[0, 0, 0.1185]),
                  peak_force_N=3.1, peak_raw_force_N=3.2, ended=ended, finished=ended, fault=None,
                  outcome=scored, scored=scored is not None, model_success=True if ended else None,
                  model_note="the left one is heavier" if ended else None, duration_s=60.0 if ended else None,
                  relay_events=[dict(turn=0, latency_s=10.0), dict(turn=1, latency_s=20.0)],
                  valid_episode=True if ended else None)
    if task_id is not None:
        record.update(task_id=task_id)
    (ep / "episode.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    rows = [[i * 0.05, 0.1 * i, 0.0, -0.05 * i, 0.0, 0.0, 0.01] for i in range(40)]
    report = dict(primitive="move_by", params=dict(dz=0.03), termination="completed", t_start=83.2, t_end=87.0,
                  start_pos=[0.45, 0.12, 0.16], end_pos=[0.45, 0.12, 0.19], commanded_disp=[0, 0, 0.03],
                  actual_disp=[0, 0, 0.03], events=[dict(t=1.5, kind="resistance_drop", force=2.9, detail="drop", pos=None)],
                  wrench_rows=rows, final_wrench=[1, 0, -1, 0, 0, 0], peak_force=4.3, peak_torque=0.4,
                  mean_force_in_contact=0.0, n_steps=40, gripper_opening=0.029, yaw_deg=0.0,
                  final_wrench_net=[1, 0, -1, 0, 0, 0], derived=dict(payload_kg=0.32, stop_displacement_m=0.03),
                  quantities=[dict(name="mass", value=0.32, sigma=0.02, unit="kg", flags=["moderate"])],
                  pos_rows=[[0.45, 0.12, 0.16]] * 40, truth={}, error=None)
    (ep / "action_00.json").write_text(json.dumps(report), encoding="utf-8")
    (relay / "obs_0.md").write_text(obs_text(0, "## SYSTEM\n" + task + "\n" + OBJECTS), encoding="utf-8")
    (relay / "obs_0_wrist.png").write_bytes(PNG)
    (relay / "obs_0_exterior.png").write_bytes(PNG)
    (relay / "cmd_0.json").write_text(json.dumps(dict(calls=[dict(name="move_by", args=dict(dx=0, dy=0, dz=0.03))],
                                                      note="lift to weigh")))
    (relay / "obs_1.md").write_text(obs_text(1, "## TOOL RESULT\n" + json.dumps(dict(text=receipt), ensure_ascii=False)),
                                    encoding="utf-8")
    (relay / "obs_1_wrist.png").write_bytes(PNG)
    if ended:
        (relay / "cmd_1.json").write_text(json.dumps(dict(calls=[dict(name="done", args=dict(success=True))])))
        (relay / "done.json").write_text("{}")
    return ep


def fake_repository(root: Path) -> Path:
    """A repository root the monitor can start runners from: a stub for every script in monitor/tasks.json
    (exits at once, writes nothing), the force calibration file, and the fixed tasks' prompt files."""
    tasks = json.loads((MONITOR / "tasks.json").read_text(encoding="utf-8"))["tasks"]
    for task in tasks:
        script = root / task["script"]
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("raise SystemExit(0)\n")
        if task.get("fixed_instruction_file"):
            prompt = root / task["fixed_instruction_file"]
            prompt.parent.mkdir(parents=True, exist_ok=True)
            prompt.write_text(f"Goal of {task['id']}: a fixed task for the tests.\nSecond line.\n", encoding="utf-8")
    (root / "configs" / "real").mkdir(parents=True)
    (root / "configs" / "real" / "force_calibration.json").write_text("{}")
    return root


@pytest.fixture
def runs(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    return root


class FakeAPI:
    """OpenAI-compatible streaming endpoint; replies come from a list, requests are recorded."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.requests.append(dict(path=self.path, auth=self.headers.get("Authorization"), body=None))
                data = {"data": [{"id": "gpt-old", "created": 1}, {"id": "text-embedding-3", "created": 3},
                                 {"id": "gpt-new", "created": 2}]}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(dict(path=self.path, auth=self.headers.get("Authorization"), body=body))
                text = outer.replies.pop(0) if outer.replies else '{"calls":[{"name":"done","args":{}}]}'
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for i in range(0, len(text), 5):
                    self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': text[i:i + 5]}}]})}\n\n".encode())
                self.wfile.write(b'data: {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3}}\n\n')
                self.wfile.write(b"data: [DONE]\n\n")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def fake_api():
    apis = []

    def make(replies):
        api = FakeAPI(replies)
        apis.append(api)
        return api
    yield make
    for api in apis:
        api.close()
