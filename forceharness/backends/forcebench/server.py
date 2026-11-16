"""ForceHarness as a ForceBench websocket policy server (backend layer): ForceBench's native `forcebench_ws` contract plus
the optional fields that our evaluator copy adds ("camera_calibration", "diag_truth", "privileged").

The evaluator opens one connection per parallel environment; on each connection rollout_id = "<task>:seed=<N>", and step 0
marks a new episode. Each episode gets a Session (see session.py) that runs `worker(session)`, usually "build FBSim, build the
primitives, run one agent-loop episode".
The server is the synchronous websockets server: one thread per connection, no keep-alive pings (a frontier model may think
for tens of seconds while the evaluator waits in recv).
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from forceharness.backends.forcebench import wire
from forceharness.backends.forcebench.session import Session

log = logging.getLogger("fh.forcebench")


def parse_seed(rollout_id: str) -> Optional[int]:
    try:
        return int(rollout_id.rsplit("seed=", 1)[1])
    except (IndexError, ValueError):
        return None


class PolicyServer:
    def __init__(self, make_worker: Callable[[str, Dict[str, Any]], Callable[[Session], Optional[Dict[str, Any]]]],
                 host: str = "127.0.0.1", port: int = 8002, metadata: Optional[Dict[str, Any]] = None,
                 log_dir: Optional[Path] = None):
        self.make_worker = make_worker
        self.host, self.port = host, int(port)
        self.metadata = dict(metadata or {})
        self.metadata.setdefault("contract", wire.POLICY_CONTRACT)
        self.metadata.setdefault("action_space", wire.ACTION_SPACE)
        self.metadata.setdefault("control_rate_hz", wire.CONTROL_RATE_HZ)
        self.log_dir = Path(log_dir) if log_dir else None
        self.sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def _finish(self, sess: Session) -> None:
        sess.close()
        sess.join(120.0)
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            rec = dict(rollout_id=sess.rollout_id, n_requests=sess.n_requests, finished=sess.finished,
                       error=sess.error, result=sess.result)
            with open(self.log_dir / "sessions.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")

    def handler(self, ws) -> None:
        from websockets.exceptions import ConnectionClosed

        ws.send(wire.dumps(self.metadata))
        sess: Optional[Session] = None
        try:
            for raw in ws:
                if isinstance(raw, str):
                    ws.send("expected a binary msgpack request")
                    continue
                req = wire.loads(raw)
                rid = str(req.get("rollout_id", ""))
                step = int(req.get("step", 0))
                if sess is None or rid != sess.rollout_id or (step == 0 and sess.n_requests > 0):
                    if sess is not None:
                        self._finish(sess)
                    sess = Session(rid, req, self.make_worker(rid, req))
                    with self._lock:
                        self.sessions[rid] = sess
                    sess.start()
                    log.info("new rollout %s", rid)
                action = sess.on_request(req)
                ws.send(wire.action_reply(action[None, :]))
        except ConnectionClosed:
            pass
        finally:
            if sess is not None:
                self._finish(sess)

    def serve_forever(self) -> None:
        from websockets.sync.server import serve

        kwargs = dict(compression=None, max_size=None)
        try:
            server = serve(self.handler, self.host, self.port, ping_interval=None, **kwargs)
        except TypeError:                       # older websockets: the sync server has no ping_interval
            server = serve(self.handler, self.host, self.port, **kwargs)
        with server:
            log.info("harness policy server on ws://%s:%d", self.host, self.port)
            print(f"FH_POLICY_SERVER_READY ws://{self.host}:{self.port}", flush=True)
            server.serve_forever()
