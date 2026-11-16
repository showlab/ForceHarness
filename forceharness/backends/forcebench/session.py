"""One ForceBench rollout <-> one harness thread (backend layer).

ForceBench's evaluator pulls: when it needs an action it sends an observation and waits for the reply, and the simulation
does not advance while it waits (so model thinking costs no simulated time). Our primitives push: they run a loop that
advances `sim.step()` one tick at a time. Two queues of capacity one connect the two:
  websocket thread: receives observation k -> hands it to the harness thread -> waits for action k -> replies
  harness thread:   `FBSim.step()` hands over action k -> blocks until observation k+1 arrives -> returns the reading
Once the harness episode is over (done / give_up / budget used up), every remaining request is answered with "hold in place"
(the evaluator keeps stepping until its success criterion holds or the step limit is reached).
When the evaluator closes the connection, a harness thread blocked in `step()` receives EvaluatorClosed (a BaseException, so
it passes through the `except Exception` clauses in the primitives and the agent loop).
"""
from __future__ import annotations

import queue
import threading
import traceback
from typing import Any, Callable, Dict, Optional

import numpy as np

_HOLD = object()
_CLOSED = object()


class EvaluatorClosed(BaseException):
    """The evaluator no longer wants actions for this episode (success criterion met / step limit / connection closed)."""


class Session:
    def __init__(self, rollout_id: str, first_obs: Dict[str, Any], worker: Callable[["Session"], Optional[Dict[str, Any]]]):
        self.rollout_id = rollout_id
        self.first_obs = first_obs
        self.last_obs = first_obs
        self.n_requests = 0
        self.finished = False
        self.closed = False
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.grip_cmd = -1.0                     # harness convention: -1 open ... +1 closed (sticky: hold actions reuse the last command)
        self.last_target: Optional[np.ndarray] = None
        self._obs: "queue.Queue[Any]" = queue.Queue(maxsize=1)
        self._act: "queue.Queue[Any]" = queue.Queue(maxsize=1)
        self._worker = worker
        self._thread = threading.Thread(target=self._main, name=f"fh-{rollout_id}", daemon=True)

    # ---------------------------------------------------------------- called from the websocket thread
    def start(self) -> None:
        self._thread.start()

    def on_request(self, obs: Dict[str, Any]) -> np.ndarray:
        """Return this tick's 7-D action (world-frame TCP xyz rpy + ForceBench gripper value). The first request is the
        first_obs that the session was created with."""
        self.n_requests += 1
        self.last_obs = obs
        if self.finished:
            return self.hold_action(obs)
        if self.n_requests > 1:
            self._obs.put(obs)
        while True:
            try:
                a = self._act.get(timeout=0.5)
            except queue.Empty:
                if not self._thread.is_alive():          # the harness thread died unexpectedly: hold in place
                    self.finished = True
                    return self.hold_action(obs)
                continue
            if a is _HOLD:
                return self.hold_action(obs)
            return a

    def close(self) -> None:
        self.closed = True
        try:
            self._obs.put_nowait(_CLOSED)
        except queue.Full:
            pass

    def join(self, timeout: float = 30.0) -> None:
        self._thread.join(timeout)

    # ---------------------------------------------------------------- called from the harness thread
    def submit(self, action: np.ndarray) -> Dict[str, Any]:
        """Hand over one tick's action and block until the next observation."""
        if self.closed:
            raise EvaluatorClosed(self.rollout_id)
        a = np.asarray(action, dtype=np.float32).reshape(7)
        self.last_target = a.copy()
        self._act.put(a)
        obs = self._obs.get()
        if obs is _CLOSED:
            raise EvaluatorClosed(self.rollout_id)
        return obs

    def hold_action(self, obs: Dict[str, Any]) -> np.ndarray:
        """Hold in place: target = measured TCP pose, gripper keeps the last command."""
        st = np.asarray(obs["observation"]["state"], dtype=np.float32)
        a = st.copy()
        a[6] = self.fb_gripper(self.grip_cmd)
        return a

    @staticmethod
    def fb_gripper(grip_cmd: float) -> float:
        """Harness gripper command (-1 open / +1 closed) -> 7th dimension of a ForceBench action (-1 closed / +1 open;
        the action space is `world_tcp_xyz_rpy_gripper_open_minus1_plus1`)."""
        return float(-np.clip(grip_cmd, -1.0, 1.0))

    def _main(self) -> None:
        try:
            self.result = self._worker(self)
        except EvaluatorClosed:
            pass
        except BaseException as e:            # noqa: BLE001 -- record it; never leave the websocket thread hanging
            self.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        finally:
            self.finished = True
            try:
                self._act.put_nowait(_HOLD)       # if the websocket thread is waiting for this tick, give it "hold"
            except queue.Full:
                pass
