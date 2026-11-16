"""File relays of the real-robot runner: the model relay and the operator's grasp/release confirmations.

The model is never called from the runner itself. Each turn RelayClient writes the observation to
<relay>/obs_<n>.md (plus obs_<n>_wrist.png and obs_<n>_exterior.png, copied by the runner) and
pending.json {"turn": n, "waiting": true}, then waits for <relay>/cmd_<n>.json
{"calls": [{"name": ..., "args": {...}}], "note": "..."} written by whoever answers: the web monitor (which calls
the model through an API or a CLI) or a person. While it waits, wait_tick() keeps the robot-side hold alive.
When the episode ends, done.json is written. Only public text crosses this boundary: no episode record,
evaluation truth or telemetry.

OperatorConfirm is the optional human-in-the-loop path (live runs without --auto-confirm): every grasp and
release asks the operator through a request file and waits for a reply file bound to that request.
"""
import json
from pathlib import Path
import time
import uuid

from forceharness.agent.llm import LLMResponse, ToolCall
from forceharness.real import receipt_text as T


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


class RelayTimeout(TimeoutError):
    pass


class RelayClient:
    """Public-only file relay for one episode, with the LLM client's chat() contract.

    wait_tick is installed by the runner on the control thread, so guards continue
    while waiting. The wall clock timeout is independent of the demo clock.
    """
    def __init__(self, session_dir, *, timeout_s=600., poll_s=.05,
                 clock=time.monotonic, sleep=time.sleep, action_prompt=None):
        if not 0 < timeout_s <= 600 or not 0 < poll_s <= 1:
            raise ValueError('Relay timeout must be in (0,600], poll in (0,1]')
        self.dir = Path(session_dir)
        self.dir.mkdir(parents=True, exist_ok=False)  # Never consume stale commands.
        self.timeout_s, self.poll_s = timeout_s, poll_s
        self.clock, self.sleep = clock, sleep
        self.wait_tick = lambda: None
        self.turn = 0
        self.action_prompt = action_prompt
        self.events = []  # Private operator record, never copied into done.json.

    def chat(self, messages, tools):
        turn = self.turn
        out = [T.TURN_HEAD+str(turn)]
        if turn == 0:
            for msg in messages:
                if msg['role'] == 'system':
                    out.append(T.SYSTEM_HEAD+'\n'+msg['text'])
                elif msg['role'] == 'user':
                    out.extend(p['text'] for p in msg['parts'] if p['type'] == 'text')
        else:
            last = max(i for i, m in enumerate(messages) if m['role'] == 'assistant')
            for msg in messages[last+1:]:
                if msg['role'] == 'tool':
                    receipt = json.loads(msg['parts'][0]['text'])
                    # Re-project at the file boundary; no metadata/telemetry.
                    out.append(T.TOOL_RESULT_HEAD+'\n'+json.dumps(
                        {k: receipt[k] for k in ('text', 'states')}, ensure_ascii=False))
        out.append(self.action_prompt or T.YOUR_MOVE)
        (self.dir/f'obs_{turn}.md').write_text('\n\n'.join(out)+'\n')
        atomic_json(self.dir/'pending.json', dict(turn=turn, waiting=True))
        start = self.clock()
        cmd = self.dir/f'cmd_{turn}.json'
        while True:
            tick_start = self.clock()
            self.wait_tick()
            if self.clock()-start >= self.timeout_s:
                raise RelayTimeout(f'No relay command within {self.timeout_s:g}s')
            if cmd.exists():
                # Writers may not use an atomic rename; wait for complete JSON.
                try:
                    data = json.loads(cmd.read_text())
                except json.JSONDecodeError:
                    data = None
                if data is not None:
                    break
            self.sleep(max(0., self.poll_s-(self.clock()-tick_start)))
        calls = data.get('calls', [data] if 'name' in data else []) if isinstance(data, dict) else data
        if not isinstance(calls, list) or not calls:
            calls = [dict(name='invalid', args={})]
        first = calls[0] if isinstance(calls[0], dict) else {}
        ignored = len(calls)-1
        event = dict(turn=turn, ignored_call_count=ignored, ignored_calls=calls[1:],
                     latency_s=self.clock()-start)
        self.events.append(event)
        atomic_json(self.dir/'pending.json', dict(turn=turn, waiting=False))
        self.turn += 1
        return LLMResponse(str(data.get('note', '')) if isinstance(data, dict) else '',
                           [ToolCall(f'r{turn}', first.get('name'), first.get('args', {}))],
                           0, 0, event['latency_s'])

    def finish(self):
        # Final observation intentionally contains no scores or exception detail.
        atomic_json(self.dir/'done.json', dict(ended=True, text=T.EPISODE_OVER))
        (self.dir/'pending.json').unlink(missing_ok=True)


class OperatorConfirm:
    """Fresh per-action confirmation by the operator; never infer consent from time or a file's existence.

    Writes <dir>/<id>.request.json and prints a CHAT_CONFIRM_REQUEST line; the operator's tool writes
    <dir>/<id>.reply.json {"request_id": <id>, "source": "chat", "user_text": "...", "answer": "CLOSE"|"OPEN"}.
    Any other answer is a decline, and user_text becomes the hint in the model's tool result."""
    WORDS = dict(close='CLOSE', open='OPEN')

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.sequence = 0

    def __call__(self, kind, context, tick):
        self.sequence += 1
        request_id = f'{self.sequence:02d}_{uuid.uuid4().hex}'
        stem = self.directory/request_id
        reply_path = stem.with_suffix('.reply.json')
        request = dict(request_id=request_id, kind=kind, expected=self.WORDS[kind],
                       context=context, reply_path=str(reply_path.resolve()))
        atomic_json(stem.with_suffix('.request.json'), request)
        print('CHAT_CONFIRM_REQUEST '+json.dumps(request, ensure_ascii=False, default=str), flush=True)
        result = dict(request_id=request_id, confirmed=False)
        try:
            while True:
                tick()
                if not reply_path.exists():
                    continue
                reply = json.loads(reply_path.read_text())
                if (reply.get('request_id') != request_id or reply.get('source') != 'chat'
                        or not isinstance(reply.get('user_text'), str) or not reply['user_text'].strip()):
                    raise ValueError('Reply must bind this request and preserve the explicit user answer')
                result.update(confirmed=reply.get('answer') == request['expected'], user_text=reply['user_text'])
                return result['confirmed']
        finally:
            atomic_json(stem.with_suffix('.result.json'), result)
