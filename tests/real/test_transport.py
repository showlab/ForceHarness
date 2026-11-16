"""The SSH transport against the real worker process (worker_fixture.py serves a synthetic plant over stdin/stdout)."""
import json
from pathlib import Path
import subprocess
import sys
import time
import pytest

from forceharness.real.config import BackendConfig
from forceharness.real.backend import ContinuousBackend
from forceharness.real.transport import SSHPort,worker_command
from forceharness.real.plant import DemoPort


def port(tmp_path,monkeypatch,motion=True):
    actual = subprocess.Popen
    fixture = Path(__file__).with_name('worker_fixture.py')
    def spawn(argv,**kw):
        return actual([sys.executable,str(fixture)]+(['--enable-motion'] if motion else []),**kw)
    monkeypatch.setattr(subprocess,'Popen',spawn)
    cfg = BackendConfig(motion_enabled=motion)
    return SSHPort(cfg,tmp_path),cfg


def test_protocol_keeps_one_controller_and_logs(tmp_path,monkeypatch):
    p,cfg = port(tmp_path,monkeypatch)
    with ContinuousBackend(p,config=cfg) as b:
        b.set_kp(75.)
        with b.motion(3.):
            for _ in range(3): b.step([.00025,0,0])
        assert not b._state['policy_running']
    rows = [json.loads(s) for s in (tmp_path/'nuc.jsonl').read_text().splitlines()]
    assert sum(r['type']=='begin' for r in rows)==1
    release=next(i for i,r in enumerate(rows) if r['type']=='release_start')
    assert len([r for r in rows[:release] if r['type']=='target'])==3
    assert len([r for r in rows[release:] if r['type']=='target'])>=5
    assert any(r['type']=='gains' and r['Kx'][0]==375 for r in rows)
    assert p.proc.poll()==0


def test_protocol_loss_and_readonly(tmp_path,monkeypatch):
    p,cfg = port(tmp_path,monkeypatch)
    p.open(); p.request('begin',budget=3.)
    time.sleep(.4)
    with pytest.raises(RuntimeError,match='NUC'): p.request('state')
    assert p.proc.poll() is not None
    rows = [json.loads(s) for s in (tmp_path/'nuc.jsonl').read_text().splitlines()]
    assert any(r['type']=='fault' and 'lease' in r['error'] for r in rows)
    monkeypatch.undo()
    p,cfg = port(tmp_path/'readonly',monkeypatch,False)
    with ContinuousBackend(p,config=cfg) as b:
        assert not b._state['policy_running']
    assert '--enable-motion' not in worker_command(cfg)
    assert '--enable-motion' not in worker_command(BackendConfig(motion_enabled=True),compile_only=True)


def test_eof_stops_active_controller(tmp_path,monkeypatch):
    p,cfg = port(tmp_path,monkeypatch)
    p.open(); p.request('begin',budget=3.)
    p.close()
    rows = [json.loads(s) for s in (tmp_path/'nuc.jsonl').read_text().splitlines()]
    assert rows[-1]['type']=='end'
    assert p.proc.poll()==0


@pytest.mark.parametrize('delay,accepted',[(.20,True),(.36,False)])
def test_step_reply_jitter_has_bounded_300ms_wait(tmp_path,delay,accepted):
    """Delayed acknowledgement only; remote guard/lease are not changed."""
    import io
    import threading
    from types import SimpleNamespace
    from forceharness.real.types import BackendError
    p=SSHPort(BackendConfig(motion_enabled=True),tmp_path)
    p.proc=SimpleNamespace(poll=lambda:None,stdin=io.StringIO())
    closed=[]
    p.close=lambda:closed.append(True)
    timer=threading.Timer(delay,lambda:p.inbox.put(dict(type='reply',id=1,ok=True)))
    timer.start()
    try:
        if accepted:
            assert p.request('step',dpos=[0,0,0])['ok']
            assert not closed
        else:
            with pytest.raises(BackendError,match='reply timeout'):p.request('step',dpos=[0,0,0])
            assert closed
    finally:timer.join()
