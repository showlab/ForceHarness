// One episode, live or finished: model turns on the left; force, the global-camera video, the
// force -> translation detail of the selected action, cameras and settings on the right.
import { get, post, stream } from '../api.js';
import {
  append, armBadge, banner, clear, dur, h, modeBadge, num, OUTCOME_TEXT, resultEl, statusEl, tile, toast, tokens, vec, when,
} from '../dom.js';
import { FORCE_SERIES, TimeChart, TORQUE_SERIES } from '../chart.js';
import { brief, callOf, callStrip, receiptEl, turnCard } from '../turns.js';
import { LivePreview, videoFacts, videoStatusEl } from '../video.js';
import { labelCell } from '../label.js';

const VIDEO_SETTLED = ['complete', 'partial', 'failed', 'interrupted'];

const LIVE_WINDOW = 60;

export class EpisodeView {
  constructor(root, app, key) {
    this.root = root;
    this.app = app;
    this.key = key;
    this.ep = null;
    this.selected = null;
    this.follow = true;
    this.live = {};
    this.manual = null;
    this.session = null;
    this.bridge = null;
    this.clock = null;
    this.forceMeta = null;
    this.forceLoaded = false;
    this.pendingForce = [];
    this.lastEventAt = Date.now();
    this.detailChart = null;
    this.detailKey = null;
    this.attachOpen = false;
    this.video = null;           // this episode's recording in runs/videos, looked up by episode
    this.videoChecked = false;
    this.videoNext = 0;
    this.videoShown = null;
    this.videoPreview = null;
    this.player = null;
    this.videoTurn = null;

    this.headEl = h('div', { class: 'ep-head' });
    this.bannerEl = h('div', { class: 'stack' });
    this.tilesEl = h('div', { class: 'tiles' });
    this.turnsEl = h('div', { class: 'turns', style: { maxHeight: 'calc(100vh - 250px)', overflow: 'auto' } });
    this.followBtn = h('button', { class: 'btn small ghost', type: 'button', 'aria-pressed': 'true' }, 'Follow latest');
    this.followBtn.addEventListener('click', () => { this.follow = true; this.render(); });
    this.forceChart = new TimeChart({ series: FORCE_SERIES, height: 230, unit: 'N', title: 'Wrist force', empty: 'No force data for this episode yet' });
    this.torqueChart = new TimeChart({ series: TORQUE_SERIES, height: 110, unit: 'N·m', title: 'Wrist torque', digits: 3,
      empty: 'No torque data for this episode yet' });
    this.forceHint = h('span', { class: 'small muted' });
    this.windowBtns = h('div', { class: 'seg' });
    this.detailTitle = h('h2', {}, 'Action details');
    this.detailEl = h('div', { class: 'panel-body' });
    this.cameraEl = h('div', { class: 'panel-body' });
    this.infoEl = h('div', { class: 'panel-body' });
    this.stripEl = h('div', { class: 'panel-body' });
    this.stripHint = h('span', { class: 'small muted' });
    this.videoHint = h('span', { class: 'small muted' });
    this.videoStatus = h('span');
    this.videoEl = h('div', { class: 'panel-body' });
    this.videoPanel = h('section', { class: 'panel', hidden: true },
      h('div', { class: 'panel-head' }, h('h2', {}, 'Video'), this.videoHint, this.videoStatus), this.videoEl);

    root.append(
      this.headEl, this.bannerEl, this.tilesEl,
      h('section', { class: 'panel', style: { marginTop: '14px' } },
        h('div', { class: 'panel-head' }, h('h2', {}, 'Primitive calls in order'), this.stripHint), this.stripEl),
      h('div', { class: 'ep-grid' },
        h('section', { class: 'panel ep-left' },
          h('div', { class: 'panel-head' }, h('h2', {}, 'Each turn: ① Input → ② Decision → ③ Primitive call → ④ Return'), this.followBtn),
          this.turnsEl),
        h('div', { class: 'ep-right' },
          h('section', { class: 'panel' },
            h('div', { class: 'panel-head' }, h('h2', {}, 'Force'), this.forceHint, this.windowBtns),
            h('div', { class: 'panel-body stack' }, this.forceChart.el, this.torqueChart.el)),
          this.videoPanel,
          h('section', { class: 'panel' }, h('div', { class: 'panel-head' }, this.detailTitle), this.detailEl),
          h('section', { class: 'panel' }, h('div', { class: 'panel-head' }, h('h2', {}, 'Camera')), this.cameraEl),
          h('section', { class: 'panel' }, h('div', { class: 'panel-head' }, h('h2', {}, 'Episode settings')), this.infoEl))));
    this.headEl.append(h('div', { class: 'muted' }, 'Loading…'));
    this.es = stream(key, (ev) => this.onEvent(ev));
    this.timer = setInterval(() => this.tick(), 1000);
  }

  destroy() {
    this.es.close();
    clearInterval(this.timer);
    this.forceChart.destroy();
    this.torqueChart.destroy();
    if (this.detailChart) this.detailChart.destroy();
    if (this.videoPreview) this.videoPreview.destroy();
  }

  get task() { return this.ep ? this.app.task(this.ep.family) : null; }
  get running() { return this.ep && !['ended', 'stale'].includes(this.ep.status); }
  get target() { return this.session ? this.session.id : this.key; }

  // ---------------------------------------------------------------- events
  onEvent(ev) {
    this.lastEventAt = Date.now();
    if (ev.type === 'snapshot') {
      const first = !this.ep;
      this.ep = ev.episode;
      if (first) {
        // a finished episode is read from the top; a running one follows the newest turn
        if (!this.running && this.ep.turns?.length) {
          this.follow = false;
          const firstMove = this.ep.turns.find((t) => t.report || t.receipt);
          this.selected = (firstMove || this.ep.turns[0]).n;
        }
        this.loadForce();
      }
      this.render();
    } else if (ev.type === 'force') {
      this.onForce(ev);
    } else if (ev.type === 'model') {
      this.onModel(ev);
    } else if (ev.type === 'session') {
      this.session = ev.session;
      if (this.ep) { this.renderHead(); this.renderBanners(); this.renderInfo(); } else this.renderSessionOnly();
    }
  }

  async loadForce() {
    try {
      const f = await get('/api/episode/force', { key: this.key });
      this.forceMeta = f;
      this.clock = f.clock;
      const rows = f.points || [];
      const lastX = rows.length ? rows[rows.length - 1][0] : -Infinity;
      for (const p of this.pendingForce) if (p[0] > lastX) rows.push(p);
      this.pendingForce = [];
      this.forceChart.setRows(rows);
      this.torqueChart.setRows(rows);
      this.forceLoaded = true;
      if (this.running) this.notePeak(rows);
      this.applyPhases();
      this.setWindow(this.running ? LIVE_WINDOW : null);
      this.updateMarks();
      this.renderForceHint();
    } catch (err) {
      this.forceHint.textContent = `Failed to read force data: ${err.message}`;
    }
  }

  // Largest raw |F| seen so far; shown while the runner has not written its own peak yet.
  notePeak(rows) {
    let changed = false;
    for (const r of rows || []) if (!(this.livePeak >= r[4])) { this.livePeak = r[4]; changed = true; }
    const m = this.ep?.metrics || {};
    if (changed && this.peakTile && (m.peak_raw_force_N ?? m.peak_force_N) == null) {
      this.peakTile.querySelector('.value').textContent = `${num(this.livePeak, 1)} N`;
    }
  }

  // labelled motion phases of the stream, if it has any: history from the overview, then live ones
  applyPhases() {
    const live = this.livePhases || [];
    const start = live.length ? live[0].x0 : Infinity;
    const phases = (this.forceMeta?.phases || []).filter((p) => p.x1 < start).concat(live);
    this.forceChart.setPhases(phases);
    this.torqueChart.setPhases(phases);
  }

  onForce(ev) {
    if (!this.ep || this.ep.status === 'ended') return;
    if (ev.clock) this.clock = ev.clock;
    const rows = ev.points || [];
    if (ev.phases) this.livePhases = ev.phases;
    this.notePeak(rows);
    if (!this.forceLoaded) { this.pendingForce.push(...rows); return; }
    if (ev.phases) this.applyPhases();
    this.forceChart.append(rows);
    this.torqueChart.append(rows);
    this.lastForce = rows[rows.length - 1];
    this.renderForceHint();
  }

  onModel(ev) {
    if (ev.kind === 'state') {
      this.bridge = Object.assign({}, this.bridge, ev);
      if (ev.status === 'manual_wait' && ev.turn !== null && ev.turn !== undefined) {
        this.manual = { turn: ev.turn, example: ev.example };
      } else if (this.manual && ev.status !== 'manual_wait') {
        this.manual = null;
      }
      if (this.ep) { this.renderHead(); this.renderBanners(); this.renderTurns(); }
      return;
    }
    const t = ev.turn;
    if (t === undefined || t === null) return;
    const st = this.live[t] || (this.live[t] = { text: '', thinking: '', active: false, error: null });
    if (ev.kind === 'request') Object.assign(st, { text: '', thinking: '', active: true, error: null });
    else if (ev.kind === 'delta') { st[ev.channel === 'thinking' ? 'thinking' : 'text'] += ev.text; st.active = true; }
    else if (ev.kind === 'done') {
      Object.assign(st, { active: false, error: null });
      if (ev.text) st.text = ev.text;
      if (ev.thinking) st.thinking = ev.thinking;
      if (this.manual && this.manual.turn === t) this.manual = null;
    } else if (ev.kind === 'error') {
      st.error = ev.final ? `Model call failed: ${ev.message}` : `${ev.message} (retrying)`;
      if (ev.final) st.active = false;
      if (ev.final) this.renderBanners();
    } else if (ev.kind === 'manual_wait') {
      this.manual = { turn: t, example: ev.example };
    }
    this.renderTurn(t);
  }

  // ---------------------------------------------------------------- rendering
  render() {
    if (!this.ep) return;
    const turns = this.ep.turns || [];
    if (this.follow && turns.length) {
      // newest step that moved the robot; else the newest decision; else the newest observation
      const rev = [...turns].reverse();
      const moved = rev.find((t) => t.report || (t.receipt && t.action?.status !== 'invalid'));
      const n = (moved || rev.find((t) => callOf(t)) || rev[0]).n;
      if (n !== this.selected) this.selected = n;
    }
    this.followBtn.setAttribute('aria-pressed', String(this.follow));
    this.renderHead();
    this.renderBanners();
    this.renderTiles();
    this.renderStrip();
    this.renderTurns();
    this.renderDetail();
    this.renderCamera();
    this.renderVideo();
    this.renderInfo();
    this.updateMarks();
    if (this.forceLoaded && !this.running && this.wasRunning) {
      this.wasRunning = false;
      this.loadForce();
    }
    if (this.running) this.wasRunning = true;
  }

  renderHead() {
    const ep = this.ep;
    const task = this.task;
    const cond = task?.conditions?.find((c) => c.harness_condition === ep.condition || c.id === ep.condition);
    clear(this.headEl);
    this.updatedEl = h('span', { class: 'small muted num' });
    this.headEl.append(
      h('div', { class: 'ep-title' },
        h('h1', {}, task ? task.name : ep.family),
        armBadge(ep.arm, cond ? `${cond.id === 'A' || cond.id === 'B' ? cond.id + ' · ' : ''}${cond.label}` : ep.condition),
        modeBadge(ep.mode),
        statusEl(ep.status, ep.fault),
        h('span', { class: 'grow' }),
        this.updatedEl),
      h('div', { class: 'ep-sub' },
        h('span', {}, ep.name),
        h('span', {}, `Started ${when(ep.started)}`),
        h('span', {}, `Model ${ep.model || '—'}`),
        ep.session?.pair_id ? h('span', {}, `Pair ${ep.session.pair_id} (${ep.session.pair_role === 'first' ? 'first' : 'second'})`) : null,
        h('span', { class: 'muted' }, ep.version)),
      ep.task_text ? h('div', { class: 'task-line' }, h('b', {}, 'Task'),
        task?.fixed_instruction ? h('div', {}, h('span', {}, ep.task_text.split('\n')[0]),
          h('details', {}, h('summary', {}, 'View fixed procedure'),
            h('div', { style: { whiteSpace: 'pre-wrap' } }, ep.task_text))) : h('span', {}, ep.task_text)) : '',
      ep.result?.label?.note ? h('div', { class: 'task-line' }, h('b', {}, 'Note'),
        h('span', { style: { whiteSpace: 'pre-wrap' } }, ep.result.label.note)) : '',
      (this.nowEl = h('div', { class: 'now-line' })),
      this.controls());
    this.tick();
  }

  // What is happening right now: the primitive being executed, or which step the model is deciding.
  renderNow() {
    if (!this.nowEl || !this.ep) return;
    clear(this.nowEl);
    const turns = this.ep.turns || [];
    const last = turns[turns.length - 1];
    const call = last ? callOf(last) : null;
    const now = Date.now() / 1000;
    if (this.running && call && last.action?.status === 'executing') {
      const since = last.x0 !== null && last.x0 !== undefined && this.ep.started_epoch ? now - this.ep.started_epoch - last.x0 : null;
      this.nowEl.append(h('span', { class: 'live-dot', 'aria-hidden': 'true' }), 'Executing primitive',
        h('span', { class: 'act-name' }, call.name), h('span', { class: 'call-arg' }, brief(call.name, call.args)),
        since !== null ? h('span', { class: 'small muted num' }, `${dur(since)} elapsed`) : '');
    } else if (this.running && last && !call) {
      const since = last.obs?.mtime ? now - last.obs.mtime : null;
      this.nowEl.append(h('span', { class: 'live-dot', style: { background: 'var(--stage-decision)' }, 'aria-hidden': 'true' }),
        `Waiting for the model to decide step ${last.n}`, since !== null ? h('span', { class: 'small muted num' }, `${dur(since)} elapsed`) : '');
    }
    this.nowEl.hidden = !this.nowEl.childNodes.length;
  }

  // Before the runner has written episode.json (starting) or when it never did (failed to start).
  renderSessionOnly() {
    const s = this.session;
    const task = this.app.task(s.task);
    clear(this.headEl);
    this.headEl.append(
      h('div', { class: 'ep-title' }, h('h1', {}, task ? task.name : s.task), armBadge(s.arm, s.condition_label),
        modeBadge(s.mode), statusEl(s.status)),
      h('div', { class: 'ep-sub' }, h('span', {}, s.id), h('span', {}, `Model ${s.model?.label || '—'}`)));
    clear(this.bannerEl);
    if (['failed', 'stopped', 'cancelled'].includes(s.status)) {
      this.bannerEl.append(banner(s.status === 'failed' ? 'critical' : 'warning', [
        h('b', {}, s.status === 'failed' ? 'The runner did not write this episode (failed to start)' : 'This episode ended before it started'),
        ` (exit ${s.exit_code ?? '—'}). `, s.log_url ? h('a', { href: s.log_url, target: '_blank', rel: 'noopener' }, 'View runner.log') : null,
        h('div', { class: 'cmd', style: { marginTop: '8px' } }, (s.command || []).join(' '))]));
    } else {
      this.bannerEl.append(banner('info', 'Runner started; waiting for its first record…'));
    }
  }

  controls() {
    const box = h('div', { class: 'ep-controls' });
    const s = this.session;
    const ep = this.ep;
    if (s && !['ended', 'failed', 'stopped', 'cancelled'].includes(s.status)) {
      if (['api', 'mock', 'manual'].includes(s.model?.source)) {
        box.append(this.button('Finish next turn', 'Make the model call done next turn (written to cmd, marked as ended by the operator)', () => this.stop('finish')));
      }
      box.append(this.button('Interrupt', 'Sends Ctrl+C to the runner: normal stop path (controller soft stop). This is not an emergency stop!', () => this.stop('interrupt'), 'danger'));
      if (s.stopped_by === 'operator_interrupt') box.append(this.button('Force quit', 'Use if the runner has not exited 8 s after Interrupt', () => this.stop('kill'), 'danger'));
      box.append(h('span', { class: 'small muted' }, 'The emergency stop is the safety device; these buttons only end the script.'));
    }
    const attachable = !s && this.running && ep.relay?.exists && !(this.bridge && this.bridge.attached && this.bridge.status !== 'stopped');
    if (attachable) {
      const btn = this.button(this.attachOpen ? 'Collapse' : 'Take over relay with API model', 'This episode was started elsewhere; the monitor replies for the model over the relay protocol', () => {
        this.attachOpen = !this.attachOpen;
        this.renderHead();
      });
      box.append(btn);
      if (this.attachOpen) box.append(this.attachForm());
    }
    if (this.bridge?.attached && this.bridge.status !== 'stopped') {
      box.append(h('span', { class: 'badge' }, `Taken over: ${this.bridge.label || ''}`),
        this.button('Finish next turn', 'Make the model call done next turn', () => this.bridgeCall('/api/bridge/finish')),
        this.button('Release', 'Stop the takeover and hand back to the external relay', async () => {
          await post('/api/detach', { key: this.key });
          this.bridge = null;
          this.renderHead();
        }));
    }
    box.append(h('span', { class: 'small ink2' }, 'Manual label:'), labelCell(ep, (label) => this.labelSaved(label), { showNote: false }));
    const sc = this.task?.score;
    const harnessScored = ep.result?.manual ? ep.result.harness?.scored : ep.result?.scored;
    if (sc && ep.status === 'ended' && !harnessScored && !ep.fault) {
      box.append(h('span', { class: 'small ink2' }, 'Observed result:'));
      for (const o of sc.options) {
        box.append(this.button(o.label, `Write ${o.value} with the harness score command`, () => this.score(o)));
      }
    }
    return box;
  }

  button(label, title, fn, kind = '') {
    const b = h('button', { class: `btn small ${kind}`, type: 'button', title }, label);
    b.addEventListener('click', async () => {
      b.disabled = true;
      try { await fn(); } catch (err) { toast(err.message, 'error'); } finally { b.disabled = false; }
    });
    return b;
  }

  async stop(how) {
    const text = { finish: 'Make the model end this episode next turn?', interrupt: 'Interrupt the runner? (The controller soft stops; this is not an emergency stop.)', kill: 'Force quit the runner process?' }[how];
    if (!confirm(text)) return;
    const r = await post('/api/sessions/stop', { id: this.session.id, how });
    this.session = r.session;
    toast(how === 'finish' ? 'done will be written next turn' : 'Sent');
    this.renderHead();
  }

  async bridgeCall(path) {
    await post(path, { target: this.target });
    toast('Sent');
  }

  // the watcher also pushes a fresh snapshot (the label is part of the fingerprint); this just avoids the wait
  labelSaved(label) {
    const r = this.ep.result || (this.ep.result = {});
    if (r.manual) Object.assign(r, r.harness, { manual: false, harness: undefined });
    r.label = label;
    if (label && label.success !== null && label.success !== undefined) {
      r.harness = { scored: r.scored, success: r.success, outcome: r.outcome };
      Object.assign(r, { scored: true, success: label.success, manual: true });
    }
    this.renderHead();
    this.renderTiles();
  }

  async score(option) {
    if (!confirm(`Write "${option.label}" to episode.json for this episode?\n(Runs the harness score command; this cannot be changed afterward.)`)) return;
    const r = await post('/api/score', { key: this.key, value: option.value });
    toast(`Scored ${r.output ? '· ' + r.output.slice(0, 80) : ''}`);
  }

  attachForm() {
    const profiles = h('select', { 'aria-label': 'Model' });
    profiles.append(h('option', { value: 'manual' }, 'Manual (enter actions on this page)'));
    get('/api/settings').then((s) => {
      for (const p of s.profiles) {
        profiles.append(h('option', { value: `api:${p.id}`, disabled: !p.has_key || !p.model ? true : null },
          `${p.name} · ${p.model || 'no model set'}${p.has_key ? '' : ' (no key)'}`));
      }
    });
    const instr = h('textarea', { rows: 2, placeholder: 'Extra instruction (optional, sent to the model with the observation on Turn 0 only)', style: { width: '100%' } });
    const go = h('button', { class: 'btn primary small', type: 'button' }, 'Take over');
    go.addEventListener('click', async () => {
      const v = profiles.value;
      const model = v === 'manual' ? { source: 'manual' } : { source: 'api', profile: v.slice(4) };
      try {
        const r = await post('/api/attach', { key: this.key, model, instruction: instr.value });
        this.bridge = Object.assign({ attached: true }, r.state || {}, { label: r.model?.label });
        this.attachOpen = false;
        toast('Taken over: the monitor replies from the next turn on');
        this.renderHead();
      } catch (err) { toast(err.message, 'error'); }
    });
    return h('div', { class: 'panel', style: { padding: '10px', display: 'grid', gap: '8px', width: '100%' } },
      h('div', { class: 'small muted' }, 'Use only when nobody else (another agent or a person) is replying in this relay folder; turns answered before the takeover are sent to the model as history.'),
      h('div', { class: 'row' }, profiles, go), instr);
  }

  renderBanners() {
    clear(this.bannerEl);
    const ep = this.ep;
    if (ep.fault) this.bannerEl.append(banner('critical', [h('b', {}, 'This episode ended with a fault: '), ep.fault]));
    if (ep.status === 'stale') this.bannerEl.append(banner('warning', 'No new data for over 15 min; the runner may have exited (end record not written).'));
    for (const c of ep.confirmations_pending || []) {
      this.bannerEl.append(banner('warning', [h('b', {}, `The harness is waiting for the operator to confirm "${c.expected}"`),
        `: ${c.question || c.kind}. Answer it where the runner asks for it; the monitor does not answer for you.`]));
    }
    if (this.bridge?.status === 'error') {
      const retry = this.button('Retry turn', 'Call the model again', () => this.bridgeCall('/api/bridge/retry'));
      this.bannerEl.append(banner('critical', [h('b', {}, `Turn ${this.bridge.turn} model call failed: `), this.bridge.error || '',
        h('div', { class: 'row', style: { marginTop: '6px' } }, retry,
          h('span', { class: 'small muted' }, 'Relay wait limit is 10 min; on timeout this episode ends with RelayTimeout.'))]));
    }
    if (this.session?.status === 'failed') {
      this.bannerEl.append(banner('critical', [h('b', {}, 'The runner did not run normally'),
        ` (exit ${this.session.exit_code ?? '—'}). See ${this.session.id}/runner.log. `, this.session.score_note || '']));
    }
    if (this.session?.score_note && this.session.status !== 'failed') {
      this.bannerEl.append(banner('info', this.session.score_note));
    }
    if (ep.session?.instruction && !ep.task_text) {
      this.bannerEl.append(banner('info', [h('b', {}, 'Operator extra instruction (sent to the model with the observation on Turn 0): '), ep.session.instruction]));
    }
  }

  renderTiles() {
    const ep = this.ep;
    const m = ep.metrics || {};
    const task = this.task;
    clear(this.tilesEl);
    const r = ep.result || {};
    const resultSub = r.model_claim === true ? 'Model reports success' : r.model_claim === false ? 'Model reports failure' : '';
    this.durationTile = tile('Duration', dur(m.duration_s));
    let usageIn = 0, usageOut = 0, cost = 0, any = false;
    for (const t of ep.turns || []) {
      if (t.model && (t.model.usage_in || t.model.usage_out)) {
        any = true;
        usageIn += t.model.usage_in || 0;
        usageOut += t.model.usage_out || 0;
        cost += t.model.cost_usd || 0;
      }
    }
    append(this.tilesEl, [
      tile('Result', resultEl(ep, { compact: true }), resultSub),
      tile('Actions', h('span', { class: 'num' }, `${m.actions ?? 0}${task?.max_actions ? ` / ${task.max_actions}` : ''}`),
        m.invalid ? `Invalid: ${m.invalid}` : `Executed: ${m.executed ?? 0}`),
      this.durationTile,
      (this.peakTile = tile('Peak force', `${num(m.peak_raw_force_N ?? m.peak_force_N ?? this.livePeak, 1)} N`,
        m.peak_force_N !== undefined && m.peak_force_N !== null && m.peak_raw_force_N !== undefined
          ? `In primitive (tared): ${num(m.peak_force_N, 1)} N` : (this.running ? 'Raw reading (this page)' : 'Raw reading'))),
      tile('Model', ep.model || '—', m.decision_latency_s ? `Mean per turn: ${dur(m.decision_latency_s)}` : ''),
      any ? tile('tokens', `${tokens(usageIn)} / ${tokens(usageOut)}`, cost ? `about $${cost.toFixed(3)}` : 'input / output') : null]);
  }

  tick() {
    if (!this.ep) return;
    this.renderNow();
    if (this.updatedEl) {
      const s = Math.round((Date.now() - this.lastEventAt) / 1000);
      this.updatedEl.textContent = this.running ? (s <= 6 ? 'Live' : `Updated ${s} s ago`) : '';
    }
    if (this.running && this.durationTile && this.ep.started_epoch) {
      this.durationTile.querySelector('.value').textContent = dur(Date.now() / 1000 - this.ep.started_epoch);
    }
    this.pollVideo();
  }

  // ---------------------------------------------------------------- video (global camera)
  // Asked every 3 s while it can still change (starting, recording, finishing); then left alone.
  pollVideo() {
    const v = this.video;
    if (this.videoBusy || Date.now() < this.videoNext || (v && VIDEO_SETTLED.includes(v.status))) return;
    if (this.videoChecked && !v && !this.running) return;
    this.videoBusy = true;
    get('/api/video', { key: this.key })
      .then((r) => { this.video = r.video; this.videoChecked = true; this.renderVideo(); })
      .catch(() => {})
      .finally(() => { this.videoBusy = false; this.videoNext = Date.now() + 3000; });
  }

  renderVideo() {
    if (!this.videoChecked) return;
    const v = this.video;
    const cfg = this.app.info.video || {};
    const recorded = (cfg.modes || ['live']).includes(this.ep.mode);
    this.videoPanel.hidden = !v && !recorded;
    if (this.videoPanel.hidden) return;
    clear(this.videoStatus);
    this.videoHint.textContent = v ? `${v.camera?.label || 'Camera'} · ${videoFacts(v)}` : '';
    if (v) this.videoStatus.append(videoStatusEl(v));
    const sig = v ? `${v.id}|${v.status}|${v.url}` : `none|${this.running}|${cfg.enabled}`;
    if (sig === this.videoShown) {
      this.updateVideoJump();
      return;
    }
    this.videoShown = sig;
    if (this.videoPreview) { this.videoPreview.destroy(); this.videoPreview = null; }
    this.player = null;
    clear(this.videoEl);
    if (!v) {
      this.videoEl.append(h('div', { class: 'muted small' }, !cfg.enabled ? `No video for this episode: ${cfg.error || 'recording is off'}`
        : this.running ? 'Waiting for recording to start…' : 'No video for this episode.'));
      return;
    }
    if (['starting', 'recording', 'finalizing'].includes(v.status)) {
      this.videoPreview = new LivePreview(v);
      this.videoEl.append(this.videoPreview.el, h('div', { class: 'small muted', style: { marginTop: '6px' } },
        v.status === 'finalizing' ? 'Episode ended; finalizing the video into a seekable file. Playback is ready in a few seconds.'
          : 'Live view refreshes about once a second; playback is ready a few seconds after the episode ends.'));
      return;
    }
    if (v.status !== 'complete') {
      this.videoEl.append(banner(v.status === 'failed' ? 'critical' : 'warning', [
        h('b', {}, v.status === 'failed' ? 'Recording failed.' : v.status === 'partial' ? 'Only partly recorded.' : 'The recorder did not finish cleanly; seeking may not work.'),
        v.error ? ` ${v.error} ` : ' ',
        v.log_url ? h('a', { href: v.log_url, target: '_blank', rel: 'noopener' }, 'View recording log') : null]));
    }
    if (!v.url) return;
    this.player = h('video', { class: 'video-frame', src: v.url, controls: true, preload: 'metadata', playsinline: true });
    this.videoJump = h('button', { class: 'btn small', type: 'button' });
    this.videoJump.addEventListener('click', () => this.jumpVideo());
    this.videoNow = h('button', { class: 'btn small ghost', type: 'button', title: 'Select this turn on the left' });
    this.videoNow.addEventListener('click', () => {
      if (this.videoTurn !== null) { this.select(this.videoTurn); this.scrollToTurn(this.videoTurn); }
    });
    this.player.addEventListener('timeupdate', () => this.updateVideoNow());
    this.videoEl.append(this.player,
      h('div', { class: 'video-bar' }, this.videoJump, this.videoNow, h('span', { class: 'grow' }),
        h('a', { class: 'small', href: v.url, target: '_blank', rel: 'noopener', title: v.path || v.file }, 'Open in new window')));
    this.updateVideoJump();
    this.updateVideoNow();
  }

  // Video time of a turn: both clocks are this computer's (episode files' mtimes, camera frames).
  videoTimeOf(t) {
    const v = this.video;
    if (!t || t.x0 === null || t.x0 === undefined || !v?.first_frame_unix || !this.ep?.started_epoch) return null;
    return this.ep.started_epoch + t.x0 - v.first_frame_unix;
  }

  updateVideoJump() {
    if (!this.player || !this.videoJump) return;
    const t = (this.ep.turns || []).find((x) => x.n === this.selected);
    const sec = this.videoTimeOf(t);
    this.videoJump.disabled = sec === null;
    this.videoJump.textContent = t ? `▶ Watch from Turn ${t.n} action` : '▶ Select a turn to jump to it';
  }

  jumpVideo() {
    const t = (this.ep.turns || []).find((x) => x.n === this.selected);
    const sec = this.videoTimeOf(t);
    if (sec === null || !this.player) return;
    this.player.currentTime = Math.max(0, sec - 1);        // one second of lead-in
    this.player.play().catch(() => {});
  }

  updateVideoNow() {
    const v = this.video;
    if (!this.player || !this.videoNow || !v?.first_frame_unix || !this.ep?.started_epoch) return;
    const x = v.first_frame_unix + this.player.currentTime - this.ep.started_epoch;
    let cur = null;
    for (const t of this.ep.turns || []) if (t.x0 !== null && t.x0 !== undefined && t.x0 <= x) cur = t;
    const call = cur ? callOf(cur) : null;
    this.videoTurn = cur ? cur.n : null;
    this.videoNow.textContent = `Frame: ${dur(Math.max(0, x))} after start · ${cur ? `Turn ${cur.n}${call ? `: ${call.name}` : ''}` : 'before the first action'}`;
    this.videoNow.disabled = !cur;
  }

  // Which primitive each turn called, in order.
  renderStrip() {
    const turns = this.ep.turns || [];
    clear(this.stripEl);
    const calls = turns.filter((t) => callOf(t)).length;
    this.stripHint.textContent = turns.length ? `Calls: ${calls} · click one to view its turn` : '';
    if (!turns.length) {
      this.stripEl.append(h('div', { class: 'muted small' }, 'No calls yet.'));
      return;
    }
    this.stripEl.append(callStrip(turns, {
      selected: this.selected,
      onSelect: (n) => { this.select(n); this.scrollToTurn(n); },
    }));
  }

  scrollToTurn(n) {
    const el = this.turnsEl.querySelector(`.turn[data-n="${n}"]`);
    if (el) this.turnsEl.scrollTo({ top: el.offsetTop - this.turnsEl.offsetTop - 4, behavior: 'smooth' });
  }

  renderTurns() {
    const keep = new Set([...this.turnsEl.querySelectorAll('details[open]')].map((d) =>
      `${d.closest('.turn')?.dataset.n}|${d.querySelector('summary')?.textContent}`));
    const atBottom = this.turnsEl.scrollHeight - this.turnsEl.scrollTop - this.turnsEl.clientHeight < 40;
    clear(this.turnsEl);
    const turns = this.ep.turns || [];
    if (!turns.length) {
      this.turnsEl.append(h('div', { class: 'empty' }, this.running ? 'Waiting for the runner to write its first observation…' : 'This episode has no turns'));
      return;
    }
    turns.forEach((t, i) => {
      this.turnsEl.append(this.card(t));
      if (i < turns.length - 1) this.turnsEl.append(h('div', { class: 'loop-link' }, `↓ Tool result becomes the input to Turn ${turns[i + 1].n}`));
    });
    for (const d of this.turnsEl.querySelectorAll('details')) {
      if (keep.has(`${d.closest('.turn')?.dataset.n}|${d.querySelector('summary')?.textContent}`)) d.open = true;
    }
    if (this.follow && (atBottom || this.running)) this.turnsEl.scrollTop = this.turnsEl.scrollHeight;
  }

  card(t) {
    const waiting = this.ep.model_source === 'relay' && !this.session && !this.bridge?.attached
      ? 'Waiting for the external relay to reply (another agent or a person). You can also use "Take over relay with API model" above.' : null;
    return turnCard(t, {
      family: this.ep.family,
      selected: t.n === this.selected,
      live: this.live[t.n],
      waitingText: waiting,
      noImages: this.task?.model_images === false ? 'No images this turn: by design, the runner for this task sends text only.'
        : this.ep.mode === 'demo' ? 'No images this turn: the demo has no camera, so the model received text only.'
          : 'No images this turn: the camera captured nothing, so the model received text only.',
      instruction: this.ep.task_text ? null : this.ep.session?.instruction,   // free task: it is in the system prompt
      manual: this.manual && this.manual.turn === t.n ? {
        example: this.manual.example,
        onSubmit: async (turn, text) => {
          try {
            await post('/api/bridge/manual', { target: this.target, turn, text });
            toast('Submitted');
          } catch (err) { toast(err.message, 'error'); }
        },
      } : null,
      onSelect: (n) => this.select(n),
    });
  }

  renderTurn(n) {
    if (!this.ep) return;
    const t = (this.ep.turns || []).find((x) => x.n === n);
    const old = this.turnsEl.querySelector(`.turn[data-n="${n}"]`);
    if (!t || !old) return;
    const fresh = this.card(t);
    old.replaceWith(fresh);
    if (this.follow) this.turnsEl.scrollTop = this.turnsEl.scrollHeight;
  }

  select(n) {
    this.follow = false;
    this.followBtn.setAttribute('aria-pressed', 'false');
    this.selected = n;
    for (const el of this.turnsEl.querySelectorAll('.turn')) el.classList.toggle('selected', Number(el.dataset.n) === n);
    this.renderStrip();
    this.renderDetail();
    this.renderCamera();
    this.updateVideoJump();
  }

  // ---------------------------------------------------------------- force panel
  setWindow(w) {
    this.forceChart.setWindow(w);
    this.torqueChart.setWindow(w);
    clear(this.windowBtns);
    for (const [label, value] of [['Last 60 s', LIVE_WINDOW], ['Whole episode', null]]) {
      const b = h('button', { type: 'button', 'aria-pressed': String(w === value) }, label);
      b.addEventListener('click', () => this.setWindow(value));
      this.windowBtns.append(b);
    }
  }

  updateMarks() {
    if (!this.ep || !this.forceMeta) return;
    const task = this.task;
    const cfg = this.ep.config || {};
    const guards = (task?.guards || []).map((g) => ({ ...g }));
    for (const g of guards) if (g.status === 'critical' && cfg.f_cap) g.value = cfg.f_cap;
    const marks = JSON.stringify([guards, cfg.t_cap]);
    if (marks !== this.marks) {
      this.marks = marks;
      this.forceChart.o.thresholds = guards.map((g) => ({ value: g.value, label: g.label, status: g.status }));
      this.torqueChart.o.thresholds = [{ value: cfg.t_cap || task?.torque_cap || 3, label: 'torque cap', status: 'critical' }];
      this.forceChart._legend();
      this.torqueChart._legend();
    }
    if (this.forceMeta.source === 'actions') {
      this.forceChart.setBands(this.forceMeta.bands);
      this.forceChart.setEvents(this.forceMeta.events);
      this.torqueChart.setBands(this.forceMeta.bands);
      return;
    }
    if (this.clock?.synthetic) return;
    const bands = [], events = [];
    for (const t of this.ep.turns || []) {
      if (t.x0 === null || t.x0 === undefined) continue;
      // the primitive being executed right now: an open band up to the newest sample
      if (t.action?.status === 'executing') bands.push({ x0: t.x0, x1: 1e12, n: t.n, label: `${callOf(t)?.name} executing` });
      if (t.x1 === null || t.x1 === undefined) continue;
      bands.push({ x0: t.x0, x1: t.x1, n: t.n, label: callOf(t)?.name });
      const dur0 = t.report?.duration_s;
      for (const e of t.report?.events || []) {
        if (dur0 && typeof e.t === 'number') events.push({ x: t.x1 - dur0 + e.t, kind: e.kind, force: e.force });
      }
    }
    this.forceChart.setBands(bands);
    this.forceChart.setEvents(events);
    this.torqueChart.setBands(bands);
  }

  renderForceHint() {
    const f = this.forceMeta;
    if (!f) return;
    let text;
    if (f.source === 'actions') text = 'Demo has no continuous force log: forces recorded by each primitive are joined end to end (tared)';
    else text = `Raw reading (joint-torque estimate, not tared)${f.rate_hz ? ` · ${f.rate_hz} Hz` : ''}${this.clock?.synthetic ? ' · demo clock' : ''}`;
    if (this.running && this.lastForce) text += ` · now |F| ${num(this.lastForce[4], 2)} N`;
    this.forceHint.textContent = text;
  }

  // ---------------------------------------------------------------- force -> translation detail
  async renderDetail() {
    const ep = this.ep;
    const t = (ep.turns || []).find((x) => x.n === this.selected);
    const call = t ? callOf(t) : null;
    clear(this.detailEl);
    if (!t) {
      this.detailTitle.textContent = 'Action details';
      this.detailEl.append(h('div', { class: 'muted' }, 'Click a turn on the left to see its force and force translation.'));
      return;
    }
    this.detailTitle.textContent = `Action ${String(t.n).padStart(2, '0')}${call ? ` · ${call.name}` : ''} · force → physical quantity → what the model reads`;
    if (!call) {
      this.detailEl.append(h('div', { class: 'muted' }, 'This turn is still waiting for the model.'));
      return;
    }
    const xl = h('div', { class: 'xlate' });
    this.detailEl.append(xl);
    const baseline = ep.arm === 'baseline';
    // ① the force the primitive measured
    const step1 = h('div', {}, h('div', { class: 'xlate-step' }, h('b', {}, '①'), 'Force measured during the primitive (tared, reaction-force convention)'));
    xl.append(step1);
    if (t.report_url) {
      const holder = h('div', {});
      step1.append(holder);
      const cacheKey = `${this.key}|${t.n}|${t.report?.n_rows}`;
      if (this.detailChart && this.detailKey === cacheKey) {
        holder.append(this.detailChart.el);
      } else {
        if (this.detailChart) this.detailChart.destroy();
        this.detailChart = new TimeChart({ series: FORCE_SERIES, height: 180, unit: 'N', title: `Force during action ${t.n}` });
        this.detailKey = cacheKey;
        holder.append(this.detailChart.el);
        try {
          const rows = await get('/api/episode/action', { key: this.key, n: t.n });
          if (this.detailKey !== cacheKey) return;
          const cfg = ep.config || {};
          this.detailChart.o.thresholds = (this.task?.guards || []).map((g) => ({
            value: g.status === 'critical' && cfg.f_cap ? cfg.f_cap : g.value, label: g.label, status: g.status }));
          this.detailChart._legend();
          this.detailChart.setRows(rows.points);
          this.detailChart.setEvents((rows.events || []).map((e) => ({ x: e.t, kind: e.kind, force: e.force })));
          this.detailChart.setNote('x-axis: seconds since this primitive started. Guard lines apply to raw readings; values here are tared and for reference only.');
        } catch (err) {
          holder.append(h('div', { class: 'muted small' }, `Failed to read: ${err.message}`));
        }
      }
    } else {
      step1.append(h('div', { class: 'small muted' }, a11yNoReport(t)));
    }
    // ② what the harness computed   ③ what the model was told
    const cols = h('div', { class: 'xlate-cols' });
    xl.append(cols);
    cols.append(h('div', {}, h('div', { class: 'xlate-step' }, h('b', {}, '②'), 'What the harness computed from force'), measuredEl(t)));
    const told = h('div', {}, h('div', { class: 'xlate-step' }, h('b', {}, '③'), 'Tool result the model received'));
    if (!t.translation) {
      told.append(h('div', { class: 'small muted' }, t.action?.status === 'done' ? 'The model chose to end this episode; no tool result.' : 'No tool result yet.'));
    } else if (baseline) {
      told.append(h('div', { class: 'gap-note' }, 'Baseline: the force above was measured but not translated for the model. The model received kinematic results only:'),
        receiptEl({ rows: t.translation.rows.filter((r) => r.kind !== 'scene') }));
    } else if (t.translation.has_force) {
      told.append(receiptEl({ rows: t.translation.rows.filter((r) => r.kind === 'force') }),
        h('details', { class: 'fold', style: { marginTop: '6px' } }, h('summary', {}, 'Full tool result (with kinematics)'),
          h('div', { class: 'fold-body' }, receiptEl(t.translation, { hideScene: true }))));
    } else {
      told.append(h('div', { class: 'small muted' }, 'The tool result for this step has no force content.'), receiptEl(t.translation, { hideScene: true }));
    }
    cols.append(told);
  }

  renderCamera() {
    clear(this.cameraEl);
    const turns = this.ep.turns || [];
    const t = turns.find((x) => x.n === this.selected);
    if (!t) {
      this.cameraEl.append(h('div', { class: 'muted' }, 'No images.'));
      return;
    }
    const after = turns.find((x) => x.n === t.n + 1);
    const rowsOf = (turn, label) => {
      const imgs = turn?.obs?.images || [];
      if (!imgs.length) return null;
      return h('div', { class: 'stack', style: { gap: '6px' } }, h('div', { class: 'small ink2' }, label),
        h('div', { style: { display: 'grid', gridTemplateColumns: `repeat(${Math.min(2, imgs.length)}, minmax(0, 1fr))`, gap: '8px' } },
          imgs.map((img) => h('a', { href: img.url, target: '_blank', rel: 'noopener', title: img.name },
            h('img', { src: img.url, alt: img.label, loading: 'lazy', style: { width: '100%', borderRadius: '4px', border: '1px solid var(--hair)', display: 'block' } })))));
    };
    const a = rowsOf(t, `What the model saw when deciding Turn ${t.n} (obs_${t.n})`);
    const b = rowsOf(after, `After the action (obs_${t.n + 1})`);
    if (!a && !b) this.cameraEl.append(h('div', { class: 'muted' }, this.ep.mode === 'demo' ? 'Demo has no camera images.' : 'No images this turn.'));
    if (b) this.cameraEl.append(b);
    if (a) this.cameraEl.append(h('div', { style: { height: '10px' } }), a);
  }

  renderInfo() {
    const ep = this.ep;
    const cfg = ep.config || {};
    clear(this.infoEl);
    const kv = h('dl', { class: 'kv' });
    const add = (k, v) => { if (v !== undefined && v !== null && v !== '') kv.append(h('dt', {}, k), h('dd', {}, v)); };
    add('Record', `${ep.version} · ${ep.key}`);
    add('Condition', `${ep.condition}${ep.interface ? ` · interface ${ep.interface}` : ''}`);
    add('Force cap', cfg.f_cap ? `${cfg.f_cap} N · torque ${cfg.t_cap ?? '—'} N·m · speed ${cfg.speed_m_s ? cfg.speed_m_s * 1000 + ' mm/s' : '—'}` : null);
    add('TCP', cfg.tcp_translation ? `flange + ${vec(cfg.tcp_translation, 4)} m` : null);
    add('Relay folder', ep.relay?.dir ? ep.relay.dir.replace(/^\d+:/, 'runs/') : '(none)');
    add('Force data', ep.stream?.kind ? `${ep.stream.kind} · ${(ep.stream.size / 1e6).toFixed(1)} MB` : '(no continuous log)');
    const s = this.session || ep.session;
    if (s?.model) add('Model source', `${s.model.label}${s.model.provider ? ` · ${s.model.provider}` : ''}${s.model.base_url ? ` · ${s.model.base_url}` : ''}`);
    this.infoEl.append(kv);
    if (s?.command) {
      this.infoEl.append(h('details', { class: 'fold', style: { marginTop: '10px' } }, h('summary', {}, 'Runner command'),
        h('div', { class: 'fold-body' }, h('div', { class: 'cmd' }, s.command.join(' ')))));
    }
  }
}

function a11yNoReport(t) {
  if (t.action?.status === 'invalid') return 'Invalid action was not executed; no force.';
  if (t.action?.status === 'done') return 'Episode ended; no force.';
  if (t.action?.status === 'executing') return 'The robot is still executing this step…';
  return 'No primitive report for this step.';
}

function measuredEl(t) {
  const r = t.report;
  if (!r) return h('div', { class: 'small muted' }, 'No primitive report.');
  const kv = h('dl', { class: 'kv' });
  const add = (k, v) => { if (v !== undefined && v !== null && v !== '') kv.append(h('dt', {}, k), h('dd', {}, v)); };
  add('Termination', r.termination);
  add('Peak', `|F| ${num(r.peak_force, 2)} N · |T| ${num(r.peak_torque, 3)} N·m`);
  add('Force at end', r.final_wrench ? vec(r.final_wrench.slice(0, 3), 2) + ' N' : null);
  for (const e of r.events || []) add(`Event t=${num(e.t, 2)} s`, `${e.kind}${e.detail ? `: ${e.detail}` : ''} (|F| ${num(e.force, 1)} N)`);
  for (const q of r.quantities || []) {
    const val = q.value === null || q.value === undefined ? '—' : `${q.value}${q.sigma !== undefined && q.sigma !== null ? ` ± ${q.sigma}` : ''} ${q.unit || ''}`;
    add(q.name, `${val}${q.flags?.length ? ` (${q.flags.join(', ')})` : ''}`);
  }
  const keep = ['turned_deg', 'commanded_deg', 'radius_m', 'resistance_peak_N', 'resistance_signal', 'contact_distance_m',
    'contact_found', 'payload_kg', 'opening_m', 'closed_on_something', 'stop_displacement_m', 'note'];
  for (const k of keep) if (r.derived && r.derived[k] !== undefined) add(k, typeof r.derived[k] === 'number' ? num(r.derived[k], 3) : String(r.derived[k]));
  if (r.derived?.relay_guard) add('Relay guard', `${r.derived.relay_guard.kind} ${num(r.derived.relay_guard.value, 3)}`);
  return kv;
}

export { OUTCOME_TEXT };
