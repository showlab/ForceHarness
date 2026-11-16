// Start an episode (task, condition, demo/live, model, instruction) and see what is running now.
import { get, post } from '../api.js';
import { armBadge, banner, clear, dur, epHref, h, modeBadge, seg, statusEl, toast, when } from '../dom.js';

function remember(key, value) {
  try {
    if (value === undefined) return JSON.parse(localStorage.getItem(key) || 'null');
    localStorage.setItem(key, JSON.stringify(value));
  } catch { /* storage may be unavailable */ }
  return null;
}

export class LiveView {
  constructor(root, app) {
    this.root = root;
    this.app = app;
    const saved = remember('fhmon-launch') || {};
    this.s = Object.assign({ task: null, cond: null, mode: 'demo', model: 'mock', instruction: '', taskText: '', name: '' }, saved,
      { inputs: {}, checklist: [], resume: false, resumeHeld: '' });
    if (!this.app.task(this.s.task)) this.s.task = this.app.info.tasks[0]?.id;
    this.profiles = [];
    this.inputs = {};
    this.drafts = {};               // fixed task id -> unsaved edit of its goal
    this.formEl = h('div', { class: 'launch-form' });
    this.previewEl = h('div', { class: 'stack' });
    this.runningEl = h('div', { class: 'run-list' });
    this.sessionsEl = h('div', { class: 'run-list' });
    root.append(
      h('div', { class: 'launch-grid' },
        h('section', { class: 'panel' },
          h('div', { class: 'panel-head' }, h('h2', {}, 'Start an episode'),
            h('span', { class: 'small muted' }, 'The monitor only starts the harness runner script; all guards still apply')),
          h('div', { class: 'panel-body' }, this.formEl)),
        h('div', { class: 'stack', style: { gap: '16px' } },
          h('section', { class: 'panel' }, h('div', { class: 'panel-head' }, h('h2', {}, 'Running now')), this.runningEl),
          h('section', { class: 'panel' }, h('div', { class: 'panel-head' }, h('h2', {}, 'Started from this monitor')), this.sessionsEl))));
    this.load();
    this.poll = setInterval(() => this.refreshLists(), 3000);
  }

  destroy() { clearInterval(this.poll); clearTimeout(this.previewTimer); }

  async load() {
    try {
      const s = await get('/api/settings');
      this.profiles = s.profiles;
    } catch (err) { toast(err.message, 'error'); }
    await this.loadInputs();
    this.renderForm();
    this.refreshLists();
  }

  async loadInputs() {
    const task = this.app.task(this.s.task);
    this.inputs = task?.live ? await get('/api/inputs', { task: this.s.task }).catch(() => ({})) : {};
  }

  get task() { return this.app.task(this.s.task); }

  conds() {
    const task = this.task;
    const ids = task.conditions.map((c) => c.id);
    if (this.s.cond === 'pair' && this.s.mode !== 'live') return ids.slice(0, 2);
    return [ids.includes(this.s.cond) ? this.s.cond : ids[0]];
  }

  scripted() { return this.s.mode === 'demo' && this.s.model === 'scripted'; }

  modelReq() {
    const m = this.s.model || 'mock';
    if (m.startsWith('api:')) return { source: 'api', profile: m.slice(4) };
    return { source: m };
  }

  request() {
    // free task: the text box is the task itself (runner --instruction); otherwise an optional note for the bridge
    const req = { task: this.s.task, conditions: this.conds(), mode: this.s.mode, model: this.modelReq(),
      instruction: this.task?.fixed_instruction_file ? '' : this.task?.instruction ? this.s.taskText : this.s.instruction, name: this.s.name };
    if (this.s.mode === 'live') {
      req.inputs = this.s.inputs;
      req.checklist = this.s.checklist;
      if (this.s.resume && this.task?.live?.resume_held) req.resume_held = this.s.resumeHeld.trim();
    }
    return req;
  }

  save() {
    remember('fhmon-launch', { task: this.s.task, cond: this.s.cond, mode: this.s.mode, model: this.s.model,
      instruction: this.s.instruction, taskText: this.s.taskText });
  }

  renderForm() {
    const f = clear(this.formEl);
    const task = this.task;
    const info = this.app.info;
    if (!(info.allow_live && task.live)) this.s.mode = 'demo';
    const row = (label, ...content) => h('div', { class: 'form-row' }, h('div', { class: 'label' }, label), h('div', { class: 'stack', style: { gap: '6px' } }, content));

    // task
    f.append(row('Task',
      seg(info.tasks.map((t) => ({ value: t.id, label: t.short || t.name, title: t.name })), this.s.task, async (v) => {
        this.s.task = v; this.s.cond = null; this.s.inputs = {}; this.s.checklist = []; this.s.resume = false; this.s.resumeHeld = '';
        this.save();
        await this.loadInputs();
        this.renderForm();
      }),
      h('div', { class: 'cond-note' }, task.summary)));

    // condition
    const condOpts = task.conditions.map((c) => ({ value: c.id, label: `${c.id.length <= 2 ? c.id + ' · ' : ''}${c.label}` }));
    condOpts.push({ value: 'pair', label: 'Run both (harness, then baseline)', disabled: this.s.mode === 'live',
      title: this.s.mode === 'live' ? 'Real-robot episodes need a manual scene reset in between' : 'Same demo scene, one episode each' });
    const current = this.s.cond === 'pair' && this.s.mode !== 'live' ? 'pair' : this.conds()[0];
    const condNotes = h('div', { class: 'stack', style: { gap: '2px' } },
      (current === 'pair' ? task.conditions : task.conditions.filter((c) => c.id === current)).map((c) =>
        h('div', { class: 'cond-note row', style: { gap: '6px' } }, armBadge(c.arm, c.label), c.note)));
    f.append(row('Condition', seg(condOpts, current, (v) => { this.s.cond = v; this.save(); this.renderForm(); }), condNotes));

    // mode
    const liveOk = info.allow_live && !!task.live;
    const liveTitle = !info.allow_live ? 'Monitor was started without --allow-live' : !task.live ? (task.live_external?.note || 'This task cannot start a real-robot episode from the monitor') : '';
    f.append(row('Mode',
      seg([{ value: 'demo', label: 'Demo (no robot)' },
        { value: 'live', label: 'Real robot', disabled: !liveOk, title: liveTitle }], liveOk ? this.s.mode : 'demo', (v) => {
        this.s.mode = v; this.s.checklist = []; this.save(); this.renderForm();
      }),
      h('div', { class: 'cond-note' }, this.s.mode === 'demo' ? (task.demo?.note || '') : (task.live?.note || '')),
      !liveOk && task.live_external ? h('div', { class: 'cond-note' }, `Real robot: ${task.live_external.note}`) : null));

    // model
    const demo = this.s.mode === 'demo';
    const sel = h('select', { 'aria-label': 'Model' });
    const grp = h('optgroup', { label: 'API models (configured in Settings)' });
    for (const p of this.profiles) {
      const ok = p.has_key && p.model;
      grp.append(h('option', { value: `api:${p.id}`, disabled: ok ? null : true },
        `${p.name} · ${p.model || 'no model name'}${p.has_key ? '' : ' (no key)'}`));
    }
    if (demo && task.demo?.scripted) sel.append(h('option', { value: 'scripted' }, 'Scripted demo (fixed actions, no model call)'));
    sel.append(grp,
      h('option', { value: 'mock', disabled: demo ? null : true }, 'Mock model (offline pipeline check, no cost)'),
      h('option', { value: 'manual' }, 'Manual: enter each action on this page'),
      h('option', { value: 'external' }, 'External relay: another agent or a person replies elsewhere'));
    const valid = [...sel.querySelectorAll('option')].some((o) => o.value === this.s.model && !o.disabled);
    if (!valid) this.s.model = !demo ? 'manual' : task.demo?.scripted ? 'scripted' : 'mock';
    sel.value = this.s.model;
    sel.addEventListener('change', () => { this.s.model = sel.value; this.save(); this.renderForm(); });
    const modelHelp = {
      scripted: 'Fixed action sequence built into the runner; shows only what the harness and tool results look like.',
      mock: 'Answers with the example action from the observation twice, then ends the episode; only checks the pipeline.',
      manual: 'Each turn the page shows example JSON; edit it and submit.',
      external: 'The monitor watches but does not reply; another agent or a person writes the cmd files.',
    }[this.s.model] || 'API model: gets the full obs text and images each turn; its reply is written to a cmd file. The key stays in memory or an environment variable.';
    f.append(row('Model', sel, h('div', { class: 'cond-note' }, modelHelp)));

    // instruction: the task itself (free task) or an optional note sent with turn 0
    if (task.fixed_instruction_file && !task.fixed_instruction) {
      f.append(row('Fixed task', h('div', { class: 'cond-note', style: { color: 'var(--critical)' } },
        `${task.fixed_instruction_file} is missing: the runner reads this task's goal from it.`)));
    } else if (task.fixed_instruction) {
      // the fixed goal is the task's prompt file: edited here, saved back, read by the runner when an episode starts
      const ta = h('textarea', { rows: 12, class: 'task-text', 'aria-label': 'Fixed task' });
      ta.value = this.drafts[task.id] ?? task.fixed_instruction;
      const saveBtn = h('button', { class: 'btn small', type: 'button' }, 'Save');
      const undo = h('button', { class: 'btn small ghost', type: 'button' }, 'Undo edits');
      const state = h('span', { class: 'small' });
      const sync = () => {
        const dirty = this.drafts[task.id] !== undefined;
        saveBtn.disabled = !dirty;
        undo.disabled = !dirty;
        state.textContent = dirty ? 'Unsaved edits; saved first when you click Start' : '';
      };
      ta.addEventListener('input', () => {
        if (ta.value.trim() === task.fixed_instruction) delete this.drafts[task.id]; else this.drafts[task.id] = ta.value;
        sync();
      });
      saveBtn.addEventListener('click', () => this.saveFixed());
      undo.addEventListener('click', () => { delete this.drafts[task.id]; ta.value = task.fixed_instruction; sync(); });
      sync();
      f.append(row('Fixed task', ta, h('div', { class: 'row' }, saveBtn, undo, state),
        h('div', { class: 'cond-note' }, `Conditions A and B share this text. Saving writes to ${task.fixed_instruction_file}, which is read when the next episode starts; running episodes are unaffected, and each episode record keeps the version it used.`)));
    } else if (task.instruction) {
      const ta = h('textarea', { rows: 4, class: 'task-text', placeholder: task.instruction.placeholder || '', 'aria-label': 'Task' });
      ta.value = this.s.taskText || '';
      ta.addEventListener('input', () => { this.s.taskText = ta.value; this.save(); this.schedulePreview(); });
      f.append(row(task.instruction.label || 'Task', ta, h('div', { class: 'cond-note' }, task.instruction.note || '')));
    } else {
      const canInstruct = !this.scripted() && this.s.model !== 'external';
      const ta = h('textarea', { rows: 3, placeholder: canInstruct ? 'Optional. Sent to the model with the observation on Turn 0; paired episodes get the same text in both conditions. Example: observe before acting, no single move over 6 cm.' : 'This model source cannot receive an extra instruction', disabled: canInstruct ? null : true });
      ta.value = canInstruct ? this.s.instruction : '';
      ta.addEventListener('input', () => { this.s.instruction = ta.value; this.save(); this.schedulePreview(); });
      f.append(row('Extra instruction', ta));
    }

    const name = h('input', { type: 'text', placeholder: 'Optional, e.g. usb_insert_A_01 (default: generated from the time)', value: this.s.name || '' });
    name.addEventListener('input', () => { this.s.name = name.value; this.schedulePreview(); });
    f.append(row('Episode name', name));

    if (this.s.mode === 'live') f.append(row('Real robot', this.liveBox()));
    f.append(row('Command', this.previewEl));
    const go = h('button', { class: `btn ${this.s.mode === 'live' ? 'danger' : 'primary'}`, type: 'button' },
      this.s.mode === 'live' ? 'Start real-robot episode' : 'Start');
    go.addEventListener('click', () => this.launch(go));
    f.append(row('', h('div', { class: 'row' }, go,
      h('span', { class: 'small muted' }, this.s.mode === 'live' ? 'The robot will move. Keep the emergency stop in hand.' : 'No robot connected.'))));
    this.schedulePreview(0);
  }

  liveBox() {
    const live = this.task.live;
    const box = h('div', { class: 'live-box' });
    for (const [name, spec] of Object.entries(this.inputs || {})) {
      const sel = h('select', { 'aria-label': spec.label });
      if (!spec.required) sel.append(h('option', { value: '' }, 'Not used (re-measure at episode start)'));
      else if (!spec.files.length) sel.append(h('option', { value: '' }, '(no files found)'));
      for (const file of spec.files) {
        sel.append(h('option', { value: file.path }, `${file.path}  ·  ${when(new Date(file.mtime * 1000).toISOString())}`));
      }
      const chosen = this.s.inputs[name] ?? (spec.default || (spec.required ? spec.files[0]?.path : '') || '');
      sel.value = chosen;
      this.s.inputs[name] = sel.value;
      sel.addEventListener('change', () => { this.s.inputs[name] = sel.value; this.schedulePreview(); });
      box.append(h('label', { class: 'field' }, h('span', {}, spec.label), sel));
    }
    (live.checklist || []).forEach((text, i) => {
      const cb = h('input', { type: 'checkbox', checked: !!this.s.checklist[i] });
      cb.addEventListener('change', () => { this.s.checklist[i] = cb.checked; this.schedulePreview(); });
      box.append(h('label', { class: 'check' }, cb, h('span', {}, text)));
    });
    const resume = live.resume_held;
    if (resume) {
      // the operator says what the still-closed gripper holds; the runner then starts holding it
      if (!this.s.resumeHeld) this.s.resumeHeld = resume.default || '';
      const held = h('input', { type: 'text', value: this.s.resumeHeld, maxlength: '40', disabled: !this.s.resume,
        placeholder: resume.placeholder || 'What it holds', 'aria-label': 'What the gripper holds' });
      held.addEventListener('input', () => { this.s.resumeHeld = held.value; this.schedulePreview(); });
      const cb = h('input', { type: 'checkbox', checked: this.s.resume });
      cb.addEventListener('change', () => { this.s.resume = cb.checked; held.disabled = !cb.checked; this.schedulePreview(); });
      box.append(h('label', { class: 'check' }, cb, h('span', {}, resume.label)),
        h('label', { class: 'field' }, h('span', {}, 'Gripper holds'), held),
        resume.note ? h('div', { class: 'small' }, resume.note) : null);
    }
    box.append(h('div', { class: 'small' }, '"Interrupt" only ends the script (controller soft stop); it is not an emergency stop. Only one real-robot episode can run at a time.'));
    const video = this.app.info.video || {};
    box.append(h('div', { class: 'small ink2' }, video.enabled && (video.modes || []).includes('live')
      ? `Video: recorded automatically with ${video.camera?.label || 'camera'} once started, saved to ${video.dir}/; watch it on the episode page or the Videos tab.`
      : `Video: this episode will not be recorded (${video.error || 'real-robot recording is off in the config'}).`));
    return box;
  }

  schedulePreview(delay = 300) {
    clearTimeout(this.previewTimer);
    this.previewTimer = setTimeout(() => this.preview(), delay);
  }

  async preview() {
    clear(this.previewEl);
    try {
      const r = await post('/api/sessions/preview', this.request());
      for (const s of r.sessions) this.previewEl.append(h('div', { class: 'cmd' }, s.command.join(' ')));
      const live = this.s.mode === 'live' && this.task.live;
      if (live && !(this.s.checklist.length === live.checklist.length && this.s.checklist.every(Boolean))) {
        this.previewEl.append(h('div', { class: 'small', style: { color: 'var(--critical)' } }, 'Before starting, tick every item on the real-robot checklist above.'));
      }
      this.previewOk = true;
    } catch (err) {
      this.previewEl.append(h('div', { class: 'small', style: { color: 'var(--critical)' } }, err.message));
      this.previewOk = false;
    }
  }

  // Save the edited fixed goal; false when the server refused it (the toast says why).
  async saveFixed() {
    const task = this.task;
    const text = this.drafts[task.id];
    if (text === undefined) return true;
    try {
      const r = await post('/api/tasks/instruction', { task: task.id, text, base: task.fixed_instruction });
      task.fixed_instruction = r.fixed_instruction;
      delete this.drafts[task.id];
      toast(`Saved to ${task.fixed_instruction_file}`);
      this.renderForm();
      return true;
    } catch (err) {
      toast(err.message, 'error');
      return false;
    }
  }

  async launch(btn) {
    const held = this.s.mode === 'live' && this.s.resume ? `\nGripper holds at start: ${this.s.resumeHeld.trim()}` : '';
    const edited = this.drafts[this.task?.id] !== undefined ? '\nThe edited task description is saved first.' : '';
    if (this.s.mode === 'live' && !confirm(`Start this real-robot episode now?\nThe robot will move. Confirm you are beside it with the emergency stop in hand.${held}${edited}`)) return;
    btn.disabled = true;
    if (!(await this.saveFixed())) { btn.disabled = false; return; }
    try {
      const r = await post('/api/sessions/launch', this.request());
      const first = r.sessions[0];
      this.s.name = '';
      toast(r.sessions.length > 1 ? `Started: ${first.id} (${r.sessions[1].id} runs next automatically)` : `Started: ${first.id}`);
      location.hash = epHref(first.key);
    } catch (err) {
      toast(err.message, 'error');
    } finally {
      btn.disabled = false;
    }
  }

  async refreshLists() {
    try {
      const [eps, ss] = await Promise.all([get('/api/episodes'), get('/api/sessions')]);
      this.renderRunning(eps.episodes.filter((e) => !['ended', 'stale'].includes(e.status)));
      this.renderSessions(ss.sessions);
    } catch { /* connection indicator shows it */ }
  }

  renderRunning(list) {
    clear(this.runningEl);
    if (!list.length) {
      this.runningEl.append(h('div', { class: 'empty' }, 'No episode is running. Episodes started from a terminal also appear here.'));
      return;
    }
    for (const e of list) this.runningEl.append(this.item(e));
  }

  item(e) {
    const task = this.app.task(e.family);
    const el = h('div', { class: 'run-item', role: 'link', tabindex: '0' },
      h('div', { class: 'stack', style: { gap: '4px', minWidth: 0 } },
        h('div', { class: 'title' }, `${task?.name || e.family} · ${e.name}`),
        h('div', { class: 'row' }, armBadge(e.arm), modeBadge(e.mode), statusEl(e.status, e.fault))),
      h('div', { class: 'small muted', style: { textAlign: 'right' } },
        h('div', {}, e.model || '—'),
        h('div', { class: 'num' }, `Actions: ${e.metrics?.actions ?? 0} · ${dur(e.metrics?.duration_s)}`)));
    const go = () => { location.hash = epHref(e.key); };
    el.addEventListener('click', go);
    el.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') go(); });
    return el;
  }

  renderSessions(list) {
    clear(this.sessionsEl);
    if (!list.length) {
      this.sessionsEl.append(h('div', { class: 'empty' }, 'None yet. Episodes started from this page are listed here.'));
      return;
    }
    for (const s of list.slice(0, 10)) {
      const el = h('div', { class: 'run-item', role: 'link', tabindex: '0' },
        h('div', { class: 'stack', style: { gap: '4px', minWidth: 0 } },
          h('div', { class: 'title' }, `${s.task_name} · ${s.id}`),
          h('div', { class: 'row' }, armBadge(s.arm, s.condition_label), modeBadge(s.mode), statusEl(s.status))),
        h('div', { class: 'small muted', style: { textAlign: 'right' } }, h('div', {}, s.model?.label || '—'),
          h('div', {}, when(s.started))));
      el.addEventListener('click', () => { location.hash = epHref(s.key); });
      this.sessionsEl.append(el);
    }
    const failed = list.find((s) => s.status === 'failed');
    if (failed) this.sessionsEl.append(h('div', { style: { padding: '10px 14px' } },
      banner('critical', `${failed.id} did not run normally (exit ${failed.exit_code}). See ${failed.id}/runner.log.`)));
  }
}
