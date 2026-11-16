// Force harness vs baseline for one task: success rate with a 95% interval, per-arm numbers,
// the episode table, and one pair side by side turn by turn (what each model was told).
import { get } from '../api.js';
import { armBadge, clear, dur, epHref, h, modeBadge, num, pct, resultEl, seg, statusEl, termChip, tile, when } from '../dom.js';
import { ciChart } from '../chart.js';
import { callOf, receiptEl } from '../turns.js';
import { args as fmtArgs } from '../dom.js';

export class CompareView {
  constructor(root, app) {
    this.root = root;
    this.app = app;
    this.task = app.info.tasks[0]?.id;
    this.mode = 'all';
    this.instruction = '';          // free task: A vs B only on the same task text
    this.data = null;
    this.headEl = h('div', { class: 'filters' });
    this.bodyEl = h('div', { class: 'stack', style: { gap: '16px' } });
    root.append(h('div', { class: 'section-title' }, h('h1', {}, 'Compare'),
      h('span', { class: 'hint' }, 'Force harness (tool results include force translation) vs baseline (same actions and cameras, no force in tool results)')), this.headEl, this.bodyEl);
    this.renderFilters();
    this.load();
  }

  destroy() {}

  renderFilters() {
    clear(this.headEl);
    this.headEl.append(
      seg(this.app.info.tasks.map((t) => ({ value: t.id, label: t.short || t.name, title: t.name })), this.task, (v) => { this.task = v; this.instruction = ''; this.load(); }),
      seg([{ value: 'all', label: 'Real robot + demo' }, { value: 'live', label: 'Real robot only' }, { value: 'demo', label: 'Demo only' }],
        this.mode, (v) => { this.mode = v; this.load(); }));
    const texts = this.data?.family === this.task ? this.data.instructions || [] : [];
    if (this.app.task(this.task)?.instruction) {
      const sel = h('select', { 'aria-label': 'Task text', class: 'task-pick' },
        h('option', { value: '' }, `All task texts (episodes: ${texts.reduce((n, t) => n + t.n, 0)}, different tasks mixed)`),
        texts.map((t) => h('option', { value: t.text }, `${t.text.length > 60 ? t.text.slice(0, 60) + '…' : t.text} (episodes: ${t.n})`)));
      sel.value = this.instruction;
      sel.addEventListener('change', () => { this.instruction = sel.value; this.load(); });
      this.headEl.append(sel);
    }
  }

  async load() {
    clear(this.bodyEl).append(h('div', { class: 'muted' }, 'Loading…'));
    try {
      const q = { task: this.task, mode: this.mode };
      if (this.instruction) q.instruction = this.instruction;
      this.data = await get('/api/compare', q);
      this.renderFilters();
      this.render();
    } catch (err) {
      clear(this.bodyEl).append(h('div', { class: 'empty' }, `Failed to read: ${err.message}`));
    }
  }

  render() {
    const d = this.data;
    const task = this.app.task(this.task);
    clear(this.bodyEl);
    const H = d.arms.harness, B = d.arms.baseline;
    if (task?.instruction && !d.instruction && (d.instructions || []).length > 1) {
      this.bodyEl.append(h('div', { class: 'banner info' }, h('div', {}, h('b', {}, 'Different tasks mixed: '),
        'Free-task success rates are only meaningful on the same task text. Select a task text above.')));
    }
    const small = Math.min(H.n_valid, B.n_valid) < 20;
    if (small) {
      this.bodyEl.append(h('div', { class: 'banner info' },
        h('div', {}, h('b', {}, 'Small sample: '), `Valid episodes: harness ${H.n_valid}, baseline ${B.n_valid}. Below 20 valid episodes the numbers count only as pipeline validation, not as a result.`)));
    }
    const armPanel = (arm, s) => {
      const cond = task.conditions.find((c) => c.arm === arm);
      const tiles = h('div', { class: 'tiles' },
        tile('Success rate', s.success_rate === null ? '—' : pct(s.success_rate),
          s.n_scored ? `${s.n_success}/${s.n_scored} · interval ${Math.round(s.ci_low * 100)}–${pct(s.ci_high)}` : 'No scored episodes yet'),
        tile('Valid episodes', `${s.n_valid}`, `Total: ${s.n}, ended: ${s.n_ended}`),
        tile('Mean actions', s.mean_actions === null ? '—' : num(s.mean_actions, 1),
          s.invalid_action_rate === null ? '' : `Invalid actions: ${pct(s.invalid_action_rate)}`),
        tile('Mean duration', dur(s.mean_duration_s)),
        tile('Median peak force', s.median_peak_force === null ? '—' : `${num(s.median_peak_force, 1)} N`),
        tile('Model reports success', s.model_claim_rate === null ? '—' : pct(s.model_claim_rate), 'Stated at done'));
      return h('section', { class: 'panel' },
        h('div', { class: 'panel-head' }, h('div', { class: 'arm-head' }, armBadge(arm, cond ? cond.label : arm)),
          h('span', { class: 'small muted' }, cond?.note || '')),
        h('div', { class: 'panel-body' }, tiles));
    };
    this.bodyEl.append(h('div', { class: 'cmp-grid' }, armPanel('harness', H), armPanel('baseline', B)));

    this.bodyEl.append(h('section', { class: 'panel' },
      h('div', { class: 'panel-head' }, h('h2', {}, 'Success rate (95% Wilson interval)'),
        h('span', { class: 'small muted' }, 'Scored valid episodes only; fault and unscored episodes are excluded')),
      h('div', { class: 'panel-body' }, ciChart([
        { label: 'Force harness', k: H.n_success, n: H.n_scored, rate: H.success_rate, lo: H.ci_low, hi: H.ci_high, color: '--harness' },
        { label: 'Baseline (no force)', k: B.n_success, n: B.n_scored, rate: B.success_rate, lo: B.ci_low, hi: B.ci_high, color: '--baseline' },
      ]), this.table([{ arm: 'Force harness', s: H }, { arm: 'Baseline (no force)', s: B }]))));

    this.bodyEl.append(this.pairPanel(d.episodes));
    this.bodyEl.append(this.episodeTable(d.episodes));
  }

  table(rows) {
    const t = h('table', { class: 'data', style: { marginTop: '12px' } },
      h('thead', {}, h('tr', {}, h('th', {}, 'Condition'), h('th', { class: 'num' }, 'Success / scored'), h('th', { class: 'num' }, 'Success rate'),
        h('th', { class: 'num' }, '95% interval'), h('th', { class: 'num' }, 'Valid episodes'), h('th', { class: 'num' }, 'Mean actions'))));
    const tb = h('tbody');
    for (const r of rows) {
      tb.append(h('tr', { style: { cursor: 'default' } }, h('td', {}, r.arm), h('td', { class: 'num' }, `${r.s.n_success} / ${r.s.n_scored}`),
        h('td', { class: 'num' }, pct(r.s.success_rate)), h('td', { class: 'num' }, r.s.n_scored ? `${pct(r.s.ci_low)} – ${pct(r.s.ci_high)}` : '—'),
        h('td', { class: 'num' }, String(r.s.n_valid)), h('td', { class: 'num' }, r.s.mean_actions === null ? '—' : num(r.s.mean_actions, 1))));
    }
    t.append(tb);
    return h('details', { class: 'fold', style: { marginTop: '8px' } }, h('summary', {}, 'View as table'), h('div', { class: 'fold-body table-wrap' }, t));
  }

  pairPanel(episodes) {
    const ended = episodes.filter((e) => e.status === 'ended');
    const harness = ended.filter((e) => e.arm === 'harness');
    const baseline = ended.filter((e) => e.arm === 'baseline');
    const panel = h('section', { class: 'panel' });
    const head = h('div', { class: 'panel-head' }, h('h2', {}, 'One pair side by side: what the model received each turn'));
    const body = h('div', { class: 'panel-body' });
    panel.append(head, body);
    if (!harness.length || !baseline.length) {
      body.append(h('div', { class: 'muted' }, `Needs at least one ended episode per condition (now harness ${harness.length}, baseline ${baseline.length}).`));
      return panel;
    }
    // default: the newest monitor pair, else the newest of each arm
    let a = harness[0], b = baseline[0];
    const paired = harness.find((e) => e.session?.pair_id && baseline.some((x) => x.session?.pair_id === e.session.pair_id));
    if (paired) { a = paired; b = baseline.find((x) => x.session?.pair_id === paired.session.pair_id); }
    const pick = (list, current) => {
      const s = h('select', {}, list.map((e) => h('option', { value: e.key }, `${when(e.started)} · ${e.name} · ${e.mode}`)));
      s.value = current.key;
      return s;
    };
    const sa = pick(harness, a), sb = pick(baseline, b);
    const grid = h('div', {});
    const draw = async () => {
      clear(grid).append(h('div', { class: 'muted' }, 'Loading…'));
      try {
        const [ea, eb] = await Promise.all([get('/api/episode', { key: sa.value }), get('/api/episode', { key: sb.value })]);
        this.drawPair(grid, ea, eb);
      } catch (err) { clear(grid).append(h('div', { class: 'muted' }, err.message)); }
    };
    sa.addEventListener('change', draw);
    sb.addEventListener('change', draw);
    body.append(h('div', { class: 'pair-grid', style: { marginBottom: '8px' } },
      h('label', { class: 'field' }, h('span', {}, 'Force harness episode'), sa),
      h('label', { class: 'field' }, h('span', {}, 'Baseline episode'), sb)), grid);
    draw();
    return panel;
  }

  drawPair(grid, ea, eb) {
    clear(grid);
    const header = (ep) => h('div', { class: 'stack', style: { gap: '4px' } },
      h('div', { class: 'row' }, armBadge(ep.arm), modeBadge(ep.mode), resultEl(ep), h('a', { href: epHref(ep.key) }, 'Open')),
      h('div', { class: 'small muted' }, `${ep.model || '—'} · actions: ${ep.metrics?.actions ?? 0} · ${dur(ep.metrics?.duration_s)}`));
    grid.append(h('div', { class: 'pair-row' }, header(ea), header(eb)));
    const n = Math.max(ea.turns.length, eb.turns.length);
    for (let i = 0; i < n; i++) {
      grid.append(h('div', { class: 'pair-row' }, compact(ea.turns[i]), compact(eb.turns[i])));
    }
  }

  episodeTable(episodes) {
    const t = h('table', { class: 'data' },
      h('thead', {}, h('tr', {}, h('th', {}, 'Started'), h('th', {}, 'Condition'), h('th', {}, 'Episode name'), h('th', {}, 'Mode'), h('th', {}, 'Model'),
        h('th', { class: 'num' }, 'Actions'), h('th', { class: 'num' }, 'Duration'), h('th', {}, 'Result'), h('th', {}, 'Status'))));
    const tb = h('tbody');
    for (const e of episodes) {
      const tr = h('tr', {}, h('td', { class: 'num nowrap' }, when(e.started)), h('td', {}, armBadge(e.arm)), h('td', { class: 'small' }, e.name),
        h('td', {}, modeBadge(e.mode)), h('td', { class: 'small' }, e.model || '—'), h('td', { class: 'num' }, String(e.metrics?.actions ?? 0)),
        h('td', { class: 'num nowrap' }, dur(e.metrics?.duration_s)), h('td', {}, resultEl(e)), h('td', {}, statusEl(e.status, e.fault)));
      tr.addEventListener('click', () => { location.hash = epHref(e.key); });
      tb.append(tr);
    }
    t.append(tb);
    return h('section', { class: 'panel' }, h('div', { class: 'panel-head' }, h('h2', {}, `All episodes for this task (${episodes.length})`)),
      episodes.length ? h('div', { class: 'table-wrap' }, t) : h('div', { class: 'empty' }, 'No episodes'));
  }
}

function compact(t) {
  if (!t) return h('div', { class: 'pair-cell muted small' }, '(this episode has no such turn)');
  const call = callOf(t);
  const a = t.action || {};
  const term = a.status === 'invalid' ? 'invalid' : (a.termination && a.termination !== 'ok' ? a.termination : null);
  return h('div', { class: 'pair-cell' },
    h('div', { class: 'turn-head' }, h('span', { class: 'turn-n' }, String(t.n).padStart(2, '0')),
      h('span', { class: 'act-name' }, call ? call.name : 'Waiting for model'), termChip(term)),
    call && call.args && Object.keys(call.args).length ? h('div', { class: 'act-args' }, fmtArgs(call.args)) : null,
    t.decision?.note ? h('div', { class: 'turn-note' }, h('span', { class: 'who' }, 'Model:'), t.decision.note) : null,
    t.translation ? receiptEl(t.translation, { hideScene: true }) : null);
}
