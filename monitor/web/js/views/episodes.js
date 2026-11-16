// Every episode the monitor can find under runs/: search, filter, open.
import { get } from '../api.js';
import { armBadge, clear, dur, epHref, h, modeBadge, num, resultEl, statusEl, when } from '../dom.js';
import { labelCell } from '../label.js';

const FILTERS = { q: '', task: 'all', arm: 'all', mode: 'all', result: 'all' };

export class EpisodesView {
  constructor(root, app) {
    this.root = root;
    this.app = app;
    this.f = Object.assign({}, FILTERS, readFilters());
    this.list = [];
    this.countEl = h('span', { class: 'small muted' });
    this.tableEl = h('div', { class: 'table-wrap' });
    const search = h('input', { type: 'search', placeholder: 'Search: episode name, model, note…', value: this.f.q, 'aria-label': 'Search' });
    search.addEventListener('input', () => { this.f.q = search.value; this.render(); });
    const select = (key, options, label) => {
      const s = h('select', { 'aria-label': label }, options.map(([v, t]) => h('option', { value: v }, t)));
      s.value = this.f[key];
      s.addEventListener('change', () => { this.f[key] = s.value; this.render(); });
      return s;
    };
    const tasks = [['all', 'All tasks'], ...app.info.tasks.map((t) => [t.id, t.name])];
    root.append(
      h('div', { class: 'section-title' }, h('h1', {}, 'Records'), this.countEl),
      h('div', { class: 'filters' }, search,
        select('task', tasks, 'Task'),
        select('arm', [['all', 'All conditions'], ['harness', 'Force harness'], ['baseline', 'Baseline (no force)']], 'Condition'),
        select('mode', [['all', 'Real robot and demo'], ['live', 'Real robot only'], ['demo', 'Demo only']], 'Mode'),
        select('result', [['all', 'All results'], ['success', 'Success'], ['fail', 'Failure'], ['unscored', 'Unscored'], ['fault', 'Fault'], ['running', 'Running']], 'Result')),
      h('section', { class: 'panel' }, this.tableEl));
    this.load();
    this.timer = setInterval(() => this.load(), 5000);
  }

  destroy() { clearInterval(this.timer); }

  async load() {
    try {
      const r = await get('/api/episodes');
      this.list = r.episodes;
      this.render();
    } catch (err) {
      clear(this.tableEl).append(h('div', { class: 'empty' }, `Failed to read: ${err.message}`));
    }
  }

  match(e) {
    const f = this.f;
    if (f.task !== 'all' && e.family !== f.task) return false;
    if (f.arm !== 'all' && e.arm !== f.arm) return false;
    if (f.mode !== 'all' && e.mode !== f.mode) return false;
    const r = e.result || {};
    if (f.result === 'success' && !(r.scored && r.success)) return false;
    if (f.result === 'fail' && !(r.scored && r.success === false)) return false;
    if (f.result === 'unscored' && (r.scored || e.status !== 'ended' || e.fault)) return false;
    if (f.result === 'fault' && !e.fault) return false;
    if (f.result === 'running' && ['ended', 'stale'].includes(e.status)) return false;
    if (f.q) {
      const hay = [e.name, e.key, e.model, e.condition, e.note, e.result?.label?.note, e.fault, e.session?.instruction].join(' ').toLowerCase();
      if (!f.q.toLowerCase().split(/\s+/).every((w) => hay.includes(w))) return false;
    }
    return true;
  }

  render() {
    writeFilters(this.f);
    const rows = this.list.filter((e) => this.match(e));
    this.countEl.textContent = `Episodes: ${this.list.length}, shown: ${rows.length}`;
    const scroll = this.tableEl.scrollTop;
    clear(this.tableEl);
    if (!rows.length) {
      this.tableEl.append(h('div', { class: 'empty' }, this.list.length ? 'No matching episodes' : 'No episodes found yet (no episode.json under the runs folder)'));
      return;
    }
    const table = h('table', { class: 'data' },
      h('thead', {}, h('tr', {},
        h('th', {}, 'Started'), h('th', {}, 'Task · episode name'), h('th', {}, 'Condition'), h('th', {}, 'Mode'), h('th', {}, 'Model'),
        h('th', { class: 'num' }, 'Actions'), h('th', { class: 'num' }, 'Duration'), h('th', { class: 'num' }, 'Peak force'),
        h('th', {}, 'Result'), h('th', {}, 'Manual label · note'), h('th', {}, 'Status'))));
    const tb = h('tbody');
    for (const e of rows) {
      const task = this.app.task(e.family);
      const tr = h('tr', { tabindex: '0' },
        h('td', { class: 'num nowrap' }, when(e.started)),
        h('td', {}, h('div', {}, task?.name || e.family), h('div', { class: 'small muted' }, e.name)),
        h('td', {}, armBadge(e.arm)),
        h('td', {}, modeBadge(e.mode)),
        h('td', { class: 'small' }, e.model || '—'),
        h('td', { class: 'num' }, `${e.metrics?.actions ?? 0}${e.metrics?.invalid ? ` (invalid ${e.metrics.invalid})` : ''}`),
        h('td', { class: 'num nowrap' }, dur(e.metrics?.duration_s)),
        h('td', { class: 'num nowrap' }, e.peak_force === null || e.peak_force === undefined ? '—' : `${num(e.peak_force, 1)} N`),
        h('td', {}, resultEl(e)),
        h('td', {}, labelCell(e, () => this.load())),
        h('td', {}, statusEl(e.status, e.fault)));
      const open = () => { location.hash = epHref(e.key); };
      tr.addEventListener('click', open);
      tr.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') open(); });
      tb.append(tr);
    }
    table.append(tb);
    this.tableEl.append(table);
    this.tableEl.scrollTop = scroll;
  }
}

function readFilters() {
  try { return JSON.parse(sessionStorage.getItem('fhmon-filters') || '{}'); } catch { return {}; }
}
function writeFilters(f) {
  try { sessionStorage.setItem('fhmon-filters', JSON.stringify(f)); } catch { /* ignore */ }
}
