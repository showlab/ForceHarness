// DOM building and formatting. Everything that came from files or a model goes in as text nodes.

export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'text') el.textContent = v;
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (k === 'style') Object.assign(el.style, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (k === 'value') el.value = v;
    else if (k === 'checked') el.checked = !!v;
    else el.setAttribute(k, v === true ? '' : String(v));
  }
  append(el, children);
  return el;
}

export function append(el, children) {
  for (const c of [children].flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export function svg(tag, attrs, ...children) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v !== null && v !== undefined) el.setAttribute(k, String(v));
  for (const c of children.flat()) if (c) el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
}

export function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }

// ---------- numbers & time ----------
export function num(v, digits = 2) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '—';
  return Number(v).toFixed(digits);
}
export function vec(v, digits = 3) {
  if (!Array.isArray(v)) return String(v);
  return '[' + v.map((x) => (typeof x === 'number' ? x.toFixed(digits) : String(x))).join(', ') + ']';
}
export function dur(s) {
  if (s === null || s === undefined || Number.isNaN(Number(s))) return '—';
  s = Number(s);
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)} s`;
  const m = Math.floor(s / 60);
  const r = Math.round(s - m * 60);
  if (m < 60) return `${m} min ${r} s`;
  return `${Math.floor(m / 60)} h ${m % 60} min`;
}
export function when(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const p = (n) => String(n).padStart(2, '0');
  const clock = `${p(d.getHours())}:${p(d.getMinutes())}`;
  // ?blind in the page address hides calendar dates (for screenshots that must not show when episodes ran).
  return new URLSearchParams(location.search).has('blind') ? clock : `${p(d.getMonth() + 1)}-${p(d.getDate())} ${clock}`;
}
export function pct(v) { return v === null || v === undefined ? '—' : `${Math.round(v * 100)}%`; }
export function tokens(n) {
  if (n === null || n === undefined) return '—';
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);
}

// compact call text: turn(axis=[...], angle_deg=20)
export function args(a) {
  if (!a || typeof a !== 'object') return '';
  return Object.entries(a).map(([k, v]) => {
    let s;
    if (Array.isArray(v)) s = vec(v, Math.abs(v[0]) >= 10 ? 1 : 3);
    else if (typeof v === 'number') s = Number.isInteger(v) ? String(v) : v.toFixed(4).replace(/0+$/, '').replace(/\.$/, '');
    else if (typeof v === 'string') s = v.length > 60 ? JSON.stringify(v.slice(0, 57) + '…') : JSON.stringify(v);
    else s = JSON.stringify(v);
    return `${k}=${s}`;
  }).join('  ');
}

// ---------- vocabulary ----------
export const STATUS_TEXT = {
  starting: 'Starting', waiting_model: 'Waiting for model', executing: 'Robot executing', running: 'Running',
  ended: 'Ended', stale: 'Stale (process may have exited)', queued: 'Queued', failed: 'Failed to start',
  stopped: 'Interrupted', cancelled: 'Cancelled',
};
export const TERM_TEXT = {
  completed: 'Completed', contact: 'Stopped on contact', resistance_drop: 'Resistance drop', stalled_short: 'Stalled short',
  path_deviation: 'Path deviation', stop_drift: 'Drift after stop', force_cap: 'Force cap hit', torque_cap: 'Torque cap hit',
  no_contact_within_max_dist: 'No contact', budget: 'Time budget used', unexpected_motion: 'Unexpected motion',
  stall: 'Stalled', timeout: 'Timed out', target_reached: 'Target reached', error: 'Error', ok: 'Completed', invalid: 'Invalid',
  stopped: 'Stopped',
};
const TERM_BAD = new Set(['force_cap', 'torque_cap', 'unexpected_motion', 'error', 'invalid', 'stopped']);
const TERM_WARN = new Set(['contact', 'stalled_short', 'path_deviation', 'stop_drift', 'stall', 'timeout', 'budget']);

export function termChip(term) {
  if (!term) return null;
  const cls = TERM_BAD.has(term) ? 'term bad' : TERM_WARN.has(term) ? 'term warn' : 'term';
  return h('span', { class: cls, title: term }, TERM_TEXT[term] || term);
}

export function armBadge(arm, label) {
  const cls = arm === 'baseline' ? 'baseline' : 'harness';
  return h('span', { class: `badge ${cls}` }, h('span', { class: 'key' }), label || (arm === 'baseline' ? 'Baseline (no force)' : 'Force harness'));
}

export function modeBadge(mode) {
  return mode === 'live'
    ? h('span', { class: 'badge live', title: 'Real robot' }, 'Real robot')
    : h('span', { class: 'badge demo', title: 'Demo: no robot connected' }, 'Demo (no robot)');
}

export function statusEl(status, fault) {
  const s = fault && status === 'ended' ? 'fault' : status;
  const text = fault && status === 'ended' ? 'Ended (fault)' : (STATUS_TEXT[status] || status || '—');
  return h('span', { class: 'status', dataset: { s } }, h('span', { class: 'dot' }), text);
}

const ICONS = {
  ok: 'M5 12.5l4.5 4.5L19 7.5',
  bad: 'M7 7l10 10M17 7L7 17',
  warn: 'M12 4l9 16H3zM12 10v4M12 17v.5',
  info: 'M12 8v.5M12 11v6',
};
export function icon(name, cls = 'glyph') {
  const s = svg('svg', { viewBox: '0 0 24 24', class: cls, 'aria-hidden': 'true' });
  if (name === 'info') s.append(svg('circle', { cx: 12, cy: 12, r: 9, fill: 'none', stroke: 'currentColor', 'stroke-width': 2 }));
  s.append(svg('path', { d: ICONS[name], fill: 'none', stroke: 'currentColor', 'stroke-width': 2.2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }));
  return s;
}

// Episode verdict for tables and headers (compact: no model self-report, it goes in a subtitle)
export function resultEl(ep, { compact = false } = {}) {
  const r = ep.result || {};
  if (r.manual) {
    const ok = !!r.success;
    const hr = r.harness || {};
    const was = ep.fault ? 'Fault' : hr.scored ? (OUTCOME_TEXT[hr.outcome] || hr.outcome || (hr.success ? 'Success' : 'Failure')) : 'Unscored';
    return h('span', { class: `result ${ok ? 'ok' : 'bad'}`, title: `Manual label (harness record: ${was})` },
      icon(ok ? 'ok' : 'bad'), ok ? 'Success' : 'Failure', h('span', { class: 'manual-tag' }, 'manual'));
  }
  if (ep.fault) return h('span', { class: 'result bad', title: ep.fault }, icon('bad'), 'Fault');
  if (r.scored) {
    const ok = !!r.success;
    const text = OUTCOME_TEXT[r.outcome] || r.outcome || (ok ? 'Success' : 'Failure');
    return h('span', { class: `result ${ok ? 'ok' : 'bad'}` }, icon(ok ? 'ok' : 'bad'), text);
  }
  if (ep.status !== 'ended') return h('span', { class: 'result none' }, 'Running');
  if (compact) return h('span', { class: 'result none' }, 'Unscored');
  const claim = r.model_claim === true ? ', model reports success' : r.model_claim === false ? ', model reports failure' : '';
  return h('span', { class: 'result none' }, `Unscored${claim}`);
}
export const OUTCOME_TEXT = { success: 'Completed', partial: 'Partial', failed: 'Not completed', not_attempted: 'Not attempted' };

// ---------- misc ----------
let toastTimer = null;
export function toast(message, kind = 'info') {
  const el = document.getElementById('toast');
  el.textContent = message;
  el.className = `toast ${kind === 'error' ? 'error' : ''}`;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, kind === 'error' ? 7000 : 3500);
}

export function banner(kind, children) {
  const glyph = kind === 'critical' ? 'bad' : kind === 'warning' ? 'warn' : 'info';
  return h('div', { class: `banner ${kind}`, role: kind === 'critical' ? 'alert' : null }, icon(glyph), h('div', { class: 'grow' }, children));
}

export function tile(label, value, sub, title) {
  return h('div', { class: 'tile', title: title || null },
    h('div', { class: 'label' }, label),
    h('div', { class: 'value' }, value),
    sub ? h('div', { class: 'sub' }, sub) : null);
}

export function seg(options, value, onChange) {
  const box = h('div', { class: 'seg', role: 'group' });
  for (const o of options) {
    const b = h('button', { type: 'button', 'aria-pressed': String(o.value === value), disabled: o.disabled || null, title: o.title || null }, o.label);
    b.addEventListener('click', () => {
      if (o.disabled) return;
      for (const x of box.children) x.setAttribute('aria-pressed', 'false');
      b.setAttribute('aria-pressed', 'true');
      onChange(o.value);
    });
    box.append(b);
  }
  return box;
}

export function keyOf(ep) { return ep.key; }
export function epHref(key) { return `#/episode/${encodeURIComponent(key)}`; }
