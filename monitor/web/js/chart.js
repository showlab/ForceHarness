// Charts. TimeChart draws on canvas so a 25 Hz live stream stays cheap; the legend, tooltip
// and table view are HTML. Conventions: 2px lines, hairline solid grid, crosshair + one tooltip
// listing every series, legend always present, table view available.
import { h, svg, clear } from './dom.js';

const cssVar = (el, name) => getComputedStyle(el).getPropertyValue(name).trim() || '#888';

function niceStep(range, target) {
  const raw = Math.max(range, 1e-9) / target;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw / mag;
  return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
}
function fmtTick(v, step) {
  const d = step >= 1 ? 0 : Math.min(3, Math.ceil(-Math.log10(step)));
  return v.toFixed(d);
}
function lowerBound(rows, x) {
  let lo = 0, hi = rows.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (rows[mid][0] < x) lo = mid + 1; else hi = mid; }
  return lo;
}

export class TimeChart {
  /** series: [{key, index, label, color: '--css-var', width}], thresholds: [{value, label, status}] */
  constructor(opts) {
    this.o = Object.assign({ height: 220, unit: 'N', window: null, thresholds: [], series: [], maxRows: 24000, title: '' }, opts);
    this.rows = [];
    this.bands = [];
    this.events = [];
    this.phases = [];
    this.hidden = new Set(this.o.hidden || []);
    this.hoverIndex = null;
    this.el = h('figure', { class: 'chart', style: { margin: '0' } });
    this.legend = h('div', { class: 'chart-legend' });
    this.canvas = h('canvas', { role: 'img', 'aria-label': this.o.title || 'Time series' });
    this.tip = h('div', { class: 'chart-tip', hidden: true });
    this.note = h('figcaption', { class: 'chart-note' });
    this.tableBox = h('div', { class: 'chart-table', hidden: true });
    this.el.append(this.legend, h('div', { style: { position: 'relative' } }, this.canvas, this.tip), this.note, this.tableBox);
    this.canvas.style.height = `${this.o.height}px`;
    this._legend();
    this.canvas.addEventListener('pointermove', (e) => this._hover(e));
    this.canvas.addEventListener('pointerleave', () => { this.hoverIndex = null; this.tip.hidden = true; this.draw(); });
    this.ro = new ResizeObserver(() => this.draw());
    this.ro.observe(this.el);
    this.themeObs = new MutationObserver(() => this.draw());
    this.themeObs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  }

  destroy() { this.ro.disconnect(); this.themeObs.disconnect(); }

  _legend() {
    clear(this.legend);
    for (const s of this.o.series) {
      const b = h('button', { type: 'button', 'aria-pressed': String(!this.hidden.has(s.key)), title: 'Click to show/hide' },
        h('span', { class: 'lk', style: { background: `var(${s.color})`, height: `${s.width || 2}px` } }), s.label);
      b.addEventListener('click', () => {
        if (this.hidden.has(s.key)) this.hidden.delete(s.key); else this.hidden.add(s.key);
        b.setAttribute('aria-pressed', String(!this.hidden.has(s.key)));
        this.draw();
        if (!this.tableBox.hidden) this._table();
      });
      this.legend.append(b);
    }
    for (const t of this.o.thresholds) {
      this.legend.append(h('span', { class: 'nowrap', title: t.label },
        h('span', { class: 'lk', style: { background: `var(--${t.status || 'warning'})`, height: '1px' } }), ` ${t.value} ${this.o.unit} ${t.label}`));
    }
    const tbl = h('button', { type: 'button', class: 'spacer', 'aria-pressed': 'false', title: 'View values as a table' }, 'Table');
    tbl.addEventListener('click', () => {
      this.tableBox.hidden = !this.tableBox.hidden;
      tbl.setAttribute('aria-pressed', String(!this.tableBox.hidden));
      if (!this.tableBox.hidden) this._table();
    });
    this.legend.append(tbl);
  }

  setRows(rows) { this.rows = rows ? rows.slice() : []; this.draw(); if (!this.tableBox.hidden) this._table(); }
  append(rows) {
    if (!rows || !rows.length) return;
    const lastX = this.rows.length ? this.rows[this.rows.length - 1][0] : -Infinity;
    for (const r of rows) if (r[0] > lastX) this.rows.push(r);
    if (this.rows.length > this.o.maxRows) this.rows.splice(0, this.rows.length - this.o.maxRows);
    this.draw();
  }
  setBands(b) { this.bands = b || []; this.draw(); }
  setEvents(e) { this.events = e || []; this.draw(); }
  setPhases(p) { this.phases = p || []; this.draw(); }
  setWindow(w) { this.o.window = w; this.draw(); }
  setNote(text) { this.note.textContent = text || ''; }

  _domain() {
    const rows = this.rows;
    if (!rows.length) return null;
    let x1 = rows[rows.length - 1][0];
    let x0 = this.o.window ? x1 - this.o.window : rows[0][0];
    if (this.o.window && this.o.windowAnchorNow) x1 = Math.max(x1, x0 + this.o.window);
    if (x1 - x0 < 1e-6) { x0 -= 0.5; x1 += 0.5; }
    const i0 = Math.max(0, lowerBound(rows, x0) - 1);
    let lo = 0, hi = 0;
    for (let i = i0; i < rows.length; i++) {
      for (const s of this.o.series) {
        if (this.hidden.has(s.key)) continue;
        const v = rows[i][s.index];
        if (v === null || v === undefined) continue;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    // Guard lines give the scale meaning; show the lowest always, higher ones when the data nears them.
    const th = this.o.thresholds.map((t) => t.value).sort((a, b) => a - b);
    const visibleTh = th.filter((v, i) => i === 0 || v <= Math.max(hi * 2, th[0]));
    for (const v of visibleTh) if (v > hi) hi = v;
    const pad = (hi - lo) * 0.08 || 1;
    return { x0, x1, i0, y0: lo < 0 ? lo - pad : 0, y1: hi + pad, th: visibleTh, hiddenTh: th.length - visibleTh.length };
  }

  draw() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = null; this._draw(); });
  }

  _draw() {
    const W = Math.max(200, this.el.clientWidth);
    const H = this.o.height;
    const dpr = window.devicePixelRatio || 1;
    const c = this.canvas;
    if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) {
      c.width = Math.round(W * dpr);
      c.height = Math.round(H * dpr);
    }
    const g = c.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const col = (n) => cssVar(this.el, n);
    const ink = col('--ink'), ink2 = col('--ink-2'), muted = col('--muted'), hair = col('--hair'), axis = col('--axis');
    const surface = col('--surface'), wash = col('--wash'), wash2 = col('--wash-2');
    g.font = '11px system-ui, -apple-system, "Segoe UI", sans-serif';
    const d = this._domain();
    const L = 40, R = 46, T = (this.bands.length || this.phases.length) ? 18 : 8, B = 20;
    const pw = W - L - R, ph = H - T - B;
    this.geom = { L, R, T, B, pw, ph, d };
    if (!d) {
      g.fillStyle = muted;
      g.fillText(this.o.empty || 'No data yet', L + 8, T + ph / 2);
      return;
    }
    const X = (x) => L + ((x - d.x0) / (d.x1 - d.x0)) * pw;
    const Y = (y) => T + ph - ((y - d.y0) / (d.y1 - d.y0)) * ph;
    this.X = X; this.Y = Y;

    // labelled motion phases (when the stream has them) and action bands: quiet washes, labels on top
    const drawSpans = (spans, labelOf) => {
      spans.forEach((b, i) => {
        if (b.x1 < d.x0 || b.x0 > d.x1) return;
        const a = Math.max(L, X(b.x0)), z = Math.min(L + pw, X(b.x1));
        g.fillStyle = i % 2 ? wash2 : wash;
        g.fillRect(a, T, Math.max(1, z - a), ph);
        // full label when it fits, else the short one (the step number), else nothing (the tooltip has it)
        const [long, short] = labelOf(b);
        const room = z - a;
        const text = long && room > g.measureText(long).width + 6 ? long
          : short && room > g.measureText(short).width + 4 ? short : null;
        if (text) {
          g.fillStyle = ink2;
          g.fillText(text, a + 3, T - 5);
        }
      });
    };
    const phaseName = (label) => (this.o.phaseText ? this.o.phaseText(label) : label);
    drawSpans(this.phases, (p) => [phaseName(p.label), '']);
    drawSpans(this.bands, (b) => [`${String(b.n).padStart(2, '0')} ${b.label || ''}`.trim(), String(b.n).padStart(2, '0')]);

    // grid + y ticks
    const yStep = niceStep(d.y1 - d.y0, Math.max(2, Math.floor(ph / 42)));
    g.lineWidth = 1;
    g.textAlign = 'right';
    g.textBaseline = 'middle';
    for (let v = Math.ceil(d.y0 / yStep) * yStep; v <= d.y1 + 1e-9; v += yStep) {
      const y = Math.round(Y(v)) + 0.5;
      g.strokeStyle = Math.abs(v) < 1e-9 ? axis : hair;
      g.beginPath(); g.moveTo(L, y); g.lineTo(L + pw, y); g.stroke();
      g.fillStyle = muted;
      g.fillText(fmtTick(v, yStep), L - 6, y);
    }
    g.textAlign = 'left';
    g.fillStyle = muted;
    g.fillText(this.o.unit, 2, T + 4);
    // x ticks
    const xStep = niceStep(d.x1 - d.x0, Math.max(2, Math.floor(pw / 90)));
    g.textBaseline = 'top';
    g.textAlign = 'center';
    for (let v = Math.ceil(d.x0 / xStep) * xStep; v <= d.x1 + 1e-9; v += xStep) {
      const x = Math.round(X(v)) + 0.5;
      g.strokeStyle = axis;
      g.beginPath(); g.moveTo(x, T + ph); g.lineTo(x, T + ph + 4); g.stroke();
      g.fillStyle = muted;
      g.fillText(`${fmtTick(v, xStep)}${this.o.xUnit ?? ' s'}`, x, T + ph + 6);
    }

    // guard lines
    g.textAlign = 'left';
    g.textBaseline = 'bottom';
    for (const t of this.o.thresholds) {
      if (!d.th.includes(t.value)) continue;
      const y = Math.round(Y(t.value)) + 0.5;
      g.strokeStyle = col(`--${t.status || 'warning'}`);
      g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(L, y); g.lineTo(L + pw, y); g.stroke();
      g.lineWidth = 1;
      g.fillStyle = ink2;
      g.fillText(`${t.value} ${this.o.unit}`, L + pw + 4, y + 5);
    }

    // event markers
    for (const e of this.events) {
      if (e.x < d.x0 || e.x > d.x1) continue;
      const x = Math.round(X(e.x)) + 0.5;
      g.strokeStyle = ink2;
      g.globalAlpha = 0.55;
      g.beginPath(); g.moveTo(x, T); g.lineTo(x, T + ph); g.stroke();
      g.globalAlpha = 1;
      g.fillStyle = ink2;
      g.beginPath(); g.moveTo(x - 4, T); g.lineTo(x + 4, T); g.lineTo(x, T + 6); g.closePath(); g.fill();
    }

    // series (|F| last so it sits on top)
    const ordered = [...this.o.series].sort((a, b) => (a.top ? 1 : 0) - (b.top ? 1 : 0));
    const rows = this.rows;
    const ends = [];
    g.save();
    g.beginPath(); g.rect(L, T - 1, pw, ph + 2); g.clip();
    for (const s of ordered) {
      if (this.hidden.has(s.key)) continue;
      g.strokeStyle = col(s.color);
      g.lineWidth = s.width || 2;
      g.lineJoin = 'round';
      g.lineCap = 'round';
      g.beginPath();
      let pen = false, last = null;
      for (let i = d.i0; i < rows.length; i++) {
        const v = rows[i][s.index];
        if (v === null || v === undefined) { pen = false; continue; }
        const x = X(rows[i][0]), y = Y(v);
        if (!pen) { g.moveTo(x, y); pen = true; } else g.lineTo(x, y);
        last = [x, y, v];
      }
      g.stroke();
      if (last) ends.push({ s, y: last[1], v: last[2] });
    }
    g.restore();
    // direct end labels only when they do not collide (else the legend carries identity)
    ends.sort((a, b) => a.y - b.y);
    const guardYs = this.o.thresholds.filter((t) => d.th.includes(t.value)).map((t) => Y(t.value) + 5);
    const clash = ends.some((e, i) => (i && e.y - ends[i - 1].y < 13) || guardYs.some((y) => Math.abs(e.y - y) < 13));
    if (!clash && ends.length <= 4) {
      g.textBaseline = 'middle';
      g.textAlign = 'left';
      for (const e of ends) {
        g.strokeStyle = col(e.s.color);
        g.lineWidth = 2;
        g.beginPath(); g.moveTo(L + pw + 3, e.y); g.lineTo(L + pw + 9, e.y); g.stroke();
        g.fillStyle = ink2;
        g.fillText(e.s.label, L + pw + 12, e.y);
      }
    }
    // crosshair
    if (this.hoverIndex !== null && this.hoverIndex !== undefined && rows[this.hoverIndex]) {
      const r = rows[this.hoverIndex];
      const x = Math.round(X(r[0])) + 0.5;
      g.strokeStyle = ink2;
      g.lineWidth = 1;
      g.beginPath(); g.moveTo(x, T); g.lineTo(x, T + ph); g.stroke();
      for (const s of this.o.series) {
        if (this.hidden.has(s.key) || r[s.index] === null || r[s.index] === undefined) continue;
        const y = Y(r[s.index]);
        g.fillStyle = surface;
        g.beginPath(); g.arc(x, y, 6, 0, Math.PI * 2); g.fill();
        g.fillStyle = col(s.color);
        g.beginPath(); g.arc(x, y, 4, 0, Math.PI * 2); g.fill();
      }
    }
    if (d.hiddenTh) {
      const above = this.o.thresholds.filter((t) => !d.th.includes(t.value)).map((t) => `${t.value} ${this.o.unit}`);
      g.fillStyle = muted;
      g.textAlign = 'right';
      g.textBaseline = 'top';
      g.fillText(`Higher guard lines off chart: ${above.join(', ')}`, L + pw - 4, T + 3);
    }
    g.fillStyle = ink;
  }

  _hover(e) {
    const gm = this.geom;
    if (!gm || !gm.d || !this.rows.length) return;
    const rect = this.canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const x = gm.d.x0 + ((px - gm.L) / gm.pw) * (gm.d.x1 - gm.d.x0);
    let i = lowerBound(this.rows, x);
    if (i >= this.rows.length) i = this.rows.length - 1;
    if (i > 0 && Math.abs(this.rows[i - 1][0] - x) < Math.abs(this.rows[i][0] - x)) i -= 1;
    if (this.rows[i][0] < gm.d.x0 - 1e-9) { this.tip.hidden = true; return; }
    this.hoverIndex = i;
    const r = this.rows[i];
    clear(this.tip);
    const band = this.bands.find((b) => r[0] >= b.x0 && r[0] <= b.x1);
    const phase = this.phases.find((p) => r[0] >= p.x0 && r[0] <= p.x1);
    const head = [`t = ${r[0].toFixed(2)} s`];
    if (band) head.push(`Action ${String(band.n).padStart(2, '0')} ${band.label || ''}`);
    if (phase) head.push(this.o.phaseText ? this.o.phaseText(phase.label) : phase.label);
    this.tip.append(h('div', { class: 'tt' }, head.join(' · ')));
    for (const s of this.o.series) {
      if (this.hidden.has(s.key)) continue;
      const v = r[s.index];
      this.tip.append(h('div', { class: 'tr' },
        h('span', { class: 'lk', style: { background: `var(${s.color})` } }),
        h('b', {}, v === null || v === undefined ? '—' : `${v.toFixed(this.o.digits ?? 2)} ${this.o.unit}`),
        h('span', {}, s.label)));
    }
    const ev = this.events.find((item) => Math.abs(this.X(item.x) - px) < 5);
    if (ev) this.tip.append(h('div', { class: 'tt', style: { marginTop: '4px' } }, `Event: ${ev.kind}${ev.force ? ` (|F| ${ev.force.toFixed(1)} N)` : ''}`));
    this.tip.hidden = false;
    const tw = this.tip.offsetWidth;
    const left = px + 14 + tw > rect.width ? px - tw - 14 : px + 14;
    this.tip.style.left = `${Math.max(0, left)}px`;
    this.tip.style.top = `${gm.T + 4}px`;
    this.draw();
  }

  _table() {
    clear(this.tableBox);
    const cols = this.o.series.filter((s) => !this.hidden.has(s.key));
    const rows = this.rows;
    const step = Math.max(1, Math.ceil(rows.length / 400));
    const table = h('table', { class: 'data' },
      h('thead', {}, h('tr', {}, h('th', { class: 'num' }, 't (s)'), cols.map((s) => h('th', { class: 'num' }, `${s.label} (${this.o.unit})`)))));
    const tb = h('tbody');
    for (let i = 0; i < rows.length; i += step) {
      tb.append(h('tr', {}, h('td', { class: 'num' }, rows[i][0].toFixed(2)),
        cols.map((s) => h('td', { class: 'num' }, rows[i][s.index] === null || rows[i][s.index] === undefined ? '—' : rows[i][s.index].toFixed(3)))));
    }
    table.append(tb);
    this.tableBox.append(table);
    if (step > 1) this.tableBox.append(h('div', { class: 'chart-note', style: { padding: '6px 10px' } }, `One of every ${step} points, ${rows.length} total`));
  }
}

export const FORCE_SERIES = [
  { key: 'fx', index: 1, label: 'Fx', color: '--axis-x' },
  { key: 'fy', index: 2, label: 'Fy', color: '--axis-y' },
  { key: 'fz', index: 3, label: 'Fz', color: '--axis-z' },
  { key: 'fn', index: 4, label: '|F|', color: '--ink', top: true },
];
export const TORQUE_SERIES = [{ key: 'tn', index: 5, label: '|T|', color: '--ink', top: true }];

export function sparkline(values, { width = 88, height = 22 } = {}) {
  const box = svg('svg', { class: 'spark', viewBox: `0 0 ${width} ${height}`, 'aria-hidden': 'true' });
  if (!values || values.length < 2) return box;
  const max = Math.max(...values, 1e-6);
  const pts = values.map((v, i) => [(i / (values.length - 1)) * (width - 4) + 2, height - 3 - (v / max) * (height - 6)]);
  box.append(svg('path', { d: 'M' + pts.map((p) => p.map((n) => n.toFixed(1)).join(',')).join('L'), fill: 'none',
    stroke: 'var(--ink-2)', 'stroke-width': 1.5, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
  const peak = values.indexOf(Math.max(...values));
  box.append(svg('circle', { cx: pts[peak][0], cy: pts[peak][1], r: 2.5, fill: 'var(--ink)' }));
  return box;
}

// Success rate with its 95% Wilson interval, one row per arm: dot + whisker on a 0-100% axis.
export function ciChart(rows) {
  const W = 680, rowH = 40, top = 22, left = 132, right = 24, bottom = 26;
  const H = top + rows.length * rowH + bottom;
  const X = (p) => left + p * (W - left - right);
  const s = svg('svg', { class: 'ci-chart', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Success rate with 95% Wilson interval' });
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    s.append(svg('line', { x1: X(t), x2: X(t), y1: top - 4, y2: H - bottom, stroke: 'var(--hair)', 'stroke-width': 1 }));
    s.append(svg('text', { x: X(t), y: H - 8, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--muted)' }, `${t * 100}%`));
  }
  rows.forEach((r, i) => {
    const y = top + i * rowH + rowH / 2;
    s.append(svg('text', { x: 0, y: y + 4, 'font-size': 12, fill: 'var(--ink)' }, r.label));
    if (r.rate === null || r.rate === undefined) {
      s.append(svg('text', { x: X(0) + 4, y: y + 4, 'font-size': 12, fill: 'var(--muted)' }, 'No scored valid episodes yet'));
      return;
    }
    s.append(svg('line', { x1: X(r.lo), x2: X(r.hi), y1: y, y2: y, stroke: `var(${r.color})`, 'stroke-width': 2, 'stroke-linecap': 'round' }));
    for (const end of [r.lo, r.hi]) s.append(svg('line', { x1: X(end), x2: X(end), y1: y - 5, y2: y + 5, stroke: `var(${r.color})`, 'stroke-width': 2 }));
    s.append(svg('circle', { cx: X(r.rate), cy: y, r: 7, fill: 'var(--surface)' }));
    s.append(svg('circle', { cx: X(r.rate), cy: y, r: 5, fill: `var(${r.color})` }));
    const label = `${r.k}/${r.n} = ${Math.round(r.rate * 100)}%`;
    const tx = X(r.rate) > W - 150 ? X(r.lo) - 8 : X(r.hi) + 8;
    s.append(svg('text', { x: tx, y: y - 8, 'font-size': 12, fill: 'var(--ink-2)', 'text-anchor': X(r.rate) > W - 150 ? 'end' : 'start' }, label));
  });
  return s;
}
