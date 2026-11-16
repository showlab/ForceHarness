// One turn = what the model saw (obs) -> what it decided (cmd) -> what the robot did -> the receipt
// it read back. The receipt's force lines are the harness's translation; baseline turns show the gap.
import { args as fmtArgs, dur, h, num, TERM_TEXT, termChip, tokens } from './dom.js';
import { sparkline } from './chart.js';

export function receiptEl(translation, { hideScene = false } = {}) {
  const box = h('div', { class: 'receipt' });
  for (const r of translation?.rows || []) {
    if (hideScene && r.kind === 'scene') continue;
    box.append(h('div', { class: `ln ${r.kind}`, title: r.kind === 'force' ? 'Force translation by the harness' : null }, r.text));
  }
  return box;
}

export function callOf(turn) {
  const a = turn.action;
  if (a && a.name) return { name: a.name, args: a.args };
  const c = turn.decision?.calls?.[0];
  return c ? { name: c.name, args: c.args } : null;
}

const f3 = (v) => (typeof v === 'number' ? v.toFixed(3) : String(v));
const vec3 = (v) => (Array.isArray(v) ? `[${v.map((x) => (typeof x === 'number' ? x.toFixed(2) : x)).join(', ')}]` : '');

/** The few arguments that say what a primitive did, e.g. turn → "20°, stop on resistance drop". */
export function brief(name, a = {}) {
  a = a || {};
  switch (name) {
    case 'move_to': return 'x' in a ? `→ (${f3(a.x)}, ${f3(a.y)}, ${f3(a.z)})` : '';
    case 'move_by': return 'dx' in a ? `Δ(${f3(a.dx)}, ${f3(a.dy)}, ${f3(a.dz)})` : '';
    case 'move_until_contact': return `along ${vec3(a.direction)}${a.max_dist !== undefined ? ` ≤${a.max_dist} m` : ''}${a.f_threshold !== undefined ? ` stop at ${a.f_threshold} N` : ''}`;
    case 'turn': return `${a.angle_deg ?? '?'}°${a.stop_on_resistance_drop ? ', stop on resistance drop' : ''}`;
    case 'rotate_z': return `${a.angle_deg ?? a.deg ?? '?'}°`;
    case 'hold': return a.seconds !== undefined ? `${a.seconds} s` : '';
    case 'lift': return a.height !== undefined ? `↑${a.height} m` : '';
    case 'push': case 'slide': case 'probe':
      return [vec3(a.direction), a.force !== undefined ? `${a.force} N` : '', a.distance !== undefined ? `${a.distance} m` : ''].filter(Boolean).join(' ');
    case 'done': return a.success === true ? 'reports success' : a.success === false ? 'reports failure' : '';
    case 'grasp': case 'release': return '';
    default: {
      const pairs = Object.entries(a).slice(0, 2).map(([k, v]) => `${k}=${Array.isArray(v) ? vec3(v) : typeof v === 'number' ? +v.toFixed(3) : v}`);
      return pairs.join(' ');
    }
  }
}

function outcome(turn) {
  const a = turn.action || {};
  if (!callOf(turn)) return { cls: 'pending', text: 'Waiting for model…' };
  if (a.status === 'executing') return { cls: 'executing', text: 'Executing…' };
  if (a.status === 'invalid') return { cls: 'invalid', text: 'Invalid, not run' };
  if (a.status === 'declined') return { cls: 'invalid', text: 'Not approved' };
  if (a.status === 'interrupted') return { cls: 'invalid', text: 'No result' };
  if (a.status === 'done') return { cls: 'done', text: 'Ended episode' };
  const term = a.termination && a.termination !== 'ok' ? a.termination : 'completed';
  const dur0 = turn.report?.duration_s;
  const text = term === 'completed' ? '✓' : (TERM_TEXT[term] || term);
  return { cls: term === 'completed' ? 'ok' : ['force_cap', 'torque_cap', 'unexpected_motion', 'error', 'stopped'].includes(term) ? 'invalid' : 'warn',
    text: dur0 ? `${text} ${dur(dur0)}` : text };
}

/** Every call of the episode in order: number, primitive, key arguments, result. */
export function callStrip(turns, { selected, onSelect }) {
  const strip = h('div', { class: 'calls', role: 'list' });
  turns.forEach((t, i) => {
    const call = callOf(t);
    const o = outcome(t);
    const chip = h('button', {
      type: 'button', role: 'listitem', class: `chip-call ${o.cls}${t.n === selected ? ' selected' : ''}`,
      title: call ? `Step ${t.n}: ${call.name}(${fmtArgs(call.args)})` : `Step ${t.n}: waiting for model`,
    },
    h('span', { class: 'call-top' },
      h('span', { class: 'call-n' }, String(t.n).padStart(2, '0')),
      h('span', { class: 'call-name' }, call ? call.name : '…'),
      call && brief(call.name, call.args) ? h('span', { class: 'call-arg' }, brief(call.name, call.args)) : null,
      h('span', { class: 'call-res' }, o.text)));
    chip.addEventListener('click', () => onSelect(t.n));
    if (i) strip.append(h('span', { class: 'call-sep', 'aria-hidden': 'true' }, '→'));
    strip.append(chip);
  });
  return strip;
}

function thumbs(images, big = false) {
  if (!images || !images.length) return null;
  const caption = (img) => (img.label === 'wrist' ? 'Wrist (finger pads marked)' : img.label === 'exterior' ? 'Exterior' : img.name);
  return h('div', { class: `thumbs${big ? ' big' : ''}` }, images.map((img) =>
    h('a', { href: img.url, target: '_blank', rel: 'noopener', title: `${img.name} (click to enlarge)` },
      h('img', { src: img.url, alt: caption(img), loading: 'lazy' }),
      h('span', { class: 'thumb-cap' }, caption(img)))));
}

function stage(kind, label, hint, ...children) {
  return h('section', { class: `stage ${kind}` },
    h('div', { class: 'stage-label' }, h('span', { class: 'stage-key', 'aria-hidden': 'true' }), h('b', {}, label),
      hint ? h('span', { class: 'stage-hint' }, hint) : null),
    ...children);
}

function argsTable(args) {
  const entries = Object.entries(args || {});
  if (!entries.length) return h('div', { class: 'stage-note' }, '(no arguments)');
  const dl = h('dl', { class: 'kv args' });
  for (const [k, v] of entries) {
    let text;
    if (Array.isArray(v)) text = '[' + v.map((x) => (typeof x === 'number' ? +x.toFixed(4) : x)).join(', ') + ']';
    else if (typeof v === 'number') text = String(+v.toFixed(4));
    else if (typeof v === 'string') text = v;
    else text = JSON.stringify(v);
    dl.append(h('dt', {}, k), h('dd', {}, text));
  }
  return dl;
}

// Free task: the objects the wrist depth camera found for this turn (the yellow boxes in the wrist image).
export function sceneEl(scene) {
  const objs = scene.objects || [];
  const head = scene.synthetic ? 'Object list (demo: synthetic scene)' : 'Detected objects (wrist depth, about ±1–2 cm)';
  if (scene.error && !objs.length) return h('div', { class: 'stage-note' }, h('b', {}, `${head}: `), scene.error);
  if (!objs.length) return h('div', { class: 'stage-note' }, h('b', {}, `${head}: `), 'No objects detected on the table in this image.');
  const f = (v, d = 3) => (typeof v === 'number' ? v.toFixed(d) : '—');
  const rows = objs.map((o) => h('tr', {},
    h('td', { class: 'num' }, `#${o.id}`),
    h('td', { class: 'num' }, `${f(o.centre?.[0])}, ${f(o.centre?.[1])}`),
    h('td', { class: 'num' }, f(o.height)),
    h('td', { class: 'num' }, `${f(o.length)} × ${f(o.width)}`),
    h('td', { class: 'num' }, typeof o.axis_deg === 'number' ? `${o.axis_deg >= 0 ? '+' : ''}${o.axis_deg.toFixed(0)}°` : '—'),
    h('td', {}, [o.label, o.under_hand ? 'under or in hand' : null, o.partial ? 'partly visible' : null].filter(Boolean).join(', '))));
  return h('div', { class: 'scene-objects' },
    h('div', { class: 'small ink2' }, h('b', {}, head), typeof scene.table_z === 'number' ? ` · table z ${scene.table_z.toFixed(3)} m` : ''),
    h('div', { class: 'table-wrap' }, h('table', { class: 'data compact' },
      h('thead', {}, h('tr', {}, h('th', { class: 'num' }, 'ID'), h('th', { class: 'num' }, 'Center x, y (m)'),
        h('th', { class: 'num' }, 'Height (m)'), h('th', { class: 'num' }, 'L × W (m)'), h('th', { class: 'num' }, 'Long axis'), h('th', {}, 'Note'))),
      h('tbody', {}, rows))));
}

/**
 * One turn as the loop it is: ① what the model saw (text + images) → ② what it decided and why →
 * ③ the primitive it called, with every parameter → ④ what came back (the receipt it reads next turn).
 * ctx: {selected, live: {text, thinking, active, error}, manual: {example, onSubmit}, onSelect,
 *       waitingText, noImages, instruction}
 */
export function turnCard(turn, ctx = {}) {
  const call = callOf(turn);
  const a = turn.action || {};
  const m = turn.model;
  const live = ctx.live;
  const pending = !call && !!turn.obs;
  const o = outcome(turn);
  const card = h('article', {
    class: `turn loop${ctx.selected ? ' selected' : ''}${pending ? ' pending' : ''}`,
    dataset: { n: String(turn.n) }, tabindex: '0', 'aria-label': `Turn ${turn.n}`,
  });
  card.addEventListener('click', (e) => {
    if (e.target.closest('a, button, summary, textarea, input, details[open] .fold-body')) return;
    ctx.onSelect && ctx.onSelect(turn.n);
  });
  card.addEventListener('keydown', (e) => { if (e.key === 'Enter' && e.target === card) ctx.onSelect && ctx.onSelect(turn.n); });

  const meta = [];
  if (m && m.latency_s !== undefined && m.latency_s !== null) meta.push(`thinking ${dur(m.latency_s)}`);
  if (m && (m.usage_in || m.usage_out)) meta.push(`tokens ${tokens(m.usage_in)} / ${tokens(m.usage_out)}`);
  if (m && m.cost_usd) meta.push(`$${m.cost_usd.toFixed(4)}`);
  card.append(h('div', { class: 'loop-head' },
    h('span', { class: 'turn-n' }, String(turn.n).padStart(2, '0')),
    h('b', {}, `Turn ${turn.n}`),
    call ? h('span', { class: 'act-name' }, call.name) : null,
    h('span', { class: `loop-res ${o.cls}` }, o.text),
    h('span', { class: 'grow' }),
    meta.length ? h('span', { class: 'small muted num' }, meta.join(' · ')) : null));

  // ① input: everything the model got this turn
  const imgs = turn.obs?.images || [];
  const input = [];
  if (imgs.length) {
    input.push(thumbs(imgs, true));
    const sent = m?.images_sent;
    if (Array.isArray(sent) && sent.length < imgs.length) {
      input.push(h('div', { class: 'small muted' }, `Images attached to the API request: ${sent.length ? sent.join(', ') : 'none'} (earlier turns send file names only)`));
    }
  } else if (turn.obs) {
    input.push(h('div', { class: 'stage-note' }, ctx.noImages || 'No images this turn: the model received text only.'));
  }
  if (turn.n === 0 && ctx.instruction) input.push(h('div', { class: 'stage-note' }, h('b', {}, 'Operator extra instruction: '), ctx.instruction));
  if (turn.scene) input.push(sceneEl(turn.scene));
  if (turn.obs?.text) {
    input.push(h('details', { class: 'fold' },
      h('summary', {}, turn.n === 0 ? `Task description and initial state (${turn.obs.name}, ${turn.obs.text.length} chars)`
        : `Previous tool result and new state (${turn.obs.name})`),
      h('div', { class: 'fold-body' }, h('pre', { class: 'obs' }, turn.obs.text))));
  }
  if (!turn.obs) input.push(h('div', { class: 'stage-note' }, 'Observation file for this turn not found.'));
  card.append(stage('input', '① Input', 'What the model saw this turn', ...input));

  // ② decision: the model's reasoning (or the live stream / the manual editor while it is pending)
  const decision = [];
  if (pending) {
    if (live && (live.active || live.text || live.thinking || live.error)) {
      if (live.thinking) decision.push(h('div', { class: 'stream-box thinking' }, live.thinking));
      decision.push(h('div', { class: `stream-box${live.active ? ' caret' : ''}` }, live.text || ''));
      if (live.error) decision.push(h('div', { class: 'small', style: { color: 'var(--critical)' } }, live.error));
    } else if (ctx.manual) {
      decision.push(manualEditor(turn.n, ctx.manual));
    } else {
      decision.push(h('div', { class: 'stage-note' }, ctx.waitingText || `Waiting for the model to reply with cmd_${turn.n}.json…`));
    }
  } else {
    const note = turn.decision?.note || (call?.name === 'done' ? call.args?.note : null);
    decision.push(note ? h('div', { class: 'decision-note' }, note)
      : h('div', { class: 'stage-note' }, 'The model returned only an action, with no reasoning.'));
    if (m?.operator) decision.push(h('div', { class: 'small muted' }, 'The operator ended this episode from the monitor; this was not a model decision.'));
    const raw = (m?.text || '').trim();
    if (m && (raw || m.thinking)) {
      decision.push(h('details', { class: 'fold' }, h('summary', {}, m.thinking ? 'Raw model output (with thinking)' : 'Raw model output'),
        h('div', { class: 'fold-body stack' },
          m.thinking ? h('div', { class: 'stream-box thinking' }, m.thinking) : null,
          raw ? h('div', { class: 'stream-box' }, raw) : null,
          m.errors?.length ? h('div', { class: 'small muted' }, `Issues along the way: ${m.errors.join('; ')}`) : null)));
    }
  }
  card.append(stage('decision', '② Decision', pending ? 'Model is deciding…' : (m?.label ? `Model: ${m.label}` : ''), ...decision));

  // ③ primitive call: name and every parameter
  if (call) {
    const extra = (turn.decision?.calls?.length || 0) > 1 ? `(model gave ${turn.decision.calls.length} actions; only the first runs)` : '';
    card.append(stage('call', '③ Primitive call', extra,
      h('div', { class: 'call-line' }, h('span', { class: 'call-big' }, call.name),
        brief(call.name, call.args) ? h('span', { class: 'call-arg' }, brief(call.name, call.args)) : null),
      call.name === 'done' ? null : argsTable(call.args)));
  }

  // ④ return: how the primitive ended and the receipt the model reads next turn
  if (call) {
    const ret = [];
    const rep = turn.report;
    if (a.status === 'executing') {
      ret.push(h('div', { class: 'stage-note' }, h('span', { class: 'live-dot', 'aria-hidden': 'true' }), 'The robot is executing this primitive…'));
    } else if (a.status === 'invalid') {
      ret.push(h('div', { class: 'ret-line' }, h('b', {}, 'Invalid, not executed:'), a.invalid || 'Invalid action or arguments'));
      if (turn.translation) ret.push(receiptEl(turn.translation, { hideScene: true }));
    } else if (a.status === 'declined') {
      ret.push(h('div', { class: 'ret-line' }, h('b', {}, 'Not approved by the operator:'), a.declined || '(no message)'));
    } else if (a.status === 'interrupted') {
      ret.push(h('div', { class: 'ret-line' }, 'This episode ended during execution; no result for this step.'));
    } else if (a.status === 'done') {
      const claim = call.args?.success === true ? ', model reports success' : call.args?.success === false ? ', model reports failure' : '';
      ret.push(h('div', { class: 'ret-line' }, h('b', {}, `Episode ended${claim}`)));
    } else {
      const term = a.termination && a.termination !== 'ok' ? a.termination : 'completed';
      ret.push(h('div', { class: 'ret-line' }, termChip(term),
        rep?.duration_s ? h('span', { class: 'num' }, `took ${dur(rep.duration_s)}`) : null,
        rep?.peak_force !== undefined && rep?.peak_force !== null ? h('span', { class: 'num' }, `peak |F| ${num(rep.peak_force, 1)} N`) : null,
        rep?.spark?.length > 1 ? sparkline(rep.spark) : null));
      if (a.ignored_args?.length) ret.push(h('div', { class: 'small muted' }, `Arguments ignored by the runner: ${a.ignored_args.join(', ')}`));
      if (a.error) ret.push(h('div', { class: 'ret-line' }, h('b', {}, 'Error:'), a.error));
      if (turn.translation) {
        ret.push(h('div', { class: 'small muted' }, 'Tool result: the model reads this next turn (blue = force translation by the harness)'),
          receiptEl(turn.translation, { hideScene: true }));
      }
    }
    card.append(stage(`return ${o.cls}`, '④ Return', '', ...ret));
  }
  return card;
}

function manualEditor(turn, manual) {
  const area = h('textarea', { class: 'code', rows: 4, spellcheck: 'false', 'aria-label': 'Manual action JSON' });
  area.value = manual.example || '{"calls": []}';
  const send = h('button', { class: 'btn primary small', type: 'button' }, 'Submit step');
  send.addEventListener('click', () => manual.onSubmit(turn, area.value));
  return h('div', { class: 'manual' },
    h('div', { class: 'small' }, 'You are the model: edit the JSON below and submit (written as-is to a cmd file; all guards still apply)'),
    area, h('div', { class: 'row' }, send));
}
