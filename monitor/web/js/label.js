// Operator label: a hand-set success flag and a note per episode, kept by the monitor (labels.json).
// The flag wins over the harness score in every table and in the comparison; clearing it falls back.
import { post } from './api.js';
import { h, icon, toast } from './dom.js';

async function save(key, success, note) {
  const r = await post('/api/label', { key, success, note });
  return r.label;
}

// ✓ / ✗ toggles plus the note. Clicking the pressed toggle clears the flag. onSaved(label) after each save.
export function labelCell(ep, onSaved, { showNote = true } = {}) {
  const lab = ep.result?.label || null;
  const flag = lab?.success ?? null;
  const note = lab?.note || '';
  const stop = (ev) => ev.stopPropagation();
  const box = h('div', { class: 'label-cell', onclick: stop, onkeydown: stop });
  const toggle = (value, glyph, text) => {
    const pressed = flag === value;
    const b = h('button', { type: 'button', class: `flag-btn ${glyph}`, 'aria-pressed': String(pressed),
      title: pressed ? `Clear manual label (back to the harness score)` : `Manual label: ${text}` }, icon(glyph));
    b.addEventListener('click', async () => {
      b.disabled = true;
      try { onSaved(await save(ep.key, pressed ? null : value, note)); } catch (err) { toast(err.message, 'error'); b.disabled = false; }
    });
    return b;
  };
  box.append(toggle(true, 'ok', 'Success'), toggle(false, 'bad', 'Failure'));
  const edit = h('button', { type: 'button', class: `note-btn${note ? '' : ' empty'}`, title: note ? `${note}\n(click to edit)` : 'Add note' },
    showNote && note ? note : (note ? 'Edit note' : '+ Note'));
  edit.addEventListener('click', () => openNoteDialog(ep, flag, note, onSaved));
  box.append(edit);
  return box;
}

function openNoteDialog(ep, flag, note, onSaved) {
  const ta = h('textarea', { rows: 5, placeholder: 'Note for this episode: what you saw, why you judged success or failure…', style: { width: '100%' } });
  ta.value = note;
  const ok = h('button', { type: 'submit', class: 'btn primary small' }, 'Save');
  const cancel = h('button', { type: 'button', class: 'btn small', onclick: () => dlg.close() }, 'Cancel');
  const form = h('form', { method: 'dialog', class: 'stack', style: { gap: '10px' } },
    h('h2', {}, 'Note'), h('div', { class: 'small muted' }, ep.name || ep.key), ta,
    h('div', { class: 'row', style: { justifyContent: 'flex-end' } }, cancel, ok));
  const dlg = h('dialog', { class: 'note-dialog' }, form);
  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    ok.disabled = true;
    try {
      const lab = await save(ep.key, flag, ta.value);
      dlg.close();
      onSaved(lab);
    } catch (err) { toast(err.message, 'error'); ok.disabled = false; }
  });
  ta.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' && (ev.ctrlKey || ev.metaKey)) form.requestSubmit(); });
  dlg.addEventListener('close', () => dlg.remove());
  document.body.append(dlg);
  dlg.showModal();
  ta.focus();
}
