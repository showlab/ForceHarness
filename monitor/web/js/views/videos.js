// Every session video in one folder (runs/videos/ by default): newest first, played here, linked to its episode.
import { get } from '../api.js';
import { armBadge, banner, clear, dur, epHref, h, modeBadge, when } from '../dom.js';
import { LivePreview, bytes, videoFacts, videoLength, videoStatusEl } from '../video.js';

export class VideosView {
  constructor(root, app) {
    this.root = root;
    this.app = app;
    this.list = [];
    this.config = null;
    this.selected = null;   // id of the video in the player
    this.shown = null;      // what the player shows; a list refresh must not restart playback
    this.preview = null;
    this.countEl = h('span', { class: 'small muted' });
    this.noteEl = h('div', { class: 'stack', style: { marginBottom: '14px' } });
    this.headEl = h('div', { class: 'panel-head' });
    this.playerEl = h('div', { class: 'panel-body video-player' });
    this.tableEl = h('div', { class: 'table-wrap' });
    root.append(
      h('div', { class: 'section-title' }, h('h1', {}, 'Videos'), this.countEl),
      this.noteEl,
      h('section', { class: 'panel' }, this.headEl, this.playerEl),
      h('section', { class: 'panel', style: { marginTop: '14px' } }, this.tableEl));
    this.load();
    this.timer = setInterval(() => this.load(), 3000);
  }

  destroy() {
    clearInterval(this.timer);
    if (this.preview) this.preview.destroy();
  }

  async load() {
    try {
      const r = await get('/api/videos');
      this.list = r.videos;
      this.config = r.config;
      this.render();
    } catch (err) {
      clear(this.tableEl).append(h('div', { class: 'empty' }, `Failed to read: ${err.message}`));
    }
  }

  render() {
    const cfg = this.config || {};
    const cam = cfg.camera || {};
    this.countEl.textContent = `Videos: ${this.list.length} · files in ${cfg.dir || 'runs/videos'}/`;
    clear(this.noteEl);
    if (!cfg.enabled) {
      this.noteEl.append(banner('warning', [h('b', {}, 'Automatic recording is off: '), cfg.error || 'recording is disabled']));
    } else {
      const which = (cfg.modes || []).includes('demo') ? 'Each episode' : 'Each real-robot episode';
      this.noteEl.append(h('div', { class: 'small ink2' },
        `${which} is recorded automatically with ${cam.label || 'camera'}${cam.serial ? ` · serial ${cam.serial}` : ''} at ${cam.width}×${cam.height} · ${cam.fps} fps, plus a few seconds after the episode ends.`));
    }
    if (!this.list.some((v) => v.id === this.selected)) {
      const first = this.list.find((v) => v.active) || this.list[0];
      this.selected = first ? first.id : null;
    }
    this.renderPlayer();
    this.renderTable();
  }

  renderPlayer() {
    const v = this.list.find((x) => x.id === this.selected);
    clear(this.headEl);
    if (!v) {
      this.headEl.append(h('h2', {}, 'No videos yet'));
      this.show(null, () => h('div', { class: 'empty' }, 'Start a real-robot episode from the Live tab and its video appears here.'));
      return;
    }
    const s = v.session || {};
    this.headEl.append(
      h('h2', {}, s.task_name || s.task || v.id),
      h('span', { class: 'small muted nowrap' }, `${when(v.first_frame_at || v.started)} · ${videoFacts(v)}`),
      videoStatusEl(v),
      v.episode_key ? h('a', { class: 'btn small', href: epHref(v.episode_key) }, 'Open episode') : null);
    this.show(`${v.id}|${v.status}|${v.url}`, () => {
      if (['starting', 'recording', 'finalizing'].includes(v.status)) {
        this.preview = new LivePreview(v);
        return h('div', { class: 'stack', style: { gap: '8px' } }, this.preview.el,
          h('div', { class: 'small muted' }, v.status === 'finalizing'
            ? 'Episode ended; finalizing the video into a seekable file. Playback is ready in a few seconds.'
            : 'Live view refreshes about once a second; playback is ready a few seconds after the episode ends.'));
      }
      const body = h('div', { class: 'stack', style: { gap: '8px' } });
      if (v.status === 'failed' || v.status === 'partial' || v.status === 'interrupted') {
        body.append(banner(v.status === 'failed' ? 'critical' : 'warning', [
          h('b', {}, v.status === 'failed' ? 'This recording failed.' : v.status === 'partial' ? 'Only partly recorded.' : 'The recorder did not finish cleanly; seeking may not work.'),
          v.error ? ` ${v.error}` : '', ' ',
          v.log_url ? h('a', { href: v.log_url, target: '_blank', rel: 'noopener' }, 'View recording log') : null]));
      }
      if (v.url) {
        body.append(h('video', { class: 'video-frame', src: v.url, controls: true, preload: 'metadata', playsinline: true }),
          h('div', { class: 'small muted' }, 'File ', h('a', { href: v.url, target: '_blank', rel: 'noopener' }, v.path || v.file)));
      }
      return body;
    });
  }

  // Rebuild the player only when it would show something else (keeps a playing video playing).
  show(sig, build) {
    if (sig !== null && sig === this.shown) return;
    this.shown = sig;
    if (this.preview) { this.preview.destroy(); this.preview = null; }
    clear(this.playerEl).append(build());
  }

  renderTable() {
    const scroll = this.tableEl.scrollTop;
    clear(this.tableEl);
    if (!this.list.length) {
      this.tableEl.append(h('div', { class: 'empty' }, 'No videos yet.'));
      return;
    }
    const table = h('table', { class: 'data' },
      h('thead', {}, h('tr', {},
        h('th', {}, 'Started'), h('th', {}, 'Task · episode name'), h('th', {}, 'Condition'), h('th', {}, 'Mode'),
        h('th', { class: 'num' }, 'Duration'), h('th', { class: 'num' }, 'Size'), h('th', {}, 'Status'), h('th', {}, ''))));
    const tb = h('tbody');
    for (const v of this.list) {
      const s = v.session || {};
      const tr = h('tr', { tabindex: '0', class: v.id === this.selected ? 'selected' : null },
        h('td', { class: 'num nowrap' }, when(v.first_frame_at || v.started)),
        h('td', {}, h('div', {}, s.task_name || s.task || '—'), h('div', { class: 'small muted' }, v.id)),
        h('td', {}, s.arm ? armBadge(s.arm, s.condition_label) : '—'),
        h('td', {}, s.mode ? modeBadge(s.mode) : '—'),
        h('td', { class: 'num nowrap' }, dur(videoLength(v))),
        h('td', { class: 'num nowrap' }, v.bytes ? bytes(v.bytes) : '—'),
        h('td', {}, videoStatusEl(v)),
        h('td', {}, v.episode_key ? h('a', { href: epHref(v.episode_key), onclick: (ev) => ev.stopPropagation() }, 'Episode') : null));
      const pick = () => {
        this.selected = v.id;
        this.render();
        this.root.scrollIntoView?.({ behavior: 'smooth', block: 'start' });
      };
      tr.addEventListener('click', pick);
      tr.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') pick(); });
      tb.append(tr);
    }
    table.append(tb);
    this.tableEl.append(table);
    this.tableEl.scrollTop = scroll;
  }
}
