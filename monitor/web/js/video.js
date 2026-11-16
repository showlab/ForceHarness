// Session videos (global camera, runs/videos/): status chip, size text and the live preview image.
import { dur, h } from './dom.js';

export const VIDEO_STATUS = {
  starting: 'Opening camera', recording: 'Recording', finalizing: 'Finalizing', complete: 'Recorded',
  partial: 'Partial (error mid-recording)', failed: 'Recording failed', interrupted: 'Not finalized',
};

export function videoStatusEl(v) {
  const s = { recording: 'executing', starting: 'starting', finalizing: 'starting', complete: 'ended', failed: 'failed',
    partial: 'stale', interrupted: 'stale' }[v.status] || 'ended';
  return h('span', { class: 'status', dataset: { s }, title: v.error || null },
    h('span', { class: 'dot' }), VIDEO_STATUS[v.status] || v.status);
}

export function bytes(n) {
  if (n === null || n === undefined) return '—';
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(n / 1e3))} KB`;
}

export function videoLength(v) {
  return v.duration_s ?? v.recorded_s ?? null;
}

export function videoFacts(v) {
  const cam = v.camera || {};
  return [dur(videoLength(v)), v.bytes ? bytes(v.bytes) : null,
    cam.width ? `${cam.width}×${cam.height} · ${cam.fps} fps` : null].filter(Boolean).join(' · ');
}

// The recorder rewrites <id>.preview.jpg about once a second; reload it while the recording runs.
export class LivePreview {
  constructor(v) {
    this.url = v.preview_url;
    this.img = h('img', { class: 'video-frame', alt: 'Live view', src: this.url ? `${this.url}&t=${Date.now()}` : null });
    this.el = h('div', { class: 'video-live' }, this.img,
      h('span', { class: 'video-badge' }, h('span', { class: 'live-dot', 'aria-hidden': 'true' }), 'Live'));
    this.timer = setInterval(() => this.refresh(), 1000);
  }

  refresh() {
    if (!this.url || this.loading) return;
    // load off-screen, then swap elements: no flicker, and no second request for a no-store image
    const next = h('img', { class: 'video-frame', alt: 'Live view' });
    this.loading = true;
    next.onload = () => { this.loading = false; this.img.replaceWith(next); this.img = next; };
    next.onerror = () => { this.loading = false; };
    next.src = `${this.url}&t=${Date.now()}`;
  }

  setUrl(url) { this.url = url; }

  destroy() { clearInterval(this.timer); }
}
