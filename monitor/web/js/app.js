// Shell: hash router, theme, connection state. Views live in ./views/.
import { get, onConnection } from './api.js';
import { clear, h, toast } from './dom.js';
import { LiveView } from './views/live.js';
import { EpisodesView } from './views/episodes.js';
import { EpisodeView } from './views/episode.js';
import { VideosView } from './views/videos.js';
import { CompareView } from './views/compare.js';
import { SettingsView } from './views/settings.js';

export const app = { info: null, view: null, task(id) { return (this.info?.tasks || []).find((t) => t.id === id); } };

function store(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    if (value === null) localStorage.removeItem(key); else localStorage.setItem(key, value);
  } catch { /* private window: preferences just don't persist */ }
  return null;
}

function applyTheme(theme) {
  if (theme === 'light' || theme === 'dark') document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
}

function setupTheme() {
  applyTheme(store('fhmon-theme'));
  document.getElementById('theme').addEventListener('click', () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === 'dark'
      : matchMedia('(prefers-color-scheme: dark)').matches;
    const next = dark ? 'light' : 'dark';
    applyTheme(next);
    store('fhmon-theme', next);
  });
}

function setupConnection() {
  const el = document.getElementById('conn');
  const text = el.querySelector('.conn-text');
  const names = { ok: 'Connected', stale: 'Data delayed', down: 'Monitor server unreachable' };
  onConnection((state) => {
    el.dataset.state = state;
    text.textContent = names[state] || state;
  });
}

function route() {
  const hash = location.hash || '#/live';
  const parts = hash.replace(/^#\//, '').split('/');
  const name = parts[0] || 'live';
  if (app.view && app.view.destroy) app.view.destroy();
  app.view = null;
  for (const a of document.querySelectorAll('.tabs a')) {
    const tab = a.dataset.tab;
    const active = tab === name || (name === 'episode' && tab === 'episodes');
    if (active) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
  }
  const main = clear(document.getElementById('view'));
  const views = { live: LiveView, episodes: EpisodesView, videos: VideosView, compare: CompareView, settings: SettingsView };
  try {
    if (name === 'episode') app.view = new EpisodeView(main, app, decodeURIComponent(parts.slice(1).join('/')));
    else app.view = new (views[name] || LiveView)(main, app);
  } catch (err) {
    main.append(h('div', { class: 'empty' }, `Page error: ${err.message}`));
    console.error(err);
  }
  main.focus({ preventScroll: true });
}

async function boot() {
  setupTheme();
  setupConnection();
  try {
    app.info = await get('/api/info');
  } catch (err) {
    toast(`Monitor server unreachable: ${err.message}`, 'error');
    return;
  }
  document.getElementById('live-flag').hidden = !app.info.allow_live;
  window.addEventListener('hashchange', route);
  route();
}

boot();
