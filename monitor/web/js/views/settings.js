// Model API profiles. A key typed here stays in the monitor's memory only (gone on restart);
// the other way is an environment variable such as FH_API_KEY from the repository's .env.
import { get, post } from '../api.js';
import { banner, clear, h, toast } from '../dom.js';

const EFFORTS = ['', 'none', 'minimal', 'low', 'medium', 'high', 'xhigh'];
const DEFAULT_BASE = { openai: 'https://api.openai.com/v1', 'openai-responses': 'https://api.openai.com/v1', anthropic: 'https://api.anthropic.com', 'codex-cli': 'https://chatgpt.com' };
const DEFAULT_ENV = { openai: 'FH_API_KEY', 'openai-responses': 'FH_API_KEY', anthropic: 'ANTHROPIC_API_KEY', 'codex-cli': '' };

export class SettingsView {
  constructor(root, app) {
    this.root = root;
    this.app = app;
    this.profiles = [];
    this.providers = {};
    this.selected = null;
    this.listEl = h('div', {});
    this.editorEl = h('div', { class: 'panel-body' });
    this.testEl = h('div', {});
    const add = h('button', { class: 'btn small', type: 'button' }, 'New profile');
    add.addEventListener('click', () => { this.selected = '__new__'; this.render(); });
    root.append(
      h('div', { class: 'section-title' }, h('h1', {}, 'Settings'), h('span', { class: 'hint' }, 'Model API profiles; keys are never written to a file')),
      h('div', { class: 'settings-grid' },
        h('section', { class: 'panel' }, h('div', { class: 'panel-head' }, h('h2', {}, 'API profiles'), add), this.listEl),
        h('div', { class: 'stack', style: { gap: '16px' } },
          h('section', { class: 'panel' }, h('div', { class: 'panel-head' }, h('h2', {}, 'Edit')), this.editorEl),
          this.aboutPanel())));
    this.load();
  }

  destroy() {}

  async load() {
    const s = await get('/api/settings');
    this.profiles = s.profiles;
    this.providers = s.providers;
    this.bridgeSystem = s.bridge_system;
    if (!this.selected || (this.selected !== '__new__' && !this.profiles.some((p) => p.id === this.selected))) {
      this.selected = this.profiles[0]?.id || '__new__';
    }
    this.render();
  }

  render() {
    clear(this.listEl);
    for (const p of this.profiles) {
      const el = h('div', { class: `profile-item${p.id === this.selected ? ' selected' : ''}`, tabindex: '0' },
        h('div', { class: 'row' }, h('b', {}, p.name), h('span', { class: 'small muted' }, this.providers[p.provider] || p.provider)),
        h('div', { class: 'small ink2' }, p.model || '(no model name yet)'),
        h('div', { class: `keystate ${p.has_key ? 'ok' : 'none'}` },
          p.provider === 'codex-cli' ? (p.has_key ? 'Codex CLI found (ChatGPT plan, not billed per token)' : 'Codex CLI not found')
            : p.has_key ? (p.key_source === 'memory' ? 'Key: in memory (re-enter after restart)' : `Key: from environment variable ${p.key_source.slice(4)}`) : 'No key'));
      el.addEventListener('click', () => { this.selected = p.id; this.render(); });
      this.listEl.append(el);
    }
    this.renderEditor();
  }

  renderEditor() {
    const isNew = this.selected === '__new__';
    const p = isNew
      ? { id: '', name: 'New profile', provider: 'openai', base_url: '', model: '', key_env: '', max_tokens: 4096, temperature: null,
        reasoning_effort: '', image_turns: 2, image_detail: 'auto', price_in_per_m: null, price_out_per_m: null, extra_body: {} }
      : this.profiles.find((x) => x.id === this.selected);
    clear(this.editorEl);
    if (!p) return;
    const fields = {};
    const input = (key, label, attrs = {}, help = '') => {
      const el = h('input', Object.assign({ type: 'text' }, attrs));
      el.value = p[key] === null || p[key] === undefined ? '' : String(p[key]);
      fields[key] = el;
      return h('label', { class: `field${attrs.wide ? ' wide' : ''}` }, h('span', {}, label), el, help ? h('span', { class: 'help' }, help) : null);
    };
    const select = (key, label, options) => {
      const el = h('select', {}, options.map(([v, t]) => h('option', { value: v }, t)));
      el.value = p[key] ?? '';
      fields[key] = el;
      return h('label', { class: 'field' }, h('span', {}, label), el);
    };
    const keyEl = h('input', { type: 'password', autocomplete: 'off', placeholder: p.has_key ? 'Key is set; leave blank to keep it' : 'sk-… (kept in memory only)' });
    const extra = h('textarea', { class: 'code', rows: 3, spellcheck: 'false' });
    extra.value = p.extra_body && Object.keys(p.extra_body).length ? JSON.stringify(p.extra_body, null, 2) : '';
    const model = this.modelPicker(p, isNew, fields, keyEl);
    // fields that only mean something for an HTTP API; the Codex CLI uses its own login and backend
    const apiOnly = [
      input('base_url', 'Base URL', { placeholder: 'Leave blank for the official URL', wide: true }, 'For other OpenAI-compatible endpoints (for example Gemini or a local vLLM server), enter their /v1 URL.'),
      h('label', { class: 'field' }, h('span', {}, 'API key'), keyEl,
        h('span', { class: 'help' }, 'Kept only in the memory of the monitor process; re-enter after a restart.')),
      input('key_env', 'Or read from environment variable', { placeholder: 'FH_API_KEY' }, 'Start the monitor with monitor/run.sh or --env-file .env to load the keys in .env.'),
      input('max_tokens', 'Max output tokens', { type: 'number', min: 1 }),
      input('temperature', 'temperature', { type: 'number', step: '0.1', min: 0, max: 2, placeholder: 'Blank = default' }),
      select('image_detail', 'Image detail (OpenAI)', [['auto', 'auto'], ['low', 'low'], ['high', 'high']]),
      input('price_in_per_m', 'Price: input $/M tokens', { type: 'number', step: '0.01', min: 0, placeholder: 'Optional' }),
      input('price_out_per_m', 'Price: output $/M tokens', { type: 'number', step: '0.01', min: 0, placeholder: 'Optional' }),
      h('label', { class: 'field wide' }, h('span', {}, 'Extra request parameters (JSON, optional)'), extra,
        h('span', { class: 'help' }, 'Merged into the request body as is, e.g. {"thinking": {"type": "enabled", "budget_tokens": 4000}} for Anthropic.'))];
    const codexNote = h('div', { class: 'field wide help' },
      'Uses the Codex CLI (codex on PATH, or --codex-bin / CODEX_BIN) and the ChatGPT plan it is signed in with: no key and no per-token billing, but it draws on the Codex allowance of the plan (limits every few hours and per week). '
      + 'Each turn starts one codex exec (read-only, no saved session) and takes about 20–60 s; blank reasoning effort = the setting in ~/.codex/config.toml; medium is faster.');
    const grid = h('div', { class: 'form-grid' },
      input('name', 'Name'),
      select('provider', 'Interface', Object.entries(this.providers)),
      codexNote,
      apiOnly[0],
      model.el,
      ...apiOnly.slice(1, 5),
      select('reasoning_effort', 'reasoning effort', EFFORTS.map((e) => [e, e || '(not sent)'])),
      input('image_turns', 'Recent turns with images', { type: 'number', min: 0, max: 20 }, 'Earlier turns keep only file names, which saves tokens.'),
      ...apiOnly.slice(5));
    const showFields = () => {
      const codex = fields.provider.value === 'codex-cli';
      for (const el of apiOnly) el.hidden = codex;
      codexNote.hidden = !codex;
    };
    fields.provider.addEventListener('change', showFields);
    showFields();
    const save = h('button', { class: 'btn primary', type: 'button' }, 'Save');
    const test = h('button', { class: 'btn', type: 'button', disabled: isNew ? true : null }, 'Test connection');
    const forget = h('button', { class: 'btn ghost', type: 'button', disabled: p.key_source === 'memory' ? null : true }, 'Clear key from memory');
    const del = h('button', { class: 'btn danger', type: 'button', disabled: isNew ? true : null }, 'Delete');
    this.testEl = h('div', {});
    model.wire();
    const collect = () => {
      const out = { id: isNew ? '' : p.id, model: model.value() };
      for (const [k, el] of Object.entries(fields)) out[k] = el.value;
      for (const k of ['max_tokens', 'image_turns']) out[k] = out[k] === '' ? null : Number(out[k]);
      for (const k of ['temperature', 'price_in_per_m', 'price_out_per_m']) out[k] = out[k] === '' ? null : Number(out[k]);
      out.extra_body = extra.value.trim();
      return out;
    };
    save.addEventListener('click', async () => {
      try {
        const r = await post('/api/settings/profile', { profile: collect(), key: keyEl.value || null });
        keyEl.value = '';
        this.selected = r.profile.id;
        if (r.profile.key_dropped) toast('Interface or Base URL changed: the old key in memory was cleared; enter the key again', 'error');
        else toast('Saved (key not written to a file)');
        await this.load();
      } catch (err) { toast(err.message, 'error'); }
    });
    test.addEventListener('click', async () => {
      test.disabled = true;
      clear(this.testEl).append(h('div', { class: 'small muted', style: { marginTop: '8px' } }, 'Requesting…'));
      try {
        const r = await post('/api/settings/test', { id: p.id });
        clear(this.testEl).append(r.ok
          ? banner('info', [h('b', {}, 'Connected ✓ '), `${r.latency_s}s · reply "${r.text || '(empty)'}" · tokens ${r.usage_in ?? '—'}/${r.usage_out ?? '—'}${r.thinking ? ' · with thinking' : ''}`])
          : banner('critical', [h('b', {}, 'Failed: '), r.error]));
      } catch (err) {
        clear(this.testEl).append(banner('critical', err.message));
      } finally { test.disabled = false; }
    });
    forget.addEventListener('click', async () => {
      await post('/api/settings/profile', { profile: Object.assign(collect(), { extra_body: p.extra_body }), clear_key: true });
      toast('Cleared');
      await this.load();
    });
    del.addEventListener('click', async () => {
      if (!confirm(`Delete profile "${p.name}"?`)) return;
      await post('/api/settings/delete', { id: p.id });
      this.selected = null;
      await this.load();
    });
    this.editorEl.append(grid, h('div', { class: 'row', style: { marginTop: '14px' } }, save, test, forget, h('span', { class: 'grow' }), del), this.testEl);
  }

  // Model name: the models this key can use (the server asks the API, so the key never reaches the page),
  // the common names from the harness price table, and Custom… for anything else.
  modelPicker(p, isNew, fields, keyEl) {
    const sel = h('select', { 'aria-label': 'Model name', style: { flex: '1 1 auto', minWidth: '0' } });
    const custom = h('input', { type: 'text', placeholder: 'Enter a model name, e.g. gpt-6-astra', hidden: true });
    const help = h('span', { class: 'help' });
    const refresh = h('button', { class: 'btn small', type: 'button', title: 'Fetch the model list from the API again with this key' }, 'Refresh list');
    let lists = { models: [], presets: [] };
    let seq = 0;
    let resetModel = false;
    const value = () => (sel.value === '__custom__' ? custom.value.trim() : sel.value);
    const fill = (current) => {
      clear(sel);
      const known = [...lists.models, ...lists.presets];
      const seen = new Set();
      if (!current) sel.append(h('option', { value: '' }, '— Select a model —'));
      else if (!known.some((m) => m.id === current)) {
        sel.append(h('option', { value: current }, `${current} (current)`));
        seen.add(current);
      }
      const group = (label, items) => {
        const g = h('optgroup', { label });
        for (const m of items) {
          if (seen.has(m.id)) continue;
          seen.add(m.id);
          g.append(h('option', { value: m.id }, m.label || m.id));
        }
        if (g.children.length) sel.append(g);
      };
      group(fields.provider?.value === 'codex-cli' ? `Available on the ChatGPT plan (${lists.models.length})`
        : `Available to this key (${lists.models.length})`, lists.models);
      group(fields.provider?.value === 'anthropic' ? 'Common' : 'Common (price table)', lists.presets);
      sel.append(h('option', { value: '__custom__' }, 'Custom… (enter manually)'));
      sel.value = current || '';
      custom.hidden = true;
    };
    const load = async (again = false) => {
      const mine = ++seq;
      const current = resetModel ? '' : (value() || p.model || '');
      resetModel = false;
      help.textContent = 'Fetching model list…';
      try {
        const r = await post('/api/settings/models', {
          id: isNew ? '' : p.id, key: keyEl.value || null, refresh: again,
          profile: { provider: fields.provider.value, base_url: fields.base_url.value, key_env: fields.key_env.value },
        });
        if (mine !== seq) return;                 // a newer request has replaced this one
        lists = r;
        fill(current);
        if (r.error) help.textContent = `Could not get the list from the API: ${r.error}. Pick from Common, or choose Custom.`;
        else if (r.source === 'codex-login') {
          help.textContent = `Models in the local Codex cache: ${r.models.length} (no API call, no allowance used).`;
        } else if (r.source) {
          const from = r.source === 'typed' ? 'the key just entered' : r.source === 'memory' ? 'the key in memory' : `environment variable ${r.source.slice(4)}`;
          help.textContent = `Models from the API: ${r.models.length} (using ${from}); selecting one fills in prices from the price table (editable).`;
        } else help.textContent = 'No key yet: listing common models. Enter a key or environment variable, then click Refresh list for the full list this key can use.';
      } catch (err) {
        if (mine === seq) help.textContent = err.message;
      }
    };
    fill(p.model || '');
    const wire = () => {
      sel.addEventListener('change', () => {
        custom.hidden = sel.value !== '__custom__';
        if (!custom.hidden) { custom.value = ''; custom.focus(); return; }
        const m = [...lists.models, ...lists.presets].find((x) => x.id === sel.value);
        if (m && m.price_in !== null && m.price_in !== undefined && !fields.price_in_per_m.value && !fields.price_out_per_m.value) {
          fields.price_in_per_m.value = m.price_in;
          fields.price_out_per_m.value = m.price_out;
          help.textContent = `Prices filled from the price table forceharness/eval/prices.yaml (input ${m.price_in} / output ${m.price_out} $/M tokens); editable.`;
        }
      });
      refresh.addEventListener('click', () => load(true));
      let last = fields.provider.value;
      fields.provider.addEventListener('change', () => {
        // keep the defaults in step with the interface, so a key is never pointed at another vendor
        const prev = last, next = fields.provider.value;
        last = next;
        if (!fields.base_url.value.trim() || fields.base_url.value.trim() === DEFAULT_BASE[prev]) fields.base_url.value = DEFAULT_BASE[next];
        if (!fields.key_env.value.trim() || fields.key_env.value.trim() === DEFAULT_ENV[prev]) fields.key_env.value = DEFAULT_ENV[next];
        resetModel = true;
        load();
      });
      for (const el of [fields.base_url, fields.key_env, keyEl]) el.addEventListener('change', () => load());
      load();
    };
    const el = h('div', { class: 'field' }, h('span', {}, 'Model name'),
      h('div', { class: 'row', style: { flexWrap: 'nowrap' } }, sel, refresh), custom, help);
    return { el, wire, value };
  }

  aboutPanel() {
    const info = this.app.info;
    const kv = h('dl', { class: 'kv' });
    const add = (k, v) => kv.append(h('dt', {}, k), h('dd', {}, v));
    add('Harness', info.harness_root);
    add('Indexed runs folder', info.runs_root);
    add('Monitor episodes', info.data_dir);
    add('Runner Python', info.python);
    add('Real robot', info.allow_live ? 'Enabled (--allow-live)' : 'Off: demo and records only. Restart with --allow-live when needed');
    const prompt = h('details', { class: 'fold', style: { marginTop: '10px' } }, h('summary', {}, 'System prompt for the API model on relay (same for both conditions)'),
      h('div', { class: 'fold-body' }, h('pre', { class: 'obs' }, info.bridge_system || '')));
    return h('section', { class: 'panel' }, h('div', { class: 'panel-head' }, h('h2', {}, 'About this monitor')),
      h('div', { class: 'panel-body' }, kv, prompt,
        h('p', { class: 'small muted', style: { marginBottom: 0 } },
          'Task prompts, the action menu, force translation, and all guards live in the harness runner; the monitor only passes observations to the model and writes replies back to cmd files over the relay protocol.')));
  }
}
