// Talking to the monitor server: JSON GET/POST (POST carries the page token) and the live stream.
const TOKEN = document.querySelector('meta[name="fhmon-token"]')?.content || '';
const listeners = new Set();

export function onConnection(fn) { listeners.add(fn); return () => listeners.delete(fn); }
function report(state) { for (const fn of listeners) fn(state); }

async function handle(res) {
  let data = null;
  try { data = await res.json(); } catch { /* empty body */ }
  if (!res.ok) throw new Error((data && data.error) || `HTTP ${res.status}`);
  return data;
}

export async function get(path, params) {
  const url = params ? `${path}?${new URLSearchParams(params)}` : path;
  try {
    const res = await fetch(url, { cache: 'no-store' });
    report('ok');
    return handle(res);
  } catch (err) {
    if (err instanceof TypeError) report('down');
    throw err;
  }
}

export async function post(path, body) {
  try {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-FHMon-Token': TOKEN },
      body: JSON.stringify(body || {}),
    });
    report('ok');
    return handle(res);
  } catch (err) {
    if (err instanceof TypeError) report('down');
    throw err;
  }
}

// One EventSource per episode view; reconnects on its own. onEvent(evt) gets parsed JSON.
export function stream(key, onEvent) {
  let closed = false;
  let es = null;
  let lastBeat = Date.now();
  const open = () => {
    es = new EventSource(`/api/stream?key=${encodeURIComponent(key)}`);
    es.onmessage = (m) => {
      lastBeat = Date.now();
      report('ok');
      let evt;
      try { evt = JSON.parse(m.data); } catch { return; }
      onEvent(evt);
    };
    es.onerror = () => { if (!closed) report('stale'); };
  };
  open();
  const watchdog = setInterval(() => {
    if (Date.now() - lastBeat > 15000) report('stale');
  }, 3000);
  return {
    close() { closed = true; clearInterval(watchdog); es && es.close(); },
    sinceBeat() { return (Date.now() - lastBeat) / 1000; },
  };
}

export function fileUrl(url) { return url; }
