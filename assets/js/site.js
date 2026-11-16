/* ForceHarness project page: result bars, the one-call panel, real-robot clip tiles and table, benchmark frames. */
(function () {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else if (v !== null && v !== undefined) n.setAttribute(k, v);
    }
    kids.forEach((c) => c !== null && c !== undefined && n.append(c));
    return n;
  };
  const SVG = "http://www.w3.org/2000/svg";
  const svg = (tag, attrs = {}) => {
    const n = document.createElementNS(SVG, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    return n;
  };

  /* ---------------------------------------------------------------- result bars */
  $$(".bars").forEach((box) => {
    const rows = JSON.parse(box.dataset.bars);
    rows.forEach(([label, pct, text], i) => {
      const ours = i === rows.length - 1;
      const bar = el("div", { class: "bar" + (ours ? " is-ours" : "") },
        el("span", { class: "bar-label", text: label }),
        el("span", { class: "bar-track", style: `--w:${pct * 0.78}%` },
          el("span", { class: "bar-fill", style: `width:${pct * 0.78}%` }),
          el("span", { class: "bar-value", text: text })));
      box.append(bar);
    });
  });

  /* ---------------------------------------------------------------- one call, start to finish */
  const panel = $(".call-panel");
  function drawTrace(node, c) {
    node.replaceChildren();
    const W = 560, H = 170, L = 34, R = 14, T = 14, B = 26;
    const tMax = c.duration_s;
    const fMax = Math.max(10, Math.ceil(Math.max(...c.trace.map((p) => p[1])) / 5) * 5);
    const x = (t) => L + (t / tMax) * (W - L - R);
    const y = (f) => T + (1 - f / fMax) * (H - T - B);
    [0, fMax / 2, fMax].forEach((f) => {
      node.append(svg("line", { class: "grid", x1: L, x2: W - R, y1: y(f), y2: y(f) }));
      const t = svg("text", { class: "axis", x: L - 6, y: y(f) + 4, "text-anchor": "end" });
      t.textContent = f;
      node.append(t);
    });
    const yl = svg("text", { class: "axis", x: L - 6, y: T - 3, "text-anchor": "end" });
    yl.textContent = "|F| N";
    node.append(yl);
    const pts = c.trace.map(([t, f]) => `${x(t).toFixed(1)},${y(f).toFixed(1)}`);
    const area = svg("path", { class: "area", d: `M${x(c.trace[0][0])},${y(0)} L${pts.join(" L")} L${x(c.trace[c.trace.length - 1][0])},${y(0)} Z` });
    const line = svg("path", { class: "line", d: `M${pts.join(" L")}` });
    node.append(area, line);
    const mx = x(c.marker.t);
    node.append(svg("line", { class: "marker", x1: mx, x2: mx, y1: T, y2: H - B }));
    const ml = svg("text", { class: "marker-label", x: Math.min(mx + 5, W - R - 60), y: H - B - 6 });
    ml.textContent = c.marker.label;
    node.append(ml);
    const end = svg("text", { class: "axis", x: W - R, y: H - 6, "text-anchor": "end" });
    end.textContent = `${c.duration_s.toFixed(1)} s, ${c.termination}`;
    const start = svg("text", { class: "axis", x: L, y: H - 6 });
    start.textContent = "0 s";
    node.append(end, start);
    const len = line.getTotalLength();
    line.style.strokeDasharray = len;
    line.style.setProperty("--len", len);
    line.classList.add("draw");
    area.classList.add("draw");
  }
  function showCall(c, data) {
    const img = $(".call-photo img", panel);
    img.src = c.photo;
    img.alt = `${c.title}: the robot at contact`;
    $(".call-photo figcaption", panel).textContent = `${c.title}. Photo from another episode of the same task.`;
    $('[data-field="call"]', panel).textContent = c.call;
    drawTrace($(".trace", panel), c);
    $('[data-field="receipt"]', panel).textContent = `${c.result_line}\n${c.receipt_line.replace(/\s{2,}/g, "  ")}`;
    $('[data-field="next"]', panel).textContent = c.next_call;
    const note = $(".note", panel);
    note.replaceChildren(`“${c.note}”`, el("small", { text: "the model's note, translated" }));
  }
  if (panel) {
    fetch("assets/data/calls.json").then((r) => r.json()).then((data) => {
      const tabs = $(".tabs");
      const buttons = data.calls.map((c, i) => {
        const b = el("button", { class: "tab", type: "button", role: "tab", "aria-selected": i === 0 ? "true" : "false",
          tabindex: i === 0 ? "0" : "-1", text: c.title });
        b.addEventListener("click", () => select(i));
        tabs.append(b);
        return b;
      });
      function select(i) {
        buttons.forEach((b, j) => { b.setAttribute("aria-selected", j === i ? "true" : "false"); b.tabIndex = j === i ? 0 : -1; });
        showCall(data.calls[i], data);
      }
      tabs.addEventListener("keydown", (e) => {
        const i = buttons.indexOf(document.activeElement);
        if (i < 0) return;
        const j = e.key === "ArrowRight" ? (i + 1) % buttons.length : e.key === "ArrowLeft" ? (i - 1 + buttons.length) % buttons.length : -1;
        if (j >= 0) { e.preventDefault(); buttons[j].focus(); select(j); }
      });
      select(0);
    }).catch(() => {
      panel.replaceChildren(el("p", { class: "section-intro", text: "Open this page through a web server to see the logged calls (python -m http.server)." }));
    });
  }

  /* ---------------------------------------------------------------- real-robot table + clips */
  const REAL = [
    ["blackboard_wipe", "Wipe a whiteboard", 1, 8, 6],
    ["screw_insert", "Drive a screw", 0, 9, 7],
    ["peel_vegetable", "Peel an eggplant", 1, 9, 9],
    ["usb_insert", "Insert a USB plug", 0, 8, 7],
    ["can_weight_sort", "Sort three bottles by weight", 0, 10, 8],
    ["touch_firmness", "Tell soft from hard", 0, 10, 10],
  ];
  const isBaseline = (cond) => /no[-\s]?force|baseline|^b$/i.test(String(cond || ""));
  const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const onScreen = "IntersectionObserver" in window ? new IntersectionObserver((entries) => entries.forEach((e) => {
    const v = e.target;
    if (e.isIntersecting && !reduceMotion) v.play().catch(() => {});
    else v.pause();
  }), { threshold: 0.35 }) : null;

  /* One tile per task: the ForceHarness episode by default, a switch to the no-force control when it was recorded. */
  function clipTile(title, ours, base, nf, fh) {
    const video = el("video", { muted: "", loop: "", playsinline: "", preload: "none" });
    video.muted = true;
    if (reduceMotion) video.controls = true;
    const speed = el("span", { class: "tile-speed" });
    const outcome = el("span", { class: "tile-outcome" });
    const media = el("div", { class: "tile-media" }, video, outcome);
    const show = (entry) => {
      video.pause();
      video.poster = `assets/video/${entry.poster}`;
      video.src = `assets/video/${entry.file}`;
      media.classList.toggle("is-landscape", (entry.width || 0) > (entry.height || 0));
      speed.textContent = `Autonomous, ${entry.speed}x`;
      const ok = /success/i.test(entry.outcome || "");
      outcome.textContent = ok ? "Success" : "Failure";
      outcome.classList.toggle("is-ok", ok);
      const r = media.getBoundingClientRect();
      if (!reduceMotion && r.bottom > 0 && r.top < innerHeight) video.play().catch(() => {});
    };
    const buttons = [];
    const option = (label, entry, score) => {
      const b = el("button", { type: "button", "aria-pressed": "false", disabled: entry ? null : "" },
        el("span", { text: label }), el("small", { text: entry ? score : `${score}, not filmed` }));
      if (entry) b.addEventListener("click", () => { buttons.forEach((x) => x.setAttribute("aria-pressed", String(x === b))); show(entry); });
      buttons.push(b);
      return b;
    };
    const sw = el("div", { class: "tile-switch", role: "group", "aria-label": `${title}: choose a condition` },
      option("ForceHarness", ours, `${fh}/10`), option("No force", base, `${nf}/10`));
    const first = ours || base;
    if (first) { show(first); buttons[ours ? 0 : 1].setAttribute("aria-pressed", "true"); }
    if (onScreen) onScreen.observe(video);
    return el("figure", { class: "tile" }, media, el("figcaption", {}, el("strong", { text: title }), speed, sw));
  }

  const tbody = $(".real-table tbody");
  const grid = $("[data-clips]");
  fetch("assets/video/manifest.json").then((r) => (r.ok ? r.json() : null)).catch(() => null).then((manifest) => {
    const clips = (manifest && (manifest.clips || manifest)) || [];
    const list = Array.isArray(clips) ? clips : [];
    REAL.forEach(([id, title, nf, fh, sol]) => {
      const mine = list.filter((c) => c.task === id && c.file && !/teaser/i.test(c.file));
      const ours = mine.find((c) => !isBaseline(c.condition) && !isBaseline(c.condition_label));
      const base = mine.find((c) => isBaseline(c.condition) || isBaseline(c.condition_label));
      if (grid && (ours || base)) grid.append(clipTile(title, ours, base, nf, fh));
      const score = (n, win) => el("td", { class: win ? "win" : null },
        el("span", { class: "score" }, el("span", { text: `${n}/10` }), el("span", { class: "mini" }, el("i", { style: `width:${n * 10}%` }))));
      const head = el("th", { scope: "row" }, el("span", { class: "task-name", text: title }));
      tbody.append(el("tr", {}, head, score(nf, false), score(fh, true), score(sol, false)));
    });
  });

  /* ---------------------------------------------------------------- benchmark frames */
  const FRAMES = {
    fb: [["weight", "Weight"], ["com", "CoM"], ["faucet", "Faucet"], ["faucetdir", "Faucet dir."], ["stove", "Stove"],
      ["stovepress", "Stove press"], ["drawer", "Drawer"], ["nut", "Nut"]],
    hp: [["weight", "Weight"], ["com", "CoM"], ["wipe", "Wipe"], ["stack", "Stack"], ["peg", "Peg"], ["latch", "Latch"], ["lock", "Lock"]],
  };
  $$("[data-frames]").forEach((box) => {
    const key = box.dataset.frames;
    FRAMES[key].forEach(([f, label]) => box.append(el("figure", {},
      el("img", { src: `assets/img/settings/fig3_${key}_${f}.webp`, alt: label, loading: "lazy" }), el("figcaption", { text: label }))));
  });

  /* ---------------------------------------------------------------- copy buttons (BibTeX, agent prompt) */
  $$("[data-copy]").forEach((b) => b.addEventListener("click", () => {
    const text = $(b.dataset.copy).textContent;
    const label = b.dataset.label || (b.dataset.label = b.textContent);
    navigator.clipboard.writeText(text).then(() => { b.textContent = "Copied"; setTimeout(() => (b.textContent = label), 1600); });
  }));
})();
