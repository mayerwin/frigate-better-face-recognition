"use strict";
// Better Face Recognition for Frigate -- vanilla client, no build step.

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

let STATE = { persons: [], frigate_persons: [], settings: {}, counts: {} };
let activeTab = "review";

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (res.status === 401) { location.href = "/login"; throw new Error("not signed in"); }
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return res.status === 204 ? null : res.json();
}

function toast(msg, kind = "") {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast show " + kind;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.className = "toast " + kind), 2400);
}

function ago(ts) {
  if (!ts) return "";
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 90) return Math.round(s) + "s ago";
  if (s < 5400) return Math.round(s / 60) + "m ago";
  if (s < 129600) return Math.round(s / 3600) + "h ago";
  return Math.round(s / 86400) + "d ago";
}

// Known names = Frigate's enrolled people + our labelled people, deduped
// case-insensitively, keeping the canonical (Frigate-preferred) casing.
function knownNames() {
  const seen = new Map();
  (STATE.frigate_persons || []).forEach((n) => seen.set(n.toLowerCase(), n));
  (STATE.persons || []).forEach((p) => { if (!seen.has(p.name.toLowerCase())) seen.set(p.name.toLowerCase(), p.name); });
  return [...seen.values()].sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()));
}
// Fold case + '-'/'_' so a name matches its Frigate-mangled form. Frigate
// rewrites '-' to '_' in train-crop labels, so a crop for "Jan-Peter" reaches us
// labelled "Jan_Peter"; this lets it canonicalize back onto the real person.
function labelKey(s) { return String(s ?? "").trim().toLowerCase().replace(/-/g, "_"); }
function canonicalName(name) {
  if (!name) return "";
  const names = knownNames();
  return names.find((n) => n.toLowerCase() === name.toLowerCase())
    || names.find((n) => labelKey(n) === labelKey(name))
    || name;
}

// ---------------------------------------------------------------- state
async function loadState() {
  try {
    STATE = await api("/api/state");
  } catch (e) {
    setStatus(null, e.message);
    return;
  }
  $("#badge-review").textContent = STATE.counts.review || 0;
  $("#badge-people").textContent = knownNames().length;
  $("#badge-filtered").textContent = (STATE.counts.auto_rejected || 0) + (STATE.counts.rejected || 0);
  setStatus(STATE.frigate, STATE.ingest);
  setHomeLink(STATE.frigate && STATE.frigate.ui_url);
  applySettings(STATE.settings);
}

// "Back to Frigate" home button. Use the server-provided UI URL if set, else
// derive Frigate's standard front door from the host we're reached on
// (https://<host>:8971/). The companion is plain http on :8975; Frigate is https.
function frigateUiUrl(serverUrl) {
  return (serverUrl && String(serverUrl)) || ("https://" + location.hostname + ":8971/");
}
function setHomeLink(serverUrl) {
  const a = $("#home-link");
  if (a) a.href = frigateUiUrl(serverUrl);
}

function setStatus(frigate, ingest) {
  const el = $("#status");
  if (typeof ingest === "string") { el.innerHTML = `<span class="err">●</span> ${escapeHtml(ingest)}`; return; }
  const reachable = frigate && frigate.reachable;
  const err = ingest && ingest.last_error;
  if (!reachable) { el.innerHTML = `<span class="err">●</span> Frigate unreachable${err ? " · " + escapeHtml(err) : ""}`; return; }
  const v = `Frigate ${escapeHtml(frigate.version)}`;
  el.innerHTML = err
    ? `<span class="err">●</span> ${v} · <span class="err">ingest: ${escapeHtml(err)}</span>`
    : `<span class="ok">●</span> ${v} · ingested ${ingest ? ingest.ingested : 0}`;
}

// ---------------------------------------------------------------- combobox
// A small filter-as-you-type combobox that lets you pick a known person or
// create a new one. Clicking an option (or Enter) calls onSelect(name).
function comboBox(value, onSelect) {
  const wrap = el("div", "combo");
  wrap.innerHTML = `<input class="combo-input" placeholder="Pick a person…" autocomplete="off" value="${escapeAttr(value || "")}" /><span class="combo-chev" aria-hidden="true">&#9662;</span><div class="combo-menu hidden"></div>`;
  const input = $(".combo-input", wrap);
  const menu = $(".combo-menu", wrap);
  const chev = $(".combo-chev", wrap);
  let items = [], hi = -1, open = false;

  // Build options. On open (empty filter) show ALL known people with the current
  // value highlighted; while typing, filter. This is the standard combobox: a
  // pre-filled value never hides the other choices.
  function build(filter) {
    const q = (filter || "").trim().toLowerCase();
    const names = knownNames();
    const matches = q ? names.filter((n) => n.toLowerCase().includes(q)) : names.slice();
    const exact = q && matches.some((n) => n.toLowerCase() === q);
    items = matches.map((n) => ({ name: n, create: false }));
    if (q && !exact) items.push({ name: filter.trim(), create: true });
    const cur = input.value.trim().toLowerCase();
    const match = items.findIndex((it) => it.name.toLowerCase() === cur);
    hi = items.length ? (match >= 0 ? match : 0) : -1;
    menu.innerHTML = items.length
      ? items.map((it, i) =>
          `<div class="combo-opt${i === hi ? " hi" : ""}" role="option" data-i="${i}">${it.create ? `<span class="create">New</span> ` : ""}${escapeHtml(it.name)}</div>`).join("")
      : `<div class="combo-empty">Type a name to create…</div>`;
  }
  // Standard dropdown: the menu is a normal absolutely-positioned element placed
  // directly below the input via CSS (top: 100%). No JS positioning -- it can't
  // be clipped because the card's overflow is visible, and it scrolls with the
  // input on page scroll like any in-flow element.
  function show(filter) {
    build(filter);
    open = true;
    menu.classList.remove("hidden");
    wrap.classList.add("open");
    input.setAttribute("aria-expanded", "true");
    const cur = $(".combo-opt.hi", menu);
    if (cur) cur.scrollIntoView({ block: "nearest" });
  }
  function close() {
    open = false;
    menu.classList.add("hidden");
    wrap.classList.remove("open");
    input.setAttribute("aria-expanded", "false");
  }
  const paint = () => {
    $$(".combo-opt", menu).forEach((o, i) => o.classList.toggle("hi", i === hi));
    const cur = $(".combo-opt.hi", menu);
    if (cur) cur.scrollIntoView({ block: "nearest" });
  };
  const choose = (i) => { if (i >= 0 && i < items.length) onSelect(items[i].name); };

  input.addEventListener("focus", () => { input.select(); show(""); });
  input.addEventListener("input", () => show(input.value));
  chev.addEventListener("mousedown", (e) => { e.preventDefault(); if (open) close(); else input.focus(); });
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); if (!open) return show(""); hi = Math.min(hi + 1, items.length - 1); paint(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); if (!open) return show(""); hi = Math.max(hi - 1, 0); paint(); }
    else if (e.key === "Enter") { e.preventDefault(); if (open && hi >= 0) choose(hi); else if (input.value.trim()) onSelect(input.value.trim()); }
    else if (e.key === "Escape") { e.preventDefault(); close(); }
  });
  menu.addEventListener("mousedown", (e) => {  // mousedown beats input blur
    const opt = e.target.closest(".combo-opt");
    if (!opt) return;
    e.preventDefault();
    choose(+opt.dataset.i);
  });
  input.addEventListener("blur", () => setTimeout(close, 150));
  return wrap;
}

// ---------------------------------------------------------------- review
async function loadReview() {
  const grid = $("#review-grid");
  let items = [];
  try { items = await api("/api/review"); } catch (e) { toast(e.message, "err"); return; }
  $("#review-empty").classList.toggle("hidden", items.length > 0);
  const ids = items.map((i) => i.id).join(",");
  if (grid.dataset.ids === ids) return;  // unchanged: don't clobber a half-typed combo
  grid.dataset.ids = ids;
  grid.innerHTML = "";
  items.forEach((it) => grid.appendChild(reviewCard(it)));
}

function reviewCard(it) {
  const card = el("div", "card");
  card.dataset.id = it.id;
  const fr = it.frigate_label && it.frigate_label !== "unknown"
    ? ` · Frigate: ${escapeHtml(canonicalName(it.frigate_label))}` : "";
  const guess = it.suggested || (it.frigate_label && it.frigate_label !== "unknown" ? it.frigate_label : "");
  const prefill = canonicalName(guess);
  const blur = it.blur != null ? ` · Q ${Math.round(it.blur)}` : "";
  card.innerHTML = `
    <div class="thumb">
      <span class="tag">face ${it.det_score ?? "?"}${blur}</span>
      ${zoomBtn(it.snapshot)}
      <img loading="lazy" src="${it.image}" alt="face crop" />
    </div>
    <div class="meta">
      <div>${ago(it.event_ts)}${fr}</div>
      ${it.suggested ? `<div class="suggest">Looks like ${escapeHtml(canonicalName(it.suggested))} (${it.score})</div>` : ""}
    </div>
    <div class="actions">
      <div class="row">
        <button class="btn assign" style="flex:1">Assign</button>
        <button class="btn warn" title="Not a real face to keep">Not a face</button>
        <button class="btn danger icon trash-btn" title="Delete forever (removes the shot)">&#x1f5d1;</button>
      </div>
    </div>`;
  const actions = $(".actions", card);
  const combo = comboBox(prefill, (name) => assign(card, name));
  actions.insertBefore(combo, actions.firstChild);
  const input = $(".combo-input", combo);
  $(".assign", card).onclick = () => assign(card, input.value.trim());
  $(".warn", card).onclick = () => reject(card);
  $(".trash-btn", card).onclick = () => del(card);
  wireZoom(card, it.snapshot);
  return card;
}

async function assign(card, name) {
  if (!name) { toast("Pick or type a name", "err"); $(".combo-input", card).focus(); return; }
  const id = card.dataset.id;
  leave(card);
  try {
    const r = await api(`/api/crops/${id}/assign`, { method: "POST", body: { name } });
    toast(r.frigate_trained ? `Labelled ${r.name} (trained Frigate)` : `Labelled ${r.name} (Frigate write failed)`,
          r.frigate_trained ? "ok" : "err");
  } catch (e) { toast(e.message, "err"); }
  await loadState();
}

async function reject(card) {
  const id = card.dataset.id;
  leave(card);
  try { await api(`/api/crops/${id}/reject`, { method: "POST" }); toast("Marked not a face", "ok"); }
  catch (e) { toast(e.message, "err"); }
  await loadState();
}

async function del(card) {
  const id = card.dataset.id;
  leave(card);
  try { await api(`/api/crops/${id}/delete`, { method: "POST" }); toast("Deleted", "ok"); }
  catch (e) { toast(e.message, "err"); }
  await loadState();
}

function leave(card) {
  card.classList.add("leaving");
  setTimeout(() => {
    card.remove();
    const grid = $("#review-grid");
    grid.dataset.ids = "";
    if (!grid.children.length) $("#review-empty").classList.remove("hidden");
  }, 170);
}

// ---------------------------------------------------------------- filtered
async function loadFiltered() {
  const grid = $("#filtered-grid");
  let items = [];
  try { items = await api("/api/filtered"); } catch (e) { toast(e.message, "err"); return; }
  $("#filtered-empty").classList.toggle("hidden", items.length > 0);
  grid.innerHTML = "";
  items.forEach((it) => {
    const card = el("div", "card");
    card.dataset.id = it.id;
    const why = { no_face: "no clear face", too_blurry: "low quality", matches_reject: "look-alike of junk", not_a_face: "you rejected" }[it.reason] || it.reason;
    card.innerHTML = `
      <div class="thumb"><span class="tag">${escapeHtml(why)}</span>
        ${zoomBtn(it.snapshot)}
        <img loading="lazy" src="${it.image}" /></div>
      <div class="actions"><div class="row">
        <button class="btn neutral" style="flex:1">Send to review</button>
        <button class="btn danger icon trash-btn" title="Delete forever">&#x1f5d1;</button>
      </div></div>`;
    wireZoom(card, it.snapshot);
    $(".neutral", card).onclick = async () => {
      leave2(card);
      try { const r = await api(`/api/crops/${it.id}/undo`, { method: "POST" }); toast(r && r.frigate_desynced ? "Back to review (already changed in Frigate)" : "Sent back to review", "ok"); }
      catch (e) { toast(e.message, "err"); }
      await loadState();
    };
    $(".trash-btn", card).onclick = async () => {
      leave2(card);
      try { await api(`/api/crops/${it.id}/delete`, { method: "POST" }); toast("Deleted", "ok"); }
      catch (e) { toast(e.message, "err"); }
      await loadState();
    };
    grid.appendChild(card);
  });
}
function leave2(card) { card.classList.add("leaving"); setTimeout(() => card.remove(), 170); }

// ---------------------------------------------------------------- people
async function loadPeople() {
  $("#person-detail").classList.add("hidden");
  const box = $("#people-list");
  box.classList.remove("hidden");
  let people = [];
  try { people = await api("/api/people"); } catch (e) { toast(e.message, "err"); return; }
  if (!people.length) { box.innerHTML = `<div class="empty">No people yet. Label some faces in Review.</div>`; return; }
  box.innerHTML = "";
  people.forEach((p) => {
    const row = el("div", "person-row clickable");
    row.innerHTML = `
      <span class="name">${escapeHtml(p.name)}</span>
      <span class="count">${p.count || 0} ${p.count === 1 ? "crop" : "crops"} in Frigate</span>
      <span class="spacer"></span>
      <span class="view">View crops ›</span>
      <button class="btn danger del-person">Delete</button>`;
    row.onclick = (e) => { if (!e.target.closest(".del-person")) showPerson(p.name); };
    $(".del-person", row).onclick = async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete "${p.name}" everywhere, including their training images in Frigate? This cannot be undone.`)) return;
      try {
        const r = await api("/api/people/delete", { method: "POST", body: { name: p.name } });
        toast(r.frigate_deleted ? `Deleted ${p.name}` : `Deleted locally (Frigate: ${escapeHtml(r.frigate_error || "failed")})`, r.frigate_deleted ? "ok" : "err");
      } catch (e) { toast(e.message, "err"); }
      await loadState();
      loadPeople();
    };
    box.appendChild(row);
  });
}

async function showPerson(name) {
  $("#people-list").classList.add("hidden");
  const detail = $("#person-detail");
  detail.classList.remove("hidden");
  detail.innerHTML = `
    <div class="detail-head">
      <button class="btn back">&#8249; People</button>
      <h2>${escapeHtml(name)}</h2>
      <span class="count" id="pd-count">loading…</span>
    </div>
    <p class="hint">Every face crop Frigate has for this person (older enrollments + ones trained here), newest first. Trash any that are wrong or low quality.</p>
    <div class="grid small" id="pd-grid"></div>
    <div class="empty hidden" id="pd-empty">No crops in Frigate for this person.</div>`;
  $(".back", detail).onclick = () => { detail.classList.add("hidden"); $("#people-list").classList.remove("hidden"); };
  let data;
  try { data = await api(`/api/person/${encodeURIComponent(name)}/crops`); }
  catch (e) { toast(e.message, "err"); return; }
  $("#pd-count").textContent = `${data.crops.length} crop(s)`;
  $("#pd-empty").classList.toggle("hidden", data.crops.length > 0);
  const grid = $("#pd-grid");
  grid.innerHTML = "";
  data.crops.forEach((c) => {
    const card = el("div", "card");
    card.innerHTML = `
      <div class="thumb">
        <button class="trash" title="Delete this crop">&#x1f5d1;</button>
        ${zoomBtn(c.snapshot, true)}
        <img loading="lazy" src="${c.image}" alt="crop" />
      </div>`;
    wireZoom(card, c.snapshot);
    $(".trash", card).onclick = async () => {
      leave2(card);
      try {
        await api(`/api/person/${encodeURIComponent(data.name)}/crops/delete`, { method: "POST", body: { filename: c.filename } });
        toast("Crop deleted", "ok");
        $("#pd-count").textContent = `${(parseInt($("#pd-count").textContent) || 1) - 1} crop(s)`;
      } catch (e) { toast(e.message, "err"); }
      await loadState();
    };
    grid.appendChild(card);
  });
}

// ---------------------------------------------------------------- settings
function applySettings(s) {
  if (!s) return;
  const sync = (id, val, fmt) => { const el = $("#" + id); if (el && document.activeElement !== el) { el.value = val; const lbl = $("#v-" + id.split("_")[0]); if (lbl) lbl.textContent = fmt(val); } };
  sync("match_threshold", s.match_threshold, (v) => (+v).toFixed(2));
  sync("reject_threshold", s.reject_threshold, (v) => (+v).toFixed(2));
  sync("blur_threshold", s.blur_threshold || 0, (v) => (+v ? Math.round(+v) : "off"));
  if ($("#auto_reject")) $("#auto_reject").checked = !!s.auto_reject;
  if ($("#auto_label")) $("#auto_label").checked = !!s.auto_label;
}

function wireSettings() {
  const push = async (key, value) => {
    try { STATE.settings = await api("/api/settings", { method: "POST", body: { key, value } }); toast("Saved", "ok"); }
    catch (e) { toast(e.message, "err"); }
  };
  $("#match_threshold").addEventListener("input", (e) => $("#v-match").textContent = (+e.target.value).toFixed(2));
  $("#reject_threshold").addEventListener("input", (e) => $("#v-reject").textContent = (+e.target.value).toFixed(2));
  $("#blur_threshold").addEventListener("input", (e) => $("#v-blur").textContent = (+e.target.value ? Math.round(+e.target.value) : "off"));
  $("#match_threshold").addEventListener("change", (e) => push("match_threshold", +e.target.value));
  $("#reject_threshold").addEventListener("change", (e) => push("reject_threshold", +e.target.value));
  $("#blur_threshold").addEventListener("change", (e) => push("blur_threshold", +e.target.value));
  $("#auto_reject").addEventListener("change", (e) => push("auto_reject", e.target.checked));
  $("#auto_label").addEventListener("change", (e) => push("auto_label", e.target.checked));
}

// ---------------------------------------------------------------- tabs
function showTab(name) {
  activeTab = name;
  $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + name));
  if (name === "review") loadReview();
  if (name === "people") loadPeople();
  if (name === "filtered") loadFiltered();
}

// ---------------------------------------------------------------- zoom / lightbox
// A magnifier that opens the whole-picture snapshot of the event a crop came
// from -- the full camera frame, which often identifies a person the tight face
// crop can't (same idea as Frigate's faces-page magnifier). `url` is null when
// the crop has no event to look up, in which case no button is rendered.
function zoomBtn(url, left) {
  return url ? `<button class="zoom${left ? " left" : ""}" title="View the whole picture">&#x1f50d;</button>` : "";
}
function wireZoom(card, url) {
  if (!url) return;
  const b = $(".zoom", card);
  if (b) b.onclick = (e) => { e.stopPropagation(); openLightbox(url); };
}
function openLightbox(url) {
  let box = $("#lightbox");
  if (!box) {
    box = el("div", "lightbox");
    box.id = "lightbox";
    box.innerHTML = `<button class="lightbox-close" title="Close (Esc)">&times;</button>
      <div class="lightbox-inner"><img alt="full snapshot" />
        <div class="lightbox-msg hidden">Full picture unavailable — snapshots may be off for this camera, or the event has expired.</div></div>`;
    document.body.appendChild(box);
    box.addEventListener("click", (e) => { if (e.target === box || e.target.closest(".lightbox-close")) closeLightbox(); });
  }
  const img = $("img", box), msg = $(".lightbox-msg", box);
  msg.classList.add("hidden");
  img.classList.remove("hidden");
  img.onerror = () => { img.classList.add("hidden"); msg.classList.remove("hidden"); };
  img.src = url;
  box.classList.add("show");
  document.addEventListener("keydown", lightboxKey);
}
function closeLightbox() {
  const box = $("#lightbox");
  if (box) { box.classList.remove("show"); $("img", box).src = ""; }
  document.removeEventListener("keydown", lightboxKey);
}
function lightboxKey(e) { if (e.key === "Escape") closeLightbox(); }

// ---------------------------------------------------------------- helpers
function el(tag, cls) { const e = document.createElement(tag); if (cls) e.className = cls; return e; }
function escapeHtml(s) { return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function escapeAttr(s) { return escapeHtml(s); }

// ---------------------------------------------------------------- boot
function boot() {
  setHomeLink();  // sensible default immediately; refined once /api/state loads
  $$("#tabs button").forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));
  const lo = $("#logout");
  if (lo) lo.onclick = async () => { try { await api("/logout", { method: "POST" }); } catch (_) {} location.href = "/login"; };
  wireSettings();
  loadState().then(() => showTab("review"));
  setInterval(async () => {
    await loadState();
    if (activeTab === "review") loadReview();
  }, 8000);
}
document.addEventListener("DOMContentLoaded", boot);
