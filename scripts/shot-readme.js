// Capture anonymized README screenshots of the better-face-recognition UI.
//
// Before each screenshot it rewrites the page in-browser so no real data is
// published: every person's name is replaced with a generic pseudonym (the real
// names are discovered at runtime from /api/state and never stored in this file),
// and every face image is swapped for a silhouette placeholder. Writes
// bfr-review.png / bfr-people.png / bfr-filtered.png / bfr-settings.png to the
// current directory; copy the ones you want into docs/screenshots/.
//
// Usage:
//   npm i playwright && npx playwright install chromium
//   BFR_SESSION=<cookie> node shot-readme.js http://<host>:8975
//   # ...or put the cookie in a .pwtok file in the working directory.
// Mint a session cookie inside the container:
//   docker exec <container> python3 -c \
//     "from app import auth; print(auth.make_token(auth.load_secret('/app/data')))"

const { chromium } = require('playwright');
const fs = require('fs');

// Generic placeholder names (NOT real users), cycled across whatever people exist.
const POOL = ['Alex', 'Sam', 'Riley', 'Jordan', 'Casey', 'Morgan', 'Taylor', 'Jamie', 'Quinn', 'Avery'];
const TINTS = ['#3a4a63', '#4a3a52', '#3a5247', '#524a3a', '#43395a', '#3a5258', '#4d4040', '#3f4a39'];

function anonymize({ names, tints }) {
  // stop the periodic poll so our edits aren't reverted before the screenshot
  const hi = setInterval(() => {}, 1e6);
  for (let i = 0; i <= hi; i++) clearInterval(i);

  const ph = (i) => {
    const t = tints[i % tints.length];
    const svg =
      "<svg xmlns='http://www.w3.org/2000/svg' width='240' height='240'>" +
      "<rect width='240' height='240' fill='" + t + "'/>" +
      "<circle cx='120' cy='94' r='46' fill='#ffffff' opacity='0.45'/>" +
      "<path d='M40 224 Q40 150 120 150 Q200 150 200 224 Z' fill='#ffffff' opacity='0.45'/>" +
      "</svg>";
    return 'data:image/svg+xml;utf8,' + encodeURIComponent(svg);
  };

  let i = 0;
  document.querySelectorAll('img').forEach((img) => {
    if ((img.getAttribute('src') || '').includes('/api/')) { img.removeAttribute('srcset'); img.src = ph(i++); }
  });
  document.querySelectorAll('*').forEach((el) => {
    const bg = el.style && el.style.backgroundImage;
    if (bg && bg.includes('/api/')) el.style.backgroundImage = 'url("' + ph(i++) + '")';
  });

  const keys = Object.keys(names);
  if (!keys.length) return;
  const esc = (k) => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const rx = new RegExp('\\b(' + keys.map(esc).join('|') + ')\\b', 'gi');
  const repl = (txt) => txt.replace(rx, (m) => {
    const v = names[m.toLowerCase()] || m;
    return m[0] === m[0].toUpperCase() ? v : v.toLowerCase();
  });
  const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (w.nextNode()) nodes.push(w.currentNode);
  nodes.forEach((n) => { n.nodeValue = repl(n.nodeValue); });
  document.querySelectorAll('input').forEach((inp) => {
    if (inp.value) inp.value = repl(inp.value);
    if (inp.placeholder) inp.placeholder = repl(inp.placeholder);
  });
  document.querySelectorAll('[title]').forEach((el) => { el.title = repl(el.getAttribute('title')); });
  document.querySelectorAll('option').forEach((o) => { o.textContent = repl(o.textContent); });
}

(async () => {
  const base = process.argv[2] || 'http://localhost:8975';
  let tok = process.env.BFR_SESSION || '';
  if (!tok) { try { tok = fs.readFileSync('.pwtok', 'utf8').trim(); } catch (_) {} }

  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 860 }, deviceScaleFactor: 2 });
  if (tok) {
    const u = new URL(base);
    await ctx.addCookies([{ name: 'bfr_session', value: tok, domain: u.hostname, path: '/' }]);
  }
  const page = await ctx.newPage();
  await page.goto(base, { waitUntil: 'networkidle', timeout: 30000 });

  // Discover the real names at runtime and map each to a generic pseudonym, so
  // this script never has to contain anyone's actual name.
  const real = await page.evaluate(async () => {
    try {
      const d = await fetch('/api/state').then((r) => r.json());
      return [...new Set([...(d.persons || []).map((p) => p.name), ...(d.frigate_persons || [])])];
    } catch (_) { return []; }
  });
  const names = {};
  real.forEach((n, i) => { if (n) names[String(n).toLowerCase()] = POOL[i % POOL.length]; });

  async function shot(tab, file, sel) {
    if (tab) await page.click(`button[data-tab="${tab}"]`);
    if (sel) await page.waitForSelector(sel, { timeout: 8000 }).catch(() => {});
    await page.waitForTimeout(1200);
    await page.evaluate(anonymize, { names, tints: TINTS });
    await page.waitForTimeout(400);
    await page.screenshot({ path: file });
    console.log('wrote', file);
  }

  await shot(null, 'bfr-review.png', '#review-grid .card');
  await shot('people', 'bfr-people.png', '#people-list');
  await shot('filtered', 'bfr-filtered.png', '#filtered-grid .card');
  await shot('settings', 'bfr-settings.png', '.settings');
  await browser.close();
})();
