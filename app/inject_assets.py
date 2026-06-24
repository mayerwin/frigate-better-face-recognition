"""The client script injected into Frigate's Face Library page.

Served at ``/__betterfaces/inject.js`` (same-origin, via Frigate's nginx, so it is
not blocked as mixed content). It adds a button right after Frigate's built-in
"Recent Recognitions" selector that opens this tool in a new tab. The button URL
is computed at runtime from where the browser is reaching Frigate; the server
prepends ``window.__BFR_PORT`` / ``window.__BFR_URL`` to override it.
"""
from __future__ import annotations

# Lucide "scan-face" icon -> visually consistent with Frigate's own lucide icons.
_SCAN_FACE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"'
    ' fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"'
    ' stroke-linejoin="round"><path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/>'
    '<path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/>'
    '<path d="M8 14s1.5 2 4 2 4-2 4-2"/><path d="M9 9h.01"/><path d="M15 9h.01"/></svg>'
)

INJECT_JS = r"""/* frigate-better-face-recognition - injected into Frigate's Face Library.
 * Served same-origin via Frigate's nginx (the frigate-ext protocol) so it is not
 * blocked as mixed content. Adds a button after the "Recent Recognitions"
 * selector that opens this companion in a new tab. Re-inserted across React
 * re-renders + SPA navigation. */
(function () {
  'use strict';
  if (window.__betterFacesLoaded) return;
  window.__betterFacesLoaded = true;

  // Where to open the tool. Default: same host you reach Frigate on, port
  // __BFR_PORT (the companion publishes plain http). __BFR_URL overrides wholesale
  // (e.g. a reverse-proxied https URL). Opening it as a top-level navigation in a
  // new tab sidesteps the https->http mixed-content block (only subresources are
  // blocked, not navigations).
  var TOOL_URL = (window.__BFR_URL && String(window.__BFR_URL)) ||
    ('http://' + location.hostname + ':' + (window.__BFR_PORT || 8975) + '/');
  var LINK_ID = '__bfr_facerec_link';

  function inject() {
    if (!/(^|\/)faces(\/|$)/.test(location.pathname)) return;
    if (document.getElementById(LINK_ID)) return;
    var btns = document.querySelectorAll('button'), anchor = null;
    for (var i = 0; i < btns.length; i++) {
      if (/recent recognition/i.test(btns[i].textContent || '')) { anchor = btns[i]; break; }
    }
    if (!anchor) return;
    var a = document.createElement('a');
    a.id = LINK_ID; a.href = TOOL_URL; a.target = '_blank'; a.rel = 'noopener noreferrer';
    a.title = 'Open Better Face Recognition';
    // Clone the selector's secondary-button look; drop the dropdown-specific
    // justify-between + the smart-capitalize transform so our label renders as-is.
    a.className = (anchor.className || '')
      .replace(/\bjustify-between\b/g, 'justify-center')
      .replace(/\bsmart-capitalize\b/g, '')
      .replace(/\s+/g, ' ').trim();
    // marginRight:auto keeps us grouped with the selector on the left while
    // Frigate's "Add Face" stays pushed to the right edge of the toolbar.
    a.style.marginLeft = '8px'; a.style.marginRight = 'auto';
    a.style.textDecoration = 'none'; a.style.gap = '6px';
    a.innerHTML = '__SCAN_FACE_SVG__';
    var span = document.createElement('span');
    span.textContent = 'Better Face Recognition';
    a.appendChild(span);
    anchor.insertAdjacentElement('afterend', a);
  }

  var t = null;
  var mo = new MutationObserver(function () {
    if (t) return;
    t = setTimeout(function () { t = null; try { inject(); } catch (e) { /* noop */ } }, 150);
  });
  mo.observe(document.documentElement, { childList: true, subtree: true });
  inject();
})();
""".replace("__SCAN_FACE_SVG__", _SCAN_FACE_SVG)
