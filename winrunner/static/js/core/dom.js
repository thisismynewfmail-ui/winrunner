// Minimal DOM helpers (no framework, no build step).

// Element.append() renders null/undefined/false as text; skip them instead so optional
// children can be written inline (cond ? node : null) everywhere.
for (const proto of [Element.prototype, DocumentFragment.prototype]) {
  const native = proto.append;
  proto.append = function append(...nodes) {
    return native.apply(this, nodes.flat(Infinity).filter((n) => n !== null && n !== undefined && n !== false));
  };
}

export function h(tag, props, ...children) {
  const el = tag === 'svg' || tag === 'use' || tag === 'path' ? document.createElementNS('http://www.w3.org/2000/svg', tag)
    : document.createElement(tag);
  if (props) {
    for (const [k, v] of Object.entries(props)) {
      if (v === undefined || v === null || v === false) continue;
      if (k === 'class') el.setAttribute('class', Array.isArray(v) ? v.filter(Boolean).join(' ') : v);
      else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
      else if (k === 'dataset') Object.assign(el.dataset, v);
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
      else if (k === 'html') el.innerHTML = v;
      else if (k === 'text') el.textContent = v;
      else if (k in el && typeof v !== 'string' && !(el instanceof SVGElement)) el[k] = v;
      else if (v === true) el.setAttribute(k, '');
      else el.setAttribute(k, v);
    }
  }
  append(el, children);
  return el;
}

export function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }

export function icon(name, cls) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', cls ? `ic ${cls}` : 'ic');
  svg.setAttribute('viewBox', '0 0 16 16');
  svg.setAttribute('aria-hidden', 'true');
  const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  use.setAttribute('href', `#i-${name}`);
  svg.appendChild(use);
  return svg;
}

export function setText(el, text) {
  if (el && el.textContent !== String(text)) el.textContent = text;
}

export function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

export function debounce(fn, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

// Global tooltip handling for [data-tip] elements.
let tipEl = null;
export function initTooltips() {
  tipEl = h('div', { class: 'tipbox hidden' });
  document.body.appendChild(tipEl);
  let current = null;
  let timer = null;
  document.addEventListener('mouseover', (e) => {
    const t = e.target.closest?.('[data-tip]');
    if (t === current) return;
    current = t;
    clearTimeout(timer);
    tipEl.classList.add('hidden');
    if (!t) return;
    timer = setTimeout(() => {
      tipEl.textContent = t.dataset.tip;
      tipEl.classList.remove('hidden');
      const r = t.getBoundingClientRect();
      const w = tipEl.offsetWidth, hgt = tipEl.offsetHeight;
      let x = Math.min(window.innerWidth - w - 8, Math.max(8, r.left));
      let y = r.bottom + 6;
      if (y + hgt > window.innerHeight - 8) y = r.top - hgt - 6;
      tipEl.style.left = `${x}px`;
      tipEl.style.top = `${y}px`;
    }, 450);
  });
  document.addEventListener('mousedown', () => { clearTimeout(timer); tipEl.classList.add('hidden'); });
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const ta = h('textarea', { style: { position: 'fixed', opacity: '0' } });
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch { ok = false; }
    ta.remove();
    return ok;
  }
}
