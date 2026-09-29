// Reusable widgets: groups, form controls, modals, toasts.

import { h, icon, copyText, clear } from '../core/dom.js';

export function group(title, body, opts = {}) {
  const tools = opts.tools ? h('div', { class: 'gt-tools' }, opts.tools) : null;
  const sub = opts.sub ? h('span', { class: 'sub' }, opts.sub) : null;
  const gt = h('div', { class: 'gt' }, opts.icon ? icon(opts.icon) : null, h('span', { class: 'ttl' }, title), sub, tools);
  const gb = h('div', { class: 'gb' }, body);
  const el = h('section', { class: ['group', opts.cls, opts.collapsible && 'collapsible', opts.collapsed && 'collapsed'] }, gt, gb);
  if (opts.collapsible) {
    gt.addEventListener('click', (e) => {
      if (e.target.closest('.gt-tools')) return;
      el.classList.toggle('collapsed');
      opts.onToggle?.(!el.classList.contains('collapsed'));
    });
  }
  el.body = gb;
  el.title = gt.querySelector('.ttl');
  el.sub = sub;
  return el;
}

export function btn(label, onClick, opts = {}) {
  const b = h('button', { class: ['btn', opts.cls], type: 'button', title: opts.title, 'data-tip': opts.tip, disabled: opts.disabled },
    opts.icon ? icon(opts.icon) : null, label ? h('span', null, label) : null);
  if (onClick) b.addEventListener('click', (e) => onClick(e, b));
  return b;
}

export function seg(options, value, onChange, opts = {}) {
  const el = h('div', { class: ['seg', opts.small && 'small'], role: 'radiogroup' });
  const set = (v) => {
    el.value = v;
    for (const b of el.children) b.classList.toggle('on', b.dataset.v === String(v));
  };
  for (const o of options) {
    const [v, label, tip] = Array.isArray(o) ? o : [o, o];
    el.appendChild(h('button', { type: 'button', 'data-v': String(v), 'data-tip': tip, onclick: () => { set(v); onChange?.(v); } }, label));
  }
  set(value);
  el.set = set;
  return el;
}

export function toggle(label, checked, onChange, opts = {}) {
  const input = h('input', { type: 'checkbox', checked: !!checked, disabled: opts.disabled });
  input.addEventListener('change', () => onChange?.(input.checked));
  const el = h('label', { class: 'check', 'data-tip': opts.tip }, input, h('span', null, label));
  el.input = input;
  el.set = (v) => { input.checked = !!v; };
  return el;
}

export function select(options, value, onChange, opts = {}) {
  const el = h('select', { class: ['field', opts.cls], style: opts.style });
  for (const o of options) {
    const [v, label] = Array.isArray(o) ? o : [o, o];
    el.appendChild(h('option', { value: String(v) }, label));
  }
  el.value = String(value ?? '');
  el.addEventListener('change', () => onChange?.(el.value));
  return el;
}

export function input(value, onChange, opts = {}) {
  const el = h('input', { class: ['field', opts.cls], type: opts.type || 'text', placeholder: opts.placeholder, style: opts.style,
    min: opts.min, max: opts.max, step: opts.step, spellcheck: 'false', autocomplete: 'off' });
  el.value = value ?? '';
  el.addEventListener(opts.live ? 'input' : 'change', () => onChange?.(opts.type === 'number' ? Number(el.value) : el.value));
  if (opts.onEnter) el.addEventListener('keydown', (e) => { if (e.key === 'Enter') opts.onEnter(el.value); });
  return el;
}

/** Slider + numeric field kept in sync. `scale: 'log2'` for powers of two. */
export function slideNum(value, { min, max, step = 1, log2 = false, width = 90 } = {}, onChange) {
  const toSlider = (v) => (log2 ? Math.log2(Math.max(1, v)) : v);
  const fromSlider = (s) => (log2 ? Math.round(2 ** s) : Number(s));
  const s = h('input', { type: 'range', class: 'slider grow', min: log2 ? Math.log2(min) : min, max: log2 ? Math.log2(max) : max,
    step: log2 ? 0.05 : step });
  const n = h('input', { class: 'field num', type: 'number', min, max, step, style: { width: `${width}px` } });
  const set = (v) => { n.value = v; s.value = toSlider(v); };
  set(value);
  s.addEventListener('input', () => {
    let v = fromSlider(s.value);
    if (log2) { const p = 2 ** Math.round(Math.log2(v)); if (Math.abs(p - v) / p < 0.06) v = p; else v = Math.round(v / 256) * 256; }
    n.value = v;
  });
  s.addEventListener('change', () => onChange?.(Number(n.value)));
  n.addEventListener('change', () => { let v = Number(n.value); v = Math.min(max, Math.max(min, v)); set(v); onChange?.(v); });
  const el = h('div', { class: 'row grow' }, s, n);
  el.set = set;
  el.slider = s;
  el.num = n;
  return el;
}

export function formRow(form, label, control, hint, opts = {}) {
  form.appendChild(h('label', { class: opts.labelCls, 'data-tip': opts.tip }, label, opts.tag || null));
  form.appendChild(h('div', { class: 'ctl' }, control));
  if (hint) form.appendChild(h('div', { class: 'hint' }, hint));
}

export function meter(value, opts = {}) {
  const fill = h('div', { class: ['fill', opts.level] });
  const lbl = opts.label !== undefined ? h('div', { class: 'lbl' }) : null;
  const el = h('div', { class: ['meter', opts.cls, opts.blocks !== false && 'seg-blocks'] }, fill, lbl);
  el.set = (v, text, level) => {
    fill.style.width = `${Math.max(0, Math.min(100, v || 0))}%`;
    if (level !== undefined) fill.className = `fill ${level || ''}`;
    if (lbl && text !== undefined) lbl.textContent = text;
  };
  el.set(value, opts.label, opts.level);
  return el;
}

export function copyBtn(getText, opts = {}) {
  return btn(opts.label || '', async (e, b) => {
    const ok = await copyText(typeof getText === 'function' ? getText() : getText);
    toast(ok ? 'Copied to clipboard' : 'Copy failed', ok ? 'ok' : 'err', 1400);
  }, { icon: 'copy', cls: ['icon', opts.small !== false && 'small'], tip: opts.tip || 'Copy' });
}

// ----- toast ---------------------------------------------------------------------
let toastHost = null;
export function toast(text, level = 'info', ttl = 3500, title) {
  if (!toastHost) { toastHost = h('div', { id: 'toasts' }); document.body.appendChild(toastHost); }
  const el = h('div', { class: ['toast', level] }, title ? h('div', { class: 't' }, title) : null, h('div', null, text));
  toastHost.appendChild(el);
  const close = () => { el.classList.add('out'); setTimeout(() => el.remove(), 300); };
  el.addEventListener('click', close);
  setTimeout(close, ttl);
  while (toastHost.children.length > 5) toastHost.firstChild.remove();
}

// ----- modal ---------------------------------------------------------------------
export function modal(title, body, { buttons = [], wide = false, onClose } = {}) {
  const foot = buttons.length ? h('div', { class: 'modal-foot' }) : null;
  const back = h('div', { class: 'modal-back' });
  const close = () => { back.remove(); document.removeEventListener('keydown', onKey); onClose?.(); };
  const onKey = (e) => { if (e.key === 'Escape') close(); };
  const win = h('div', { class: ['modal', wide && 'wide'], role: 'dialog' },
    h('div', { class: 'modal-title' }, h('span', null, title), btn('', close, { icon: 'x', cls: 'icon small x', tip: 'Close' })),
    h('div', { class: 'modal-body' }, body), foot);
  for (const b of buttons) {
    foot.appendChild(btn(b.label, async () => { const r = await b.onClick?.(); if (r !== false) close(); }, { cls: b.cls }));
  }
  back.appendChild(win);
  back.addEventListener('mousedown', (e) => { if (e.target === back) close(); });
  document.addEventListener('keydown', onKey);
  document.body.appendChild(back);
  return { close, win, body: win.querySelector('.modal-body') };
}

export function confirmBox(title, text, okLabel = 'OK') {
  return new Promise((resolve) => {
    let done = false;
    modal(title, h('div', { style: { maxWidth: '520px' } }, text), {
      buttons: [
        { label: okLabel, cls: 'primary', onClick: () => { done = true; resolve(true); } },
        { label: 'Cancel', onClick: () => { done = true; resolve(false); } },
      ],
      onClose: () => { if (!done) resolve(false); },
    });
  });
}

export function kv(pairs, cls = 'kv') {
  const dl = h('dl', { class: cls });
  for (const [k, v, tip] of pairs) {
    if (v === undefined) continue;
    dl.appendChild(h('dt', { 'data-tip': tip }, k));
    dl.appendChild(h('dd', null, v));
  }
  return dl;
}

export function stat(k, v, tip) {
  const vEl = h('div', { class: 'v' }, v);
  const el = h('div', { class: 'stat', 'data-tip': tip }, h('div', { class: 'k' }, k), vEl);
  el.set = (x) => { clear(vEl); vEl.append(x instanceof Node ? x : document.createTextNode(String(x))); };
  return el;
}

export function empty(title, text, action) {
  return h('div', { class: 'empty' }, h('div', { class: 'big' }, title), text ? h('div', null, text) : null,
    action ? h('div', { style: { marginTop: '10px' } }, action) : null);
}
