// LOGS: engine (llama-server) and application logs with filtering and search.

import { h, clear, escapeHtml } from '../core/dom.js';
import { api, wr } from '../core/api.js';
import { store } from '../core/store.js';
import * as fmt from '../core/fmt.js';
import { btn, select, input, toggle, toast } from '../components/ui.js';
import { takeNavParams } from '../app.js';

const MAX_LINES = 4000;

export function mount(root) {
  const nav = takeNavParams();
  const offs = [];
  let source = nav?.iid || '';
  let levels = { info: true, warn: true, error: true, debug: false };
  let query = '';
  let follow = true;
  let paused = [];

  const srcSel = select([], source, (v) => { source = v; reload(); }, { style: { minWidth: '260px' } });
  const q = input('', (v) => { query = v.toLowerCase(); rerender(); }, { placeholder: 'Search...', live: true, cls: 'grow' });
  const lvl = (k, label) => toggle(label, levels[k], (v) => { levels[k] = v; rerender(); });
  const followT = toggle('Follow', true, (v) => { follow = v; if (v) { flushPaused(); view.scrollTop = view.scrollHeight; } });
  const count = h('span', { class: 'dim' });
  const view = h('div', { class: 'logview' });
  root.append(
    h('div', { class: 'pg-head' }, h('h2', null, 'Logs'), count),
    h('div', { class: 'toolbar' }, srcSel, lvl('error', 'Errors'), lvl('warn', 'Warnings'), lvl('info', 'Info'), lvl('debug', 'Debug'),
      q, followT, btn('Clear view', () => { lines = []; rerender(); }, { icon: 'trash', cls: 'small' }),
      btn('Download', () => { window.location.href = wr(`/logs/download${source ? `?iid=${encodeURIComponent(source)}` : ''}`); }, { icon: 'download', cls: 'small' })),
    view);

  let lines = [];
  function fillSources() {
    clear(srcSel);
    srcSel.appendChild(h('option', { value: '' }, 'Application (WinRunner)'));
    for (const i of store.instances.values()) srcSel.appendChild(h('option', { value: i.id }, `Engine · ${i.model} (${i.id})`));
    if (source && !store.instances.has(source)) source = '';
    srcSel.value = source;
  }

  function lineEl(x) {
    const text = escapeHtml(x.text);
    const hl = query ? text.replace(new RegExp(query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi'), (m) => `<mark>${m}</mark>`) : text;
    const el = h('div', { class: `logline ${x.level} ${x.source === 'winrunner' ? 'winrunner' : ''}` },
      h('span', { class: 't' }, fmt.time(x.t)), h('span', { class: 'l' }, x.level));
    const xx = h('span', { class: 'x' });
    xx.innerHTML = hl;
    el.appendChild(xx);
    return el;
  }
  const visible = (x) => (levels[x.level] ?? (x.level === 'trace' ? levels.debug : true)) && (!query || x.text.toLowerCase().includes(query));

  function rerender() {
    clear(view);
    const frag = document.createDocumentFragment();
    let n = 0;
    for (const x of lines) if (visible(x)) { frag.appendChild(lineEl(x)); n++; }
    view.appendChild(frag);
    count.textContent = `${fmt.num(n)} of ${fmt.num(lines.length)} lines`;
    if (follow) view.scrollTop = view.scrollHeight;
  }

  function add(x) {
    lines.push(x);
    if (lines.length > MAX_LINES) lines.splice(0, lines.length - MAX_LINES);
    if (!visible(x)) return;
    if (!follow) { paused.push(x); return; }
    view.appendChild(lineEl(x));
    while (view.childNodes.length > MAX_LINES) view.removeChild(view.firstChild);
    view.scrollTop = view.scrollHeight;
  }
  function flushPaused() { for (const x of paused) view.appendChild(lineEl(x)); paused = []; }

  async function reload() {
    try {
      if (source) {
        const d = await api.get(wr(`/instances/log?iid=${encodeURIComponent(source)}&limit=${MAX_LINES}`));
        lines = d.lines;
      } else {
        const d = await api.get(wr(`/logs/app?limit=${MAX_LINES}`));
        lines = d.lines;
      }
      rerender();
    } catch (e) { toast(e.message, 'err'); }
  }

  offs.push(store.on('englog', (ev) => { if (ev.iid === source) add(ev); }));
  offs.push(store.on('applog', (ev) => { if (!source) add(ev); }));
  offs.push(store.on('instance', fillSources));
  offs.push(store.on('instance_removed', fillSources));
  fillSources();
  if (!source) {
    const p = store.primary;
    if (p) { source = p.id; srcSel.value = source; }
  }
  reload();
  return () => offs.forEach((f) => f());
}
