// Bottom status bar: system load at a glance.

import { h, setText, icon } from '../core/dom.js';
import { store } from '../core/store.js';
import * as fmt from '../core/fmt.js';
import { meter } from './ui.js';

export function mountStatusBar(root) {
  const cpuM = meter(0, { cls: 'thin', blocks: false });
  const cpuV = h('b');
  const ramV = h('b');
  const gpuHost = h('div', { style: { display: 'contents' } });
  const reqV = h('b');
  const tokV = h('b');
  const apiLed = h('span', { class: 'led' });
  const apiV = h('b');
  root.append(
    h('div', { class: 'cell', 'data-tip': 'CPU utilisation (all cores)' }, 'CPU', cpuM, cpuV),
    h('div', { class: 'cell', 'data-tip': 'System memory in use / installed' }, 'RAM', ramV),
    gpuHost,
    h('div', { class: 'cell', 'data-tip': 'Requests served since start' }, icon('chat'), reqV),
    h('div', { class: 'cell', 'data-tip': 'Tokens processed since start (prompt + generated)' }, 'TOK', tokV),
    h('div', { class: 'grow' }),
    h('div', { class: 'cell', 'data-tip': 'API server state and port' }, apiLed, apiV),
  );
  const gpuCells = new Map();

  function render(s) {
    if (s) {
      cpuM.set(s.cpu.util);
      setText(cpuV, fmt.pct(s.cpu.util));
      setText(ramV, `${fmt.gib(s.mem.used)} / ${fmt.gib(s.mem.total, 0)} GiB`);
      for (const g of s.gpus || []) {
        let c = gpuCells.get(g.id);
        if (!c) {
          const m = meter(0, { cls: 'thin', blocks: false });
          const v = h('b');
          c = { m, v, el: h('div', { class: 'cell', 'data-tip': `${g.name}: GPU utilisation and dedicated memory` }, g.id.toUpperCase(), m, v) };
          gpuCells.set(g.id, c);
          gpuHost.appendChild(c.el);
        }
        c.m.set(g.util ?? 0);
        setText(c.v, `${g.util !== undefined ? fmt.pct(g.util) : 'n/a'} · ${g.vram_used !== undefined ? `${fmt.gib(g.vram_used)} GiB` : '-'}`);
      }
    }
    const t = store.status?.totals;
    if (t) {
      setText(reqV, fmt.num(t.requests));
      setText(tokV, fmt.num((t.prompt_tokens || 0) + (t.completion_tokens || 0)));
    }
    const sv = store.status?.server;
    if (sv) {
      apiLed.className = `led ${!store.connected ? 'err' : sv.api_enabled ? 'ok' : 'warn'}`;
      setText(apiV, `${sv.api_enabled ? 'API' : 'API STOPPED'} :${sv.port}${sv.auth ? ' · key' : ''}${sv.jit ? ' · JIT' : ''}`);
    }
  }
  store.on('metrics', render);
  store.on('status', () => render(store.lastMetrics));
  store.on('ws', () => render(store.lastMetrics));
  let pending = null;
  store.on('request', ({ rec }) => {
    if (rec.t_end && !pending) pending = setTimeout(() => { pending = null; store.refreshStatus(); }, 800);
  });
  setInterval(() => store.refreshStatus(), 10000);
}
