// Application header: brand, global state, API endpoint, clock.

import { h, setText, copyText } from '../core/dom.js';
import { store } from '../core/store.js';
import * as fmt from '../core/fmt.js';
import { api, wr } from '../core/api.js';
import { engineState } from './state.js';
import { btn, toast, confirmBox } from './ui.js';

export function mountHeader(root) {
  const led = h('span', { class: 'led big' });
  const stLabel = h('span', { class: 'st' }, '...');
  const stModel = h('span', { class: 'md ellipsis' }, '');
  const url = h('span', { class: 'url', 'data-tip': 'OpenAI / LM Studio compatible endpoint. Click to copy.' }, '');
  const clock = h('b');
  const up = h('span');

  url.addEventListener('click', async () => {
    if (await copyText(url.textContent)) toast(`Copied ${url.textContent}`, 'ok', 1600);
  });

  const exitBtn = btn('', async () => {
    if (await confirmBox('Exit WinRunner', 'Stop the API server, unload all models and close WinRunner?', 'Exit')) {
      await api.post(wr('/app/exit'));
    }
  }, { icon: 'power', cls: 'icon hidden', tip: 'Exit WinRunner' });

  root.append(
    h('div', { class: 'brand' },
      h('svg', { class: 'mark', viewBox: '0 0 48 48' }, h('use', { href: '#logo' })),
      h('div', { class: 'words' },
        h('div', { class: 'name', html: 'WIN<b>RUNNER</b>' }),
        h('div', { class: 'tag' }, 'Local Inference Server'))),
    h('div', { class: 'hdr-mid' },
      h('div', { class: 'hdr-endpoint' }, h('span', { class: 'dim' }, 'API'), url),
      h('div', { class: 'hdr-state', style: { minWidth: '0' } }, led, stLabel, stModel)),
    h('div', { class: 'hdr-right' },
      h('div', { class: 'hdr-clock' }, clock, h('br'), h('span', { class: 'dim' }, 'up '), up),
      exitBtn),
  );

  let lastKey = '';
  function render() {
    const s = engineState();
    led.className = `led big ${s.led}`;
    setText(stLabel, s.label);
    stLabel.style.color = s.key === 'error' || s.key === 'offline' ? 'var(--err)' : s.key === 'ready' || s.key === 'gen' ? 'var(--ok)' : s.key === 'idle' ? 'var(--tx-1)' : 'var(--acc)';
    setText(stModel, s.detail || '');
    stModel.dataset.tip = s.detail || '';
    const st = store.status;
    if (st?.server) setText(url, st.server.lan?.[0] || st.server.local);
    exitBtn.classList.toggle('hidden', !st?.window);
    if (s.key !== lastKey) {
      document.title = `WinRunner - ${s.label}${s.detail && s.key !== 'idle' ? ` - ${s.detail}` : ''}`;
      lastKey = s.key;
    }
  }

  function tick() {
    setText(clock, fmt.time(Date.now() / 1000));
    const st = store.status;
    if (st) setText(up, fmt.uptime(store.now() - st.started_at));
  }

  for (const ev of ['status', 'instance', 'instance_progress', 'instance_removed', 'request', 'ws', 'hello']) store.on(ev, render);
  setInterval(tick, 1000);
  tick();
  render();
}
