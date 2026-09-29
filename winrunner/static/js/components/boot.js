// Start-up screen listing real initialisation results.

import { h } from '../core/dom.js';
import { store } from '../core/store.js';
import { api, wr } from '../core/api.js';
import * as fmt from '../core/fmt.js';

export async function runBoot(el, enabled) {
  if (!enabled || document.documentElement.dataset.anim === 'off') { el.classList.add('done'); return; }
  const lines = h('div', { class: 'lines' });
  const bar = h('div', { class: 'progress' }, h('div', { class: 'fill' }));
  el.append(h('div', { class: 'box' },
    h('div', { class: 'logo' }, h('svg', { viewBox: '0 0 48 48' }, h('use', { href: '#logo' })),
      h('div', null, h('div', { class: 'name', html: 'WIN<b>RUNNER</b>' }), h('div', { class: 'tag' }, 'Local Inference Server'))),
    lines, bar, h('div', { class: 'skip' }, 'Click to skip')));
  let skipped = false;
  const finish = () => { el.classList.add('done'); setTimeout(() => el.remove(), 500); };
  el.addEventListener('click', () => { skipped = true; finish(); });
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  let cur = null;
  const add = async (text, cls = '', progress) => {
    if (skipped) return;
    if (cur) cur.classList.remove('cur');
    cur = h('div', { class: `cur ${cls}` }, text);
    lines.appendChild(cur);
    if (progress !== undefined) bar.firstChild.style.width = `${progress}%`;
    await sleep(110);
  };

  await add('Connecting to control service', '', 8);
  const hello = await new Promise((resolve) => {
    if (store.connected) resolve(true);
    const off = store.on('hello', () => { off(); resolve(true); });
    setTimeout(() => resolve(false), 4000);
  });
  if (!hello) { await add('Control service not reachable', 'warn', 100); await sleep(600); finish(); return; }
  const st = store.status;
  await add(`WinRunner ${st.version} · API ${st.server.lan?.[0] || st.server.local}`, 'ok', 25);
  const sys = st.system;
  if (sys) {
    await add(`${sys.cpu} · ${sys.cores_physical}C/${sys.cores_logical}T · ${fmt.bytes(sys.ram_total, 0)} RAM`, '', 40);
  }
  if (st.engine) await add(`llama.cpp ${st.engine.build ? `build ${st.engine.build}` : st.engine.version} · ${st.engine.backend_label} backend · ${st.engine.flag_count} options`, 'ok', 55);
  else await add('No llama.cpp engine installed (Settings > Engine)', 'warn', 55);
  try {
    const hw = await Promise.race([api.get(wr('/hardware')), sleep(2500).then(() => null)]);
    if (hw?.engine_devices?.length) {
      for (const d of hw.engine_devices) await add(`${d.name}: ${d.description} · ${fmt.num(d.free_mib)} / ${fmt.num(d.total_mib)} MiB free`, 'ok', 70);
    } else if (hw) {
      await add(hw.device_error ? `Device query: ${hw.device_error.slice(0, 80)}` : 'No GPU devices reported by the engine', 'warn', 70);
    }
  } catch { /* ignore */ }
  await add(`Model library: ${st.library.models} models (${st.library.vision} with vision)`, '', 85);
  const inst = st.instances?.find((i) => i.state === 'ready');
  await add(inst ? `Model loaded: ${inst.model} · context ${fmt.num(inst.ctx)}` : 'Ready', 'ok', 100);
  await sleep(350);
  finish();
}
