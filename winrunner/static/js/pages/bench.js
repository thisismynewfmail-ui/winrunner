// BENCHMARK: llama-bench runs with the model's load configuration.

import { h, clear, setText } from '../core/dom.js';
import { api, wr } from '../core/api.js';
import { store } from '../core/store.js';
import * as fmt from '../core/fmt.js';
import { group, btn, select, input, toggle, toast, kv, empty, confirmBox } from '../components/ui.js';

export function mount(root) {
  const offs = [];
  let models = [];
  const modelSel = select([], '', null, { style: { minWidth: '300px' } });
  const pp = input('512', null, { cls: 'mono', style: { width: '140px' } });
  const tg = input('128', null, { cls: 'mono', style: { width: '140px' } });
  const depth = input('0', null, { cls: 'mono', style: { width: '140px' } });
  const reps = select([[1, '1'], [3, '3'], [5, '5']], 3);
  const unload = toggle('Unload loaded models first (frees VRAM)', true);
  const runBtn = btn('Run benchmark', () => run(), { icon: 'play', cls: 'primary big' });
  const cancelBtn = btn('Cancel', async () => { await api.post(wr('/bench/cancel')); }, { icon: 'stop', cls: 'big hidden' });
  const logBox = h('pre', { class: 'code bench-log' }, '');
  const status = h('div', { class: 'dim' });
  const results = h('div');
  const hist = h('div');

  const form = h('div', { class: 'form' },
    h('label', null, 'Model'), h('div', { class: 'ctl' }, modelSel),
    h('label', { 'data-tip': 'Prompt lengths (comma separated) for prompt processing speed' }, 'Prompt tokens (pp)'), h('div', { class: 'ctl' }, pp, h('span', { class: 'dim' }, 'e.g. 512,2048')),
    h('label', { 'data-tip': 'Tokens generated for text generation speed' }, 'Generated tokens (tg)'), h('div', { class: 'ctl' }, tg, h('span', { class: 'dim' }, 'e.g. 128')),
    h('label', { 'data-tip': 'Context already filled before measuring (tests speed at long context)' }, 'Context depth'), h('div', { class: 'ctl' }, depth, h('span', { class: 'dim' }, 'e.g. 0,16384')),
    h('label', null, 'Repetitions'), h('div', { class: 'ctl' }, reps),
    h('div', { class: 'full' }, unload),
    h('div', { class: 'full dim' }, 'Uses the model\'s saved load configuration (GPU layers, tensor split, flash attention, KV cache type, batch sizes) so results match what the server will do.'));

  root.append(h('div', { class: 'pg-head' }, h('h2', null, 'Benchmark'), h('span', { class: 'sub' }, 'llama-bench: prompt processing (pp) and generation (tg) throughput')),
    group('Run', [form, h('div', { class: 'row', style: { marginTop: '10px' } }, runBtn, cancelBtn, status)], { icon: 'bench' }),
    group('Output', logBox, { icon: 'log', collapsible: true }),
    group('Latest result', results, { icon: 'bolt' }),
    group('History', hist, { icon: 'list', tools: [btn('Clear', async () => {
      if (await confirmBox('Clear history', 'Delete all benchmark results?', 'Delete')) { await api.del(wr('/bench')); loadHist(); }
    }, { cls: 'small', icon: 'trash' })] }));

  async function loadModels() {
    const d = await api.get(wr('/library'));
    models = d.models.filter((m) => m.kind === 'llm');
    clear(modelSel);
    for (const m of models) modelSel.appendChild(h('option', { value: m.id }, `${m.id} · ${m.quant} · ${fmt.bytes(m.file_size)}`));
    const loaded = store.primary?.model;
    if (loaded && models.find((m) => m.id === loaded)) modelSel.value = loaded;
  }

  async function run() {
    try {
      clear(logBox);
      const r = await api.post(wr('/bench'), { id: modelSel.value, pp: pp.value.trim() || '512', tg: tg.value.trim() || '128',
        depth: depth.value.trim() || '0', reps: Number(reps.value), unload: unload.input.checked });
      logBox.textContent = `$ ${r.command}\n`;
      busy(true);
    } catch (e) { toast(e.message, 'err', 6000); }
  }

  function busy(b) {
    runBtn.classList.toggle('hidden', b);
    cancelBtn.classList.toggle('hidden', !b);
    setText(status, b ? 'Running... (the GPU will be fully loaded)' : '');
  }

  function table(run) {
    const max = Math.max(...run.results.map((r) => r.avg_ts || 0), 1);
    return h('div', null,
      kv([['Model', `${run.meta.model} (${run.meta.quant})`], ['Engine', run.meta.engine], ['Configuration',
        `ngl ${run.meta.ngl} · FA ${run.meta.fa} · KV ${run.meta.kv} · batch ${run.meta.b}/${run.meta.ub}${run.meta.ts ? ` · split ${run.meta.ts.join(',')}` : ''}`],
      ['Devices', run.results[0]?.gpu_info || run.results[0]?.backends || '-'], ['Time', fmt.dateTime(run.t)]]),
      h('table', { class: 'tbl', style: { marginTop: '8px' } }, h('thead', null, h('tr', null, h('th', null, 'Test'), h('th', { class: 'num' }, 'Tokens/s'),
        h('th', { class: 'num' }, '± stddev'), h('th', { style: { width: '45%' } }))),
      h('tbody', null, run.results.map((r) => h('tr', null, h('td', null, r.test), h('td', { class: 'num', style: { fontWeight: 'bold' } }, fmt.num(r.avg_ts, 2)),
        h('td', { class: 'num dim' }, fmt.num(r.stddev_ts, 2)), h('td', null, h('div', { class: 'bench-bar', style: { width: `${(r.avg_ts / max) * 100}%` } })))))),
      run.error ? h('div', { class: 'note err' }, run.error) : null);
  }

  async function loadHist() {
    const d = await api.get(wr('/bench'));
    busy(!!d.running);
    clear(hist);
    if (!d.history.length) { hist.appendChild(empty('No results yet', 'Run a benchmark to measure this machine.')); return; }
    const tb = h('tbody');
    for (const r of d.history) {
      const ppr = r.results.find((x) => x.n_prompt && !x.n_gen);
      const tgr = r.results.find((x) => x.n_gen && !x.n_prompt);
      tb.appendChild(h('tr', { onclick: () => { clear(results); results.appendChild(table(r)); } },
        h('td', null, fmt.dateTime(r.t)), h('td', null, r.meta.model), h('td', null, r.meta.engine), h('td', null, `FA ${r.meta.fa} · KV ${r.meta.kv} · ub ${r.meta.ub}`),
        h('td', { class: 'num' }, ppr ? fmt.num(ppr.avg_ts, 1) : '-'), h('td', { class: 'num' }, tgr ? fmt.num(tgr.avg_ts, 1) : '-'),
        h('td', { class: r.state === 'done' ? 'ok' : 'err' }, r.state)));
    }
    hist.appendChild(h('div', { class: 'tbl-wrap', style: { maxHeight: '320px' } }, h('table', { class: 'tbl compact' }, h('thead', null, h('tr', null,
      h('th', null, 'Time'), h('th', null, 'Model'), h('th', null, 'Engine'), h('th', null, 'Settings'), h('th', { class: 'num' }, 'pp t/s'),
      h('th', { class: 'num' }, 'tg t/s'), h('th', null, 'State'))), tb)));
    if (!results.children.length && d.history[0]) results.appendChild(table(d.history[0]));
  }

  offs.push(store.on('bench', (ev) => {
    if (ev.state === 'log') { logBox.textContent += `${ev.line}\n`; logBox.scrollTop = logBox.scrollHeight; }
    if (ev.state === 'done' || ev.state === 'error') {
      busy(false);
      if (ev.run) { clear(results); results.appendChild(table(ev.run)); }
      if (ev.error) toast(ev.error, 'err');
      loadHist();
    }
  }));
  loadModels().then(loadHist).catch((e) => toast(e.message, 'err'));
  if (!results.children.length) results.appendChild(empty('No result', ''));
  return () => offs.forEach((f) => f());
}
