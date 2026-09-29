// MONITOR: GPU / CPU / memory / engine process telemetry and throughput history.

import { h, clear, setText } from '../core/dom.js';
import { api, wr } from '../core/api.js';
import { store } from '../core/store.js';
import * as fmt from '../core/fmt.js';
import { TimeChart, BarChart } from '../core/charts.js';
import { group, seg, stat, empty } from '../components/ui.js';

const GiB = 1024 ** 3;

export function mount(root) {
  let windowSec = 300;
  const charts = [];
  const offs = [];
  const now = () => store.now();
  const mk = (canvas, opt) => { const c = new TimeChart(canvas, { window: windowSec, now, ...opt }); charts.push(c); return c; };

  const range = seg([[60, '1 min'], [300, '5 min'], [900, '15 min'], [1800, '30 min'], [3600, '60 min']], windowSec,
    (v) => { windowSec = Number(v); for (const c of charts) c.setWindow?.(windowSec); }, { small: true });
  root.append(h('div', { class: 'pg-head' }, h('h2', null, 'Monitor'), h('span', { class: 'sub' }, 'Live telemetry'), h('span', { class: 'spacer' }), range));

  const gpuHost = h('div');
  const sysHost = h('div', { class: 'grid2' });
  const procHost = h('div');
  const tpHost = h('div');
  root.append(gpuHost, sysHost, procHost, tpHost);

  // ---- GPUs ---------------------------------------------------------------------------
  const gpuViews = new Map();
  let sysinfo = null;
  let engineDevs = [];
  let devMap = {};

  function gpuView(g) {
    if (gpuViews.has(g.id)) return gpuViews.get(g.id);
    const info = sysinfo?.gpus?.find((x) => x.id === g.id) || {};
    const dev = Object.entries(devMap).find(([, gid]) => gid === g.id)?.[0];
    const ro = {
      util: stat('Utilisation', '-'), vram: stat('VRAM used', '-'), eng: stat('Engine VRAM', '-', 'Dedicated memory used by llama-server on this GPU'),
      edge: stat('Edge temp', '-'), hot: stat('Junction', '-'), mem: stat('Memory temp', '-'), power: stat('Board power', '-'),
      gfx: stat('GFX clock', '-'), mclk: stat('Mem clock', '-'), fan: stat('Fan', '-'),
    };
    const cv = { util: h('canvas', { class: 'chart' }), vram: h('canvas', { class: 'chart' }), temp: h('canvas', { class: 'chart' }),
      power: h('canvas', { class: 'chart' }), clk: h('canvas', { class: 'chart' }) };
    const cell = (cap, canvas) => h('div', { class: 'mon-cell' }, h('div', { class: 'cap' }, cap), canvas);
    const el = group(`${g.id.toUpperCase()} · ${g.name}`, [
      h('div', { class: 'readouts' }, Object.values(ro)),
      h('div', { class: 'mon-gpu-grid' }, cell('Utilisation %', cv.util), cell('VRAM GiB', cv.vram), cell('Temperature °C', cv.temp),
        cell('Power W', cv.power), cell('Clocks MHz', cv.clk)),
    ], { icon: 'chip', sub: [info.vendor, info.driver ? `driver ${info.driver}` : null, info.bus !== undefined && info.bus !== null ? `PCI bus ${info.bus}` : null,
      `${fmt.gib(g.vram_total)} GiB`, dev ? `engine device ${dev}` : null].filter(Boolean).join(' · ') });
    gpuHost.appendChild(el);
    const v = {
      ro,
      util: mk(cv.util, { yMax: 100, series: [{ key: 'u', color: '--c-1', label: 'GPU' }, { key: 'e', color: '--c-2', label: 'engine', fill: false, width: 1 }], fmt: (x) => `${x.toFixed(0)}` }),
      vram: mk(cv.vram, { yMax: g.vram_total / GiB, series: [{ key: 'v', color: '--c-kv', label: 'used' }, { key: 'e', color: '--c-2', label: 'engine', fill: false, width: 1 }], fmt: (x) => x.toFixed(1) }),
      temp: mk(cv.temp, { minMax: 60, series: [{ key: 'edge', color: '--c-1', label: 'edge', fill: false }, { key: 'hot', color: '--c-3', label: 'junction', fill: false }, { key: 'mem', color: '--c-2', label: 'mem', fill: false }], fmt: (x) => x.toFixed(0) }),
      power: mk(cv.power, { minMax: 50, series: [{ key: 'p', color: '--c-3', label: 'W' }], fmt: (x) => x.toFixed(0) }),
      clk: mk(cv.clk, { minMax: 500, series: [{ key: 'g', color: '--c-1', label: 'gfx', fill: false }, { key: 'm', color: '--c-2', label: 'mem', fill: false }], fmt: (x) => x.toFixed(0) }),
    };
    gpuViews.set(g.id, v);
    return v;
  }

  function engineShare(s, gid, key) {
    let tot = 0; let any = false;
    for (const p of Object.values(s.procs || {})) {
      const v = p[key]?.[gid];
      if (v !== undefined) { tot += v; any = true; }
    }
    return any ? tot : null;
  }

  function pushGpu(s, v, g) {
    const eVram = engineShare(s, g.id, 'vram');
    const eUtil = engineShare(s, g.id, 'gpu_util');
    v.util.push(s.t, { u: g.util ?? null, e: eUtil });
    v.vram.push(s.t, { v: g.vram_used !== undefined ? g.vram_used / GiB : null, e: eVram !== null ? eVram / GiB : null });
    v.temp.push(s.t, { edge: g.temp_edge ?? null, hot: g.temp_hotspot ?? null, mem: g.temp_mem ?? null });
    v.power.push(s.t, { p: g.power ?? null });
    v.clk.push(s.t, { g: g.clk_gfx ?? null, m: g.clk_mem ?? null });
    const r = v.ro;
    r.util.set(g.util !== undefined ? fmt.pct(g.util) : 'n/a');
    r.vram.set(g.vram_used !== undefined ? `${fmt.gib(g.vram_used)} GiB` : 'n/a');
    r.eng.set(eVram !== null ? `${fmt.gib(eVram)} GiB` : '-');
    r.edge.set(g.temp_edge !== undefined ? `${g.temp_edge.toFixed(0)} °C` : 'n/a');
    r.hot.set(g.temp_hotspot !== undefined ? `${g.temp_hotspot.toFixed(0)} °C` : 'n/a');
    r.mem.set(g.temp_mem !== undefined ? `${g.temp_mem.toFixed(0)} °C` : 'n/a');
    r.power.set(g.power !== undefined ? `${g.power.toFixed(0)} W` : 'n/a');
    r.gfx.set(g.clk_gfx !== undefined ? `${g.clk_gfx.toFixed(0)} MHz` : 'n/a');
    r.mclk.set(g.clk_mem !== undefined ? `${g.clk_mem.toFixed(0)} MHz` : 'n/a');
    r.fan.set(g.fan_rpm !== undefined ? `${g.fan_rpm.toFixed(0)} rpm` : g.fan_pct !== undefined ? `${g.fan_pct.toFixed(0)} %` : 'n/a');
  }

  // ---- CPU / memory ----------------------------------------------------------------------
  const cpuRo = { util: stat('Utilisation', '-'), freq: stat('Clock', '-'), cores: stat('Cores', '-') };
  const cpuCv = h('canvas', { class: 'chart' });
  const coresEl = h('div', { class: 'cores' });
  const cpuG = group('CPU', [h('div', { class: 'readouts' }, Object.values(cpuRo)), cpuCv, coresEl, h('div', { style: { height: '14px' } })], { icon: 'chip' });
  const cpuChart = mk(cpuCv, { yMax: 100, series: [{ key: 'u', color: '--c-1', label: 'CPU %' }], fmt: (x) => x.toFixed(0) });
  const memRo = { used: stat('RAM used', '-'), avail: stat('Available', '-'), swap: stat('Page file', '-') };
  const memCv = h('canvas', { class: 'chart' });
  const memG = group('Memory', [h('div', { class: 'readouts' }, Object.values(memRo)), memCv], { icon: 'mem' });
  let memChart = null;
  sysHost.append(cpuG, memG);

  // ---- engine processes ----------------------------------------------------------------------
  const procCv = h('canvas', { class: 'chart' });
  const procRo = h('div', { class: 'readouts' });
  const procG = group('Engine process', [procRo, procCv], { icon: 'server', sub: 'llama-server' });
  const procChart = mk(procCv, { minMax: 10, series: [{ key: 'c', color: '--c-1', label: 'CPU %' }, { key: 'r', color: '--c-2', label: 'disk read MB/s', fill: false }], fmt: (x) => x.toFixed(0) });
  procHost.appendChild(procG);

  // ---- throughput ------------------------------------------------------------------------------
  const tpCv = h('canvas', { class: 'chart tall' });
  const tpStats = h('div', { class: 'stats' });
  const tpG = group('Generation speed per request', [tpStats, h('div', { style: { height: '6px' } }), tpCv], { icon: 'bolt', sub: 'tokens per second (last 60 requests)' });
  const tpChart = new BarChart(tpCv, { color: '--c-1', fmt: (v) => `${v.toFixed(1)} t/s` });
  charts.push(tpChart);
  tpHost.appendChild(tpG);

  function renderTp() {
    const list = store.tps.slice(-60);
    tpChart.setData(list.map((x) => ({ a: x.tg })));
    const tg = list.map((x) => x.tg).filter(Boolean);
    const pp = list.map((x) => x.pp).filter(Boolean);
    const avg = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);
    const t = store.status?.totals || {};
    clear(tpStats);
    tpStats.append(stat('Requests', fmt.num(t.requests)), stat('Prompt tokens', fmt.num(t.prompt_tokens)), stat('Generated', fmt.num(t.completion_tokens)),
      stat('Avg gen', `${fmt.tps(avg(tg))} t/s`), stat('Best gen', `${fmt.tps(tg.length ? Math.max(...tg) : null)} t/s`), stat('Avg prompt', `${fmt.tps(avg(pp))} t/s`),
      stat('Errors', fmt.num(t.errors)));
  }

  function onSample(s, bulk = false) {
    const gl = s.gpus || [];
    if (!gl.length && !gpuHost.querySelector('.empty')) {
      gpuHost.appendChild(group('GPU', empty('No GPU telemetry', 'No supported GPU counters were found. On Windows, GPU data comes from the performance counters (VRAM, utilisation) and the AMD driver (temperatures, clocks, power).'), { icon: 'chip' }));
    }
    for (const g of gl) pushGpu(s, gpuView(g), g);
    cpuChart.push(s.t, { u: s.cpu.util });
    if (!memChart) memChart = mk(memCv, { yMax: s.mem.total / GiB, series: [{ key: 'u', color: '--c-2', label: 'RAM GiB' }, { key: 's', color: '--c-3', label: 'swap', fill: false }], fmt: (x) => x.toFixed(0) });
    memChart.push(s.t, { u: s.mem.used / GiB, s: s.mem.swap_used / GiB });
    const procs = Object.values(s.procs || {});
    procChart.push(s.t, { c: procs.reduce((a, p) => a + (p.cpu || 0), 0), r: procs.reduce((a, p) => a + (p.read_rate || 0), 0) / 1e6 });
    if (bulk) return;
    cpuRo.util.set(fmt.pct(s.cpu.util));
    cpuRo.freq.set(s.cpu.freq_mhz ? `${fmt.num(s.cpu.freq_mhz)} MHz` : 'n/a');
    cpuRo.cores.set(`${s.cpu.per_core.length} threads`);
    if (coresEl.children.length !== s.cpu.per_core.length) {
      clear(coresEl);
      s.cpu.per_core.forEach((_, i) => coresEl.appendChild(h('div', { class: 'core', 'data-tip': `Thread ${i}` }, h('i'), h('span', null, i))));
    }
    s.cpu.per_core.forEach((v, i) => { coresEl.children[i].firstChild.style.height = `${v}%`; });
    memRo.used.set(`${fmt.gib(s.mem.used)} / ${fmt.gib(s.mem.total, 0)} GiB`);
    memRo.avail.set(`${fmt.gib(s.mem.available)} GiB`);
    memRo.swap.set(`${fmt.gib(s.mem.swap_used)} GiB`);
    clear(procRo);
    if (!procs.length) procRo.appendChild(h('div', { class: 'dim' }, 'No engine process running.'));
    for (const [pid, p] of Object.entries(s.procs || {})) {
      procRo.append(stat(`${p.label} (pid ${pid})`, `${p.cpu}% CPU`), stat('Resident', fmt.bytes(p.rss)), stat('Private', fmt.bytes(p.private)),
        stat('Threads', p.threads), stat('Disk read', `${fmt.bytes(p.read_rate || 0)}/s`));
    }
  }

  (async () => {
    try {
      const hw = await api.get(wr('/hardware'));
      sysinfo = hw.system; engineDevs = hw.engine_devices; devMap = hw.device_map || {};
    } catch { /* ignore */ }
    for (const s of store.metrics.slice(-3600)) onSample(s, true);
    if (store.lastMetrics) onSample(store.lastMetrics);
    renderTp();
  })();
  offs.push(store.on('metrics', (s) => onSample(s)));
  offs.push(store.on('request', ({ rec }) => { if (rec.t_end) renderTp(); }));
  offs.push(store.on('status', renderTp));
  return () => { offs.forEach((f) => f()); charts.forEach((c) => c.destroy()); };
}
