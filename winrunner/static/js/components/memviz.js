// Memory allocation visualisations: per-device VRAM bars and the layer map.

import { h, clear } from '../core/dom.js';
import * as fmt from '../core/fmt.js';

export const DEV_COLORS = ['--c-1', '--c-2', '--c-3', '--c-4', '--c-5'];
const SEGMENTS = [
  ['other', 'In use (other apps)', '--c-other'],
  ['weights', 'Model weights', '--c-weights'],
  ['kv', 'KV cache', '--c-kv'],
  ['compute', 'Compute buffers', '--c-compute'],
  ['mmproj', 'Vision projector', '--c-mmproj'],
  ['draft', 'Draft model', '--c-4'],
  ['margin', 'Safety margin', 'margin'],
];

/**
 * Per-device stacked bars.
 * devices: [{name, description, total, free, weights, kv, compute, mmproj, draft, margin, gpu}]
 */
export function memBars() {
  const el = h('div', { class: 'membars' });
  el.update = (devices, { host } = {}) => {
    clear(el);
    for (const d of devices) {
      const total = Math.max(1, d.total || 0);
      const other = Math.max(0, total - (d.free ?? total));
      const vals = { other, weights: d.weights || 0, kv: d.kv || 0, compute: d.compute || 0, mmproj: d.mmproj || 0,
        draft: d.draft || 0, margin: d.margin || 0 };
      const used = Object.values(vals).reduce((a, b) => a + b, 0);
      const over = used > total;
      const bar = h('div', { class: ['membar', over && 'over'] });
      for (const [k, label, color] of SEGMENTS) {
        const v = vals[k];
        if (!v) continue;
        const seg = h('div', { class: ['mseg', `m-${k}`], 'data-tip': `${label}: ${fmt.mib(v)}` });
        seg.style.width = `${Math.min(100, (v / Math.max(total, used)) * 100)}%`;
        if (color !== 'margin') seg.style.background = `var(${color})`;
        bar.appendChild(seg);
      }
      const free = total - used;
      const pctUsed = Math.min(100, ((used - vals.margin) / total) * 100);
      el.appendChild(h('div', { class: 'membar-row' },
        h('div', { class: 'membar-head' },
          h('span', { class: 'swatch', style: { background: `var(${d.color || '--c-1'})` } }),
          h('b', null, d.name), h('span', { class: 'dim ellipsis' }, d.description || ''),
          h('span', { class: 'spacer' }),
          h('span', { class: over ? 'err' : free < 256 ? 'warn' : 'muted' },
            over ? `over by ${fmt.mib(-free)}` : `${fmt.mib(used - vals.margin)} / ${fmt.mib(total)} · ${pctUsed.toFixed(0)}%`)),
        bar,
        h('div', { class: 'membar-legend' },
          SEGMENTS.filter(([k]) => vals[k] > 0).map(([k, label, color]) => h('span', null,
            h('i', { class: `m-${k}`, style: color !== 'margin' ? { background: `var(${color})` } : null }), `${label} ${fmt.mib(vals[k])}`)),
          h('span', null, h('i', { class: 'm-free' }), `Free ${fmt.mib(Math.max(0, free))}`))));
    }
    if (host) {
      el.appendChild(h('div', { class: 'membar-row host' },
        h('div', { class: 'membar-head' }, h('span', { class: 'swatch', style: { background: 'var(--c-other)' } }), h('b', null, 'System RAM'),
          h('span', { class: 'dim' }, 'weights kept on the CPU side, host KV cache and buffers'), h('span', { class: 'spacer' }),
          h('span', { class: 'muted' }, fmt.mib((host.weights_mib || 0) + (host.kv_mib || 0) + (host.compute_mib || 0)))),
        h('div', { class: 'membar-legend' },
          h('span', null, `Weights ${fmt.mib(host.weights_mib || 0)}`), h('span', null, `KV ${fmt.mib(host.kv_mib || 0)}`),
          h('span', null, `Buffers ${fmt.mib(host.compute_mib || 0)}`))));
    }
  };
  return el;
}

/** Devices for memBars from a plan (planner output). */
export function planDevices(plan) {
  return (plan.devices || []).map((d, i) => ({
    name: d.name, description: d.description, total: d.total_mib, free: d.free_mib,
    weights: d.weights_mib + (d.output_mib || 0), kv: d.kv_mib, compute: d.compute_mib, mmproj: d.mmproj_mib,
    draft: d.draft_mib, margin: d.margin_mib, color: DEV_COLORS[i % DEV_COLORS.length],
  }));
}

/** Devices for memBars from a loaded instance (actual engine allocations). */
export function actualDevices(inst, engineDevices) {
  const buf = inst.load?.buffers || {};
  const plan = inst.plan || {};
  const out = [];
  (plan.devices || []).forEach((d, i) => {
    const b = buf[d.name] || {};
    const ed = (engineDevices || []).find((x) => x.name === d.name);
    const total = ed?.total_mib || d.total_mib;
    const self = (b.model || 0) + (b.kv || 0) + (b.compute || 0) + (b.output || 0);
    out.push({
      name: d.name, description: d.description, total,
      free: ed ? ed.free_mib + self : d.free_mib,
      weights: b.model || 0, kv: b.kv || 0, compute: (b.compute || 0) + (b.output || 0),
      mmproj: inst.mmproj ? (inst.load?.mmproj?.est_mib || d.mmproj_mib || 0) : 0,
      color: DEV_COLORS[i % DEV_COLORS.length],
    });
  });
  return out;
}

/** Device index holding each layer's attention and KV cache (-1 = CPU), from a plan and, if known, the load. */
function layerOwners(plan, loadInfo) {
  const n = plan.n_layer;
  const devs = plan.devices || [];
  // WinRunner's own placement is exact (-ts layer counts and --override-tensor): the plan is what gets loaded.
  if (plan.layer_home?.length === n && plan.strategy !== 'engine') return plan.layer_home.slice();
  // llama.cpp --fit placed the layers: derive the split from the engine's actual buffers
  let counts = devs.map((d) => d.layers || 0);
  const buf = loadInfo?.buffers;
  const off = loadInfo?.offload;
  if (buf && off && devs.length > 1) {
    const w = devs.map((d) => (buf[d.name]?.model || 0));
    const tot = w.reduce((a, b) => a + b, 0);
    const gpuLayers = Math.min(n, off.gpu_layers);
    if (tot > 0) {
      counts = w.map((x) => Math.round((x / tot) * gpuLayers));
      counts[counts.length - 1] += gpuLayers - counts.reduce((a, b) => a + b, 0);
    }
  } else if (off && devs.length === 1) {
    counts = [Math.min(n, off.gpu_layers)];
  }
  const owner = new Array(n).fill(-1);
  let il = n - counts.reduce((a, b) => a + b, 0);
  counts.forEach((c, di) => { for (let k = 0; k < c && il < n; k++) owner[il++] = di; });
  return owner;
}

/** Tensor groups of each layer kept in system RAM (manual --n-cpu-moe / --n-cpu-ffn included). */
function layerRamParts(plan) {
  const n = plan.n_layer;
  if (plan.ram_parts?.length === n) return plan.ram_parts;
  const out = new Array(n).fill(null).map(() => []);
  for (let i = 0; i < Math.min(n, plan.n_cpu_moe || 0); i++) out[i] = ['experts'];
  for (let i = 0; i < Math.min(n, plan.n_cpu_ffn || 0); i++) out[i] = ['ffn_up', 'ffn_gate', 'ffn_down'];
  return out;
}

const partLabel = (g) => g.replace(/^ffn_/, '').replace(/_(ch)?exps$/, ' experts').replace(/_/g, ' ');

/**
 * Layer map: one block per transformer layer, coloured by the GPU holding its attention and KV cache.
 * A lower grey band marks feed-forward (or expert) weights kept in system RAM.
 */
export function layerMap() {
  const grid = h('div', { class: 'layermap' });
  const legend = h('div', { class: 'layermap-legend' });
  const el = h('div', { class: 'layermap-wrap' }, grid, legend);
  let sig = '';
  el.update = (plan, { progress = 1, loading = false, loadInfo = null } = {}) => {
    if (!plan || !plan.n_layer) { clear(grid); clear(legend); sig = ''; return; }
    const n = plan.n_layer;
    const devs = plan.devices || [];
    const owner = layerOwners(plan, loadInfo);
    const ram = layerRamParts(plan);
    const lit = loading ? Math.floor(Math.max(0, Math.min(1, (progress - 0.12) / 0.72)) * n) : n;
    const s = `${n}|${owner.join(',')}|${ram.map((x) => x.length).join(',')}|${lit}|${loading}`;
    if (s === sig) return;
    const rebuild = sig.split('|')[0] !== String(n) || !grid.children.length;
    sig = s;
    if (rebuild) {
      clear(grid);
      grid.appendChild(h('div', { class: 'lb emb', 'data-tip': 'Token embeddings (kept in system RAM by the engine)' }, 'E'));
      for (let i = 0; i < n; i++) grid.appendChild(h('div', { class: 'lb' }));
      grid.appendChild(h('div', { class: 'lb out', 'data-tip': 'Output layer' }, 'O'));
    }
    const blocks = grid.children;
    const moe = ram.some((x) => x.some((g) => g.includes('exps') || g === 'experts'));
    for (let i = 0; i < n; i++) {
      const b = blocks[i + 1];
      const d = owner[i];
      const color = d >= 0 ? `var(${DEV_COLORS[d % DEV_COLORS.length]})` : 'var(--c-other)';
      const where = d >= 0 ? devs[d]?.name || `GPU${d}` : 'the CPU';
      const parts = ram[i] || [];
      if (d >= 0 && parts.length) {
        // the RAM share grows with the number of matrices kept there (up, gate, down)
        const top = Math.round(100 - Math.min(3, parts.length) * 18);
        b.style.background = `linear-gradient(to bottom, ${color} 0 ${top}%, var(--c-other) ${top}% 100%)`;
        b.dataset.tip = `Layer ${i}: attention and KV cache on ${where}; ${parts.map(partLabel).join(', ')} in system RAM`;
      } else {
        b.style.background = color;
        b.dataset.tip = d >= 0 ? `Layer ${i}: entirely on ${where}` : `Layer ${i}: on the CPU (weights, KV cache and attention in system RAM)`;
      }
      b.classList.toggle('off', i >= lit);
      b.classList.toggle('cpu', d < 0);
    }
    let outDev = devs.findIndex((d) => (d.output_mib || 0) > 0);
    if (outDev < 0 && plan.gpu_layers > n) outDev = Math.max(-1, ...owner);
    const ob = blocks[n + 1];
    ob.style.background = outDev >= 0 ? `var(${DEV_COLORS[outDev % DEV_COLORS.length]})` : 'var(--c-other)';
    ob.dataset.tip = `Output layer: ${outDev >= 0 ? devs[outDev]?.name || `GPU${outDev}` : 'CPU'}`;
    ob.classList.toggle('off', loading && lit < n);
    clear(legend);
    devs.forEach((dv, di) => {
      const c = owner.filter((x) => x === di).length;
      if (!c) return;
      legend.appendChild(h('span', null, h('i', { style: { background: `var(${DEV_COLORS[di % DEV_COLORS.length]})` } }),
        `${dv.name || `GPU${di}`}: ${c} layers`));
    });
    const cpu = owner.filter((x) => x < 0).length;
    if (cpu) legend.appendChild(h('span', null, h('i', { style: { background: 'var(--c-other)' } }), `CPU: ${cpu} layers`));
    const split = ram.filter((x, i) => x.length && owner[i] >= 0).length;
    if (split) {
      legend.appendChild(h('span', { 'data-tip': 'Their attention and KV cache stay on the GPU' }, h('i', { class: 'split' }),
        `${split} layers with ${moe ? 'expert' : 'feed-forward'} weights in RAM`));
    }
  };
  return el;
}
