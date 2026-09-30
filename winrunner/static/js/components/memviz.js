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
    // compute includes the backend's run-time scratch (e.g. flash attention's F16 copy of a quantized KV cache)
    weights: d.weights_mib + (d.output_mib || 0), kv: d.kv_mib,
    compute: Math.max(0, d.compute_mib + (d.scratch_mib || 0) + (d.calib_mib || 0)), mmproj: d.mmproj_mib,
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

/**
 * Layer map: one block per transformer layer, coloured by the device holding it.
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
    // device for each layer; prefer the actual allocation when known
    const owner = new Array(n).fill(-1);
    let counts = devs.map((d) => d.layers || 0);
    const buf = loadInfo?.buffers;
    const off = loadInfo?.offload;
    if (buf && off && devs.length > 1 && plan.source !== 'manual') {
      const w = devs.map((d) => (buf[d.name]?.model || 0));
      const tot = w.reduce((a, b) => a + b, 0);
      const gpuLayers = Math.min(n, off.gpu_layers);
      if (tot > 0) {
        counts = w.map((x) => Math.round((x / tot) * gpuLayers));
        const diff = gpuLayers - counts.reduce((a, b) => a + b, 0);
        counts[counts.length - 1] += diff;
      }
    } else if (off && devs.length === 1) {
      counts = [Math.min(n, off.gpu_layers)];
    }
    const gpuTotal = counts.reduce((a, b) => a + b, 0);
    let il = n - gpuTotal;
    counts.forEach((c, di) => { for (let k = 0; k < c && il < n; k++) owner[il++] = di; });
    const moe = plan.n_cpu_moe || 0;
    const lit = loading ? Math.floor(Math.max(0, Math.min(1, (progress - 0.12) / 0.72)) * n) : n;
    const s = `${n}|${owner.join(',')}|${moe}|${lit}|${loading}`;
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
    for (let i = 0; i < n; i++) {
      const b = blocks[i + 1];
      const d = owner[i];
      const color = d >= 0 ? `var(${DEV_COLORS[d % DEV_COLORS.length]})` : 'var(--c-other)';
      const where = d >= 0 ? devs[d]?.name || `GPU${d}` : 'CPU';
      if (d >= 0 && i < moe) {
        b.style.background = `linear-gradient(to bottom, ${color} 0 45%, var(--c-other) 45% 100%)`;
        b.dataset.tip = `Layer ${i}: attention on ${where}, expert weights in system RAM`;
      } else {
        b.style.background = color;
        b.dataset.tip = `Layer ${i}: ${where}`;
      }
      const on = i < lit;
      b.classList.toggle('off', !on);
      b.classList.toggle('cpu', d < 0);
    }
    const outDev = gpuTotal > 0 ? counts.length - 1 : -1;
    const ob = blocks[n + 1];
    ob.style.background = outDev >= 0 ? `var(${DEV_COLORS[outDev % DEV_COLORS.length]})` : 'var(--c-other)';
    ob.classList.toggle('off', loading && lit < n);
    clear(legend);
    counts.forEach((c, di) => {
      if (!c) return;
      legend.appendChild(h('span', null, h('i', { style: { background: `var(${DEV_COLORS[di % DEV_COLORS.length]})` } }),
        `${devs[di]?.name || `GPU${di}`}: ${c} layers`));
    });
    if (n - gpuTotal > 0) legend.appendChild(h('span', null, h('i', { style: { background: 'var(--c-other)' } }), `CPU: ${n - gpuTotal} layers`));
    if (moe) legend.appendChild(h('span', null, h('i', { class: 'split' }), `${moe} layers with experts in RAM`));
  };
  return el;
}
