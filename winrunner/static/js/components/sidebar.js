// Right-hand activity column: engine, request pipeline, throughput, GPUs, token stream, event log.

import { h, clear, setText, icon } from '../core/dom.js';
import { store } from '../core/store.js';
import * as fmt from '../core/fmt.js';
import { TimeChart } from '../core/charts.js';
import { group, meter } from './ui.js';
import { engineState } from './state.js';
import { layerMap } from './memviz.js';

export function mountSidebar(root) {
  // ---- ENGINE ----------------------------------------------------------------
  const eLed = h('span', { class: 'led big' });
  const eState = h('span', { class: 'eng-state' });
  const eUp = h('span', { class: 'dim num' });
  const eModel = h('div', { class: 'eng-model ellipsis' });
  const eMeta = h('div', { class: 'eng-meta' });
  const eProg = h('div', { class: 'progress' }, h('div', { class: 'fill' }));
  const eProgLbl = h('div', { class: 'eng-phase dim' });
  const eLoadBox = h('div', { class: 'eng-load hidden' }, eProg, eProgLbl);
  const lmap = layerMap();
  const engine = group('Engine', [
    h('div', { class: 'eng-top' }, eLed, eState, h('span', { class: 'spacer' }), eUp),
    eModel, eMeta, eLoadBox, lmap,
  ], { icon: 'chip' });

  // ---- PIPELINE ----------------------------------------------------------------
  const stages = ['queued', 'prompt', 'generating', 'done'];
  const stageLabels = { queued: 'Receive', prompt: 'Prompt', generating: 'Generate', done: 'Complete' };
  const stageEls = stages.map((s) => h('div', { class: 'stage', 'data-s': s }, h('span', null, stageLabels[s])));
  const pReq = h('div', { class: 'pipe-req ellipsis dim' }, 'No requests yet');
  const pBar = meter(0, { label: '', cls: 'tall' });
  const pPrompt = h('div', { class: 'pipe-line' });
  const pGen = h('div', { class: 'pipe-line' });
  const pipeline = group('Pipeline', [
    pReq,
    h('div', { class: 'stages' }, stageEls),
    h('div', { class: 'pipe-lbl dim' }, 'Prompt evaluation'), pBar, pPrompt, pGen,
  ], { icon: 'bolt' });

  // ---- THROUGHPUT ----------------------------------------------------------------
  const tNow = h('div', { class: 'tp-big num' }, '-');
  const tUnit = h('div', { class: 'tp-unit dim' }, 'tokens / s');
  const tMeter = meter(0, { cls: 'tall' });
  const tSub = h('div', { class: 'tp-sub dim' });
  const tCanvas = h('canvas', { class: 'chart short' });
  const throughput = group('Throughput', [
    h('div', { class: 'tp-row' }, h('div', null, tNow, tUnit), h('div', { class: 'grow' }, tMeter, tSub)), tCanvas,
  ], { icon: 'monitor' });
  const tChart = new TimeChart(tCanvas, {
    window: 120, series: [{ key: 'tg', color: '--c-1', label: 'gen' }, { key: 'pp', color: '--c-2', label: 'prompt', fill: false, width: 1 }],
    fmt: (v) => v.toFixed(0), legend: false, now: () => store.now(), minMax: 10,
  });

  // ---- GPUs ----------------------------------------------------------------
  const gpuBody = h('div', { class: 'gpus' });
  const gpus = group('Devices', gpuBody, { icon: 'chip' });
  const gpuRows = new Map();

  // ---- TOKEN STREAM ----------------------------------------------------------------
  const tsBox = h('div', { class: 'tokstream' });
  const tsInfo = h('span', { class: 'sub' }, '');
  const tokens = group('Token Stream', tsBox, { icon: 'chat', cls: 'fill' });
  tokens.querySelector('.gt').appendChild(tsInfo);

  // ---- EVENT LOG ----------------------------------------------------------------
  const evBox = h('div', { class: 'evlog' });
  const events = group('Activity', evBox, { icon: 'log', cls: 'fill' });

  root.append(engine, pipeline, throughput, gpus, tokens, events);

  // ================= render functions =================
  let shownRid = null;
  let tokenBoundaries = store.settings?.ui?.token_boundaries !== false;
  store.on('settings', (s) => { tokenBoundaries = s?.ui?.token_boundaries !== false; });

  function renderEngine() {
    const s = engineState();
    eLed.className = `led big ${s.led}`;
    setText(eState, s.label);
    const inst = s.inst || store.primary;
    if (!inst) {
      setText(eModel, store.status?.engine ? 'No model loaded' : 'No engine installed');
      const e = store.status?.engine;
      setText(eMeta, e ? `llama.cpp ${e.build ? `b${e.build}` : e.version} · ${e.backend_label || e.backend}` : 'Settings › Engine › Download');
      eLoadBox.classList.add('hidden');
      lmap.update(null);
      setText(eUp, '');
      return;
    }
    setText(eModel, inst.name || inst.model);
    eModel.dataset.tip = inst.path;
    const p = inst.plan || {};
    const li = inst.load || {};
    const off = li.offload ? `${li.offload.gpu_layers}/${li.offload.total_layers} layers GPU` : `${p.gpu_layers ?? '-'}/${(p.n_layer || 0) + 1} layers GPU`;
    const fa = li.flash_attn === null || li.flash_attn === undefined ? p.flash_attn : (li.flash_attn ? 'on' : 'off');
    const kv = li.kv ? `${li.kv.k_type}/${li.kv.v_type}` : `${p.kv_k}/${p.kv_v}`;
    clear(eMeta);
    eMeta.append(
      h('span', null, `ctx ${fmt.ctx(inst.n_ctx_engine || inst.ctx)}`), h('span', null, `KV ${kv}`), h('span', null, `FA ${fa}`),
      h('span', null, off), h('span', null, `${inst.engine?.backend || ''} b${inst.engine?.build || ''}`),
      inst.vision ? h('span', { class: 'badge vision' }, icon('eye'), 'vision') : null,
      li.slots?.n_slots ? h('span', null, `${li.slots.n_slots} slots`) : null,
    );
    const loading = inst.state === 'loading' || inst.state === 'starting';
    eLoadBox.classList.toggle('hidden', !loading);
    if (loading) {
      eProg.firstChild.style.width = `${(inst.progress || 0) * 100}%`;
      setText(eProgLbl, `${inst.phase_label || inst.phase} · ${Math.round((inst.progress || 0) * 100)}%`);
    }
    setText(eUp, inst.state === 'ready' && inst.t_ready ? fmt.uptime(store.now() - inst.t_ready) : inst.recovering ? 'restarting' : inst.state);
    lmap.update(p, { progress: inst.progress, loading, loadInfo: loading ? null : li });
  }

  function focusReq() {
    const act = store.activeRequests;
    if (act.length) return act[act.length - 1];
    return store.focusRid ? store.requests.get(store.focusRid) : null;
  }

  function renderPipeline() {
    const r = focusReq();
    if (!r) return;
    clear(pReq);
    pReq.append(h('b', null, r.id), ` · ${r.client} · ${r.endpoint}`, r.images ? h('span', { class: 'badge vision', style: { marginLeft: '6px' } }, icon('eye'), `${r.images} img`) : null);
    const phase = r.phase === 'loading_model' ? 'queued' : r.phase;
    const idx = phase === 'done' ? 3 : phase === 'error' || phase === 'cancelled' ? 3 : stages.indexOf(phase);
    stageEls.forEach((el, i) => {
      el.classList.toggle('on', i === idx && !r.t_end);
      el.classList.toggle('past', i < idx || (r.t_end && i <= idx));
      el.classList.toggle('err', i === 3 && (phase === 'error' || phase === 'cancelled'));
    });
    stageEls[3].firstChild.textContent = phase === 'error' ? 'Error' : phase === 'cancelled' ? 'Cancelled' : 'Complete';
    const total = r.prompt_total || 0;
    const done = r.phase === 'generating' || r.t_end ? total : r.prompt_processed || 0;
    const pct = total ? (done / total) * 100 : r.t_end ? 100 : 0;
    const cached = r.prompt_cached || 0;
    let ppRate = r.prompt_tps;
    if (!ppRate && r.t_prompt_start && done > cached) ppRate = (done - cached) / Math.max(0.001, store.now() - r.t_prompt_start);
    pBar.set(pct, total ? `${fmt.num(done)} / ${fmt.num(total)} tokens` : r.phase === 'loading_model' ? 'loading model...' : '', r.phase === 'prompt' ? '' : 'ok');
    clear(pPrompt);
    pPrompt.append(h('span', null, 'cached ', h('b', null, fmt.num(cached))), h('span', null, 'rate ', h('b', null, `${fmt.tps(ppRate)} t/s`)),
      h('span', null, 'time ', h('b', null, r.prompt_ms ? fmt.ms(r.prompt_ms) : '-')));
    clear(pGen);
    const tps = r.gen_tps || r.live_tps;
    pGen.append(h('span', null, 'generated ', h('b', null, fmt.num(r.tokens))), h('span', null, 'TTFT ', h('b', null, r.ttft_ms ? fmt.ms(r.ttft_ms) : '-')),
      h('span', null, 'speed ', h('b', null, `${fmt.tps(tps)} t/s`)), r.finish_reason ? h('span', null, 'stop ', h('b', null, r.finish_reason)) : null,
      r.error ? h('div', { class: 'err ellipsis', 'data-tip': r.error }, r.error) : null);
  }

  let peak = 20;
  function renderThroughput() {
    const r = focusReq();
    const live = r && !r.t_end && r.phase === 'generating';
    const v = r ? (live ? r.live_tps : r.gen_tps) : null;
    setText(tNow, v ? fmt.tps(v) : '-');
    tNow.classList.toggle('live', !!live);
    const hist = store.tps.slice(-50).map((x) => x.tg || 0);
    peak = Math.max(20, ...hist, v || 0);
    tMeter.set(v ? (v / peak) * 100 : 0, undefined, live ? 'ok' : '');
    const avg = hist.length ? hist.reduce((a, b) => a + b, 0) / hist.length : null;
    setText(tSub, `peak ${fmt.tps(Math.max(...hist, 0) || null)} · avg ${fmt.tps(avg)} · last pp ${fmt.tps(r?.prompt_tps)} t/s`);
  }

  function pushTps() {
    const r = focusReq();
    const now = store.now();
    const live = r && !r.t_end && r.phase === 'generating' ? r.live_tps : null;
    let pp = null;
    if (r && !r.t_end && r.phase === 'prompt' && r.t_prompt_start && r.prompt_processed > (r.prompt_cached || 0)) {
      pp = (r.prompt_processed - (r.prompt_cached || 0)) / Math.max(0.001, now - r.t_prompt_start);
    }
    tChart.push(now, { tg: live || 0, pp: pp || 0 });
  }

  function gpuRow(g) {
    let row = gpuRows.get(g.id);
    if (!row) {
      const util = meter(0, { label: '' });
      const vram = meter(0, { label: '' });
      const info = h('div', { class: 'gpu-info dim' });
      const name = h('div', { class: 'gpu-name ellipsis' });
      row = { el: h('div', { class: 'gpu' }, name, h('div', { class: 'gpu-bars' }, h('span', { class: 'dim' }, 'GPU'), util, h('span', { class: 'dim' }, 'MEM'), vram), info), util, vram, info, name };
      gpuRows.set(g.id, row);
      gpuBody.appendChild(row.el);
    }
    return row;
  }

  function renderGpus(sample) {
    if (!sample) return;
    const list = sample.gpus || [];
    if (!list.length) {
      if (!gpuBody.querySelector('.none')) {
        clear(gpuBody);
        gpuRows.clear();
        gpuBody.appendChild(h('div', { class: 'none dim' }, 'No GPU telemetry available on this system.'));
      }
      return;
    }
    gpuBody.querySelector('.none')?.remove();
    for (const g of list) {
      const row = gpuRow(g);
      setText(row.name, `${g.id.toUpperCase()} · ${g.name}`);
      const u = g.util ?? null;
      row.util.set(u ?? 0, u === null ? 'n/a' : `${u.toFixed(0)}%`, u > 90 ? 'ok' : '');
      const used = g.vram_used ?? null;
      const tot = g.vram_total || 0;
      const vp = used !== null && tot ? (used / tot) * 100 : 0;
      row.vram.set(vp, used === null ? 'n/a' : `${fmt.gib(used)}/${fmt.gib(tot)} GiB`, vp > 95 ? 'err' : vp > 85 ? 'warn' : '');
      const parts = [];
      if (g.temp_edge !== undefined) parts.push(`${g.temp_edge.toFixed(0)}°C`);
      if (g.temp_hotspot !== undefined) parts.push(`jct ${g.temp_hotspot.toFixed(0)}°C`);
      if (g.power !== undefined) parts.push(`${g.power.toFixed(0)} W`);
      if (g.clk_gfx !== undefined) parts.push(`${g.clk_gfx.toFixed(0)} MHz`);
      if (g.fan_rpm !== undefined) parts.push(`${g.fan_rpm.toFixed(0)} rpm`);
      setText(row.info, parts.join(' · ') || ' ');
    }
  }

  // token stream
  function resetTokens(rid) {
    shownRid = rid;
    clear(tsBox);
    const buf = store.tokenBuf.get(rid) || [];
    if (buf.length) appendTokens(buf.slice(-1500), false);
    else {
      const r = store.requests.get(rid);
      if (r?.preview) tsBox.appendChild(h('span', { class: 'tk dim' }, r.preview));
    }
  }
  function appendTokens(pieces, animate = true) {
    const atBottom = tsBox.scrollTop + tsBox.clientHeight >= tsBox.scrollHeight - 30;
    const frag = document.createDocumentFragment();
    for (const [text, kind] of pieces) {
      const span = document.createElement('span');
      span.textContent = text;
      span.className = `tk ${kind === 'r' ? 'r' : kind === 't' ? 't' : ''}${tokenBoundaries ? ' b' : ''}${animate ? ' new' : ''}`;
      frag.appendChild(span);
    }
    tsBox.appendChild(frag);
    while (tsBox.childNodes.length > 2500) tsBox.removeChild(tsBox.firstChild);
    if (atBottom) tsBox.scrollTop = tsBox.scrollHeight;
  }
  function renderTokInfo() {
    const r = shownRid ? store.requests.get(shownRid) : null;
    setText(tsInfo, r ? `${r.id} · ${fmt.num(r.tokens)} tok${r.reasoning_tokens ? ` (${fmt.num(r.reasoning_tokens)} reasoning)` : ''}` : '');
  }

  // activity log
  function addEvent(ev, animate = true) {
    const lvl = ev.level || 'info';
    const row = h('div', { class: ['ev', lvl, animate && 'flash'] },
      h('span', { class: 'ev-t num' }, fmt.time(ev.t)), h('span', { class: `ev-l ${lvl}` }), h('span', { class: 'ev-x' }, ev.text));
    const atBottom = evBox.scrollTop + evBox.clientHeight >= evBox.scrollHeight - 30;
    evBox.appendChild(row);
    while (evBox.children.length > 300) evBox.removeChild(evBox.firstChild);
    if (atBottom) evBox.scrollTop = evBox.scrollHeight;
  }

  // ================= wiring =================
  store.on('hello', () => {
    clear(evBox);
    for (const ev of store.activity.slice(-200)) addEvent(ev, false);
    evBox.scrollTop = evBox.scrollHeight;
    tChart.setData([]);
    if (store.focusRid) resetTokens(store.focusRid);
    renderEngine(); renderPipeline(); renderThroughput(); renderTokInfo();
    renderGpus(store.lastMetrics);
  });
  store.on('activity', (ev) => addEvent(ev));
  store.on('metrics', (s) => { renderGpus(s); pushTps(); renderEngine(); });
  for (const e of ['instance', 'instance_progress', 'instance_removed', 'status', 'ws']) store.on(e, renderEngine);
  store.on('request', ({ rec, isNew, rerun }) => {
    if (isNew || (rerun && rec.id === shownRid)) resetTokens(rec.id);
    if (rec.id === shownRid) renderTokInfo();
    renderPipeline();
    renderThroughput();
    renderEngine();
  });
  store.on('tokens', (ev) => {
    if (ev.rid !== shownRid) {
      if (!store.requests.get(shownRid) || store.requests.get(shownRid)?.t_end) resetTokens(ev.rid);
      else return;
    }
    appendTokens(ev.pieces);
    renderTokInfo();
    renderPipeline();
    renderThroughput();
  });
}
