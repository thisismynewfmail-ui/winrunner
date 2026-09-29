// SERVER: endpoint, loaded models, slots, request history, client snippets.

import { h, clear, icon, setText, escapeHtml } from '../core/dom.js';
import { api, wr } from '../core/api.js';
import { store } from '../core/store.js';
import * as fmt from '../core/fmt.js';
import { group, btn, seg, toggle, toast, modal, kv, stat, empty, copyBtn, confirmBox } from '../components/ui.js';
import { memBars, actualDevices } from '../components/memviz.js';
import { navigate } from '../app.js';

export function mount(root) {
  const offs = [];
  const st = () => store.status || {};
  let devices = [];

  // ---- endpoint ---------------------------------------------------------------
  const epList = h('div', { class: 'endpoints' });
  const controls = h('div', { class: 'row wrap', style: { marginTop: '8px' } });
  const epGroup = group('API endpoint', [h('div', { class: 'dim', style: { marginBottom: '6px' } },
    'Base URL for OpenAI-compatible and LM Studio-compatible clients (Open WebUI, SillyTavern, Continue, AnythingLLM, OpenAI SDKs, LM Studio SDK).'),
  epList, controls], { icon: 'globe' });

  function renderEndpoint() {
    const s = st().server;
    if (!s) return;
    clear(epList);
    const rows = [...(s.lan || []).map((u) => ['Network', u]), ['This PC', s.local]];
    for (const [lbl, u] of rows) {
      epList.appendChild(h('div', { class: 'endpoint-row' }, h('span', { class: 'lbl' }, lbl), h('span', { class: 'endpoint-big grow' }, u), copyBtn(u, { tip: 'Copy base URL' })));
    }
    if (s.host !== '0.0.0.0' && s.host !== '::') {
      epList.appendChild(h('div', { class: 'note warn' }, `Bound to ${s.host}: other computers cannot connect. Set the bind address to 0.0.0.0 in Settings › Network.`));
    }
    if (s.restart_required) epList.appendChild(h('div', { class: 'note warn' }, 'Network settings changed: restart WinRunner to apply the new address/port.'));
    clear(controls);
    const set = async (patch) => { await store.saveSettings({ server: patch }); await store.refreshStatus(); };
    const ss = store.settings?.server || {};
    controls.append(
      toggle('API server running', s.api_enabled, (v) => set({ api_enabled: v }), { tip: 'When stopped, /v1 and /api/v0 answer 503' }),
      toggle('Just-in-time model loading', ss.jit_loading, (v) => set({ jit_loading: v }), { tip: 'Load the requested model automatically when a request names a model that is not loaded (LM Studio behaviour)' }),
      toggle('Unload other models when loading', ss.auto_evict, (v) => set({ auto_evict: v }), { tip: 'Keep only one model in VRAM (recommended)' }),
      toggle('Require API key', !!ss.api_key, async (v) => {
        const key = v ? `wr-${Array.from(crypto.getRandomValues(new Uint8Array(18))).map((b) => b.toString(16).padStart(2, '0')).join('')}` : '';
        await set({ api_key: key });
        if (key) modal('API key created', h('div', null, h('p', null, 'Clients must send this key as "Authorization: Bearer <key>":'),
          h('pre', { class: 'code wrap' }, key), copyBtn(key, { label: 'Copy key', small: false })));
        renderEndpoint();
      }, { tip: 'Protect the API with a bearer token' }),
      ss.api_key ? copyBtn(ss.api_key, { label: 'Copy key' }) : null,
    );
  }

  // ---- instances ---------------------------------------------------------------
  const instBody = h('div');
  const instGroup = group('Loaded models', instBody, { icon: 'chip' });
  const instCards = new Map();

  function renderInstances() {
    const list = [...store.instances.values()];
    if (!list.length) {
      clear(instBody);
      instCards.clear();
      instBody.appendChild(empty('No model loaded', store.settings?.server?.jit_loading
        ? 'Requests that name a model will load it automatically (just-in-time loading is on).' : 'Load a model from the Library.',
      btn('Open Library', () => navigate('library'), { icon: 'model' })));
      return;
    }
    instBody.querySelector('.empty')?.remove();
    for (const [id, c] of instCards) if (!store.instances.has(id)) { c.el.remove(); instCards.delete(id); }
    for (const inst of list) {
      let c = instCards.get(inst.id);
      if (!c) {
        c = { el: h('div', { class: 'inst-card' }), head: h('div', { class: 'inst-head' }), stats: h('div', { class: 'stats' }), bars: memBars(), slots: h('div', { class: 'slots' }), extra: h('div') };
        c.el.append(c.head, c.stats, h('div', { class: 'divider' }), c.bars, h('div', { class: 'lm-title dim' }, 'Slots'), c.slots, c.extra);
        instCards.set(inst.id, c);
        instBody.appendChild(c.el);
      }
      const led = inst.state === 'ready' ? 'ok' : inst.state === 'error' ? 'err' : 'acc blink';
      clear(c.head);
      c.head.append(h('span', { class: `led big ${led}` }), h('b', null, inst.model),
        h('span', { class: 'chip' }, inst.state === 'loading' ? `${inst.phase_label} ${Math.round(inst.progress * 100)}%` : inst.state),
        inst.vision ? h('span', { class: 'badge vision' }, icon('eye'), 'vision') : null,
        h('span', { class: 'dim' }, `llama.cpp b${inst.engine?.build} ${inst.engine?.backend} · pid ${inst.pid ?? '-'}`),
        h('span', { class: 'spacer' }),
        btn('Library', () => navigate('library', { model: inst.model }), { cls: 'small', icon: 'model' }),
        btn('Log', () => navigate('logs', { iid: inst.id }), { cls: 'small', icon: 'log' }),
        btn('Unload', async () => { await api.post(wr('/models/unload'), { id: inst.id }); }, { cls: 'small', icon: 'eject' }));
      const li = inst.load || {};
      clear(c.stats);
      c.stats.append(
        stat('Uptime', inst.state === 'ready' && inst.t_ready ? fmt.uptime(store.now() - inst.t_ready) : '-'),
        stat('Context', fmt.num(inst.n_ctx_engine || inst.ctx)),
        stat('KV cache', li.kv ? `${li.kv.k_type} ${fmt.mib(li.kv.mib)}` : '-'),
        stat('GPU layers', li.offload ? `${li.offload.gpu_layers}/${li.offload.total_layers}` : '-'),
        stat('Flash attn', li.flash_attn === null || li.flash_attn === undefined ? '-' : li.flash_attn ? 'on' : 'off'),
        stat('Requests', fmt.num(inst.requests_served)),
        stat('Active', fmt.num(inst.active_requests)),
        stat('Template', inst.template_verified === true ? 'GGUF ✓' : inst.template_verified === false ? 'override' : '-'));
      c.bars.update(actualDevices(inst, devices));
      renderSlots(inst, c.slots);
      clear(c.extra);
      if (inst.error) c.extra.appendChild(h('div', { class: 'note err' }, inst.error));
    }
  }

  function renderSlots(inst, host) {
    const slots = store.slots.get(inst.id) || [];
    clear(host);
    if (!slots.length) { host.appendChild(h('div', { class: 'dim' }, inst.state === 'ready' ? 'Waiting for slot data...' : '-')); return; }
    for (const s of slots) {
      const used = s.n_prompt ? s.n_prompt + (s.n_decoded || 0) : 0;
      const m = h('div', { class: `meter seg-blocks ${s.processing ? 'busy' : ''}` }, h('div', { class: `fill ${s.processing ? 'ok' : ''}`, style: { width: `${Math.min(100, (used / (s.n_ctx || 1)) * 100)}%` } }));
      host.appendChild(h('div', { class: 'slot-row' }, h('span', null, h('span', { class: `led ${s.processing ? 'ok pulse' : ''}` }), ` slot ${s.id}`), m,
        h('span', { class: 'dim num' }, `${fmt.num(used)} / ${fmt.ctx(s.n_ctx)}${s.processing ? ' · busy' : ''}`)));
    }
  }

  // ---- requests ---------------------------------------------------------------
  const reqBody = h('tbody');
  const reqGroup = group('Requests', h('div', { class: 'tbl-wrap', style: { maxHeight: '46vh' } },
    h('table', { class: 'tbl req-tbl compact' }, h('thead', null, h('tr', null,
      ['Time', 'ID', 'Client', 'Endpoint', 'Model', 'Img', 'Prompt', 'Cached', 'Output', 'TTFT', 'PP t/s', 'TG t/s', 'Total', 'Status'].map((x, i) =>
        h('th', { class: i >= 5 && i <= 12 ? 'num' : '' }, x)))), reqBody)), { icon: 'list', sub: '' });
  const rows = new Map();
  function reqRow(r) {
    const cls = r.t_end ? (r.error ? 'error' : '') : 'active';
    const cells = [fmt.time(r.t_start), r.id, r.client, r.endpoint, r.model, r.images || '', fmt.num(r.prompt_total), r.prompt_cached ? fmt.num(r.prompt_cached) : '',
      fmt.num(r.tokens), r.ttft_ms ? fmt.ms(r.ttft_ms) : '-', fmt.tps(r.prompt_tps), fmt.tps(r.gen_tps || r.live_tps), fmt.ms(r.duration_ms),
      r.error ? 'error' : r.t_end ? (r.finish_reason || r.phase) : r.phase];
    let tr = rows.get(r.id);
    if (!tr) {
      tr = h('tr', { onclick: () => showReq(r.id) }, cells.map((_, i) => h('td', { class: i >= 5 && i <= 12 ? 'num' : '' })));
      rows.set(r.id, tr);
      reqBody.insertBefore(tr, reqBody.firstChild);
      tr.classList.add('flash');
      while (reqBody.children.length > 300) { const last = reqBody.lastChild; rows.forEach((v, k) => { if (v === last) rows.delete(k); }); last.remove(); }
    }
    tr.className = cls;
    cells.forEach((v, i) => setText(tr.children[i], v ?? ''));
    tr.children[13].dataset.tip = r.error || '';
  }
  async function showReq(id) {
    const r = await api.get(wr(`/requests/item?id=${encodeURIComponent(id)}`));
    modal(`Request ${r.id}`, h('div', null,
      kv([['Endpoint', r.endpoint], ['Client', `${r.client} · ${r.user_agent || ''}`], ['Model', r.model], ['Instance', r.instance || '-'],
        ['Stream', r.stream ? 'yes' : 'no (streamed internally)'], ['Images', r.images || 0], ['Parameters', JSON.stringify(r.params)],
        ['Prompt', `${fmt.num(r.prompt_total)} tokens (${fmt.num(r.prompt_cached)} from cache) · ${fmt.ms(r.prompt_ms)} · ${fmt.tps(r.prompt_tps)} t/s`],
        ['Generation', `${fmt.num(r.tokens)} tokens${r.reasoning_tokens ? ` (${r.reasoning_tokens} reasoning)` : ''} · ${fmt.ms(r.gen_ms)} · ${fmt.tps(r.gen_tps)} t/s`],
        ['Time to first token', fmt.ms(r.ttft_ms)], ['Total time', fmt.ms(r.duration_ms)], ['Finish reason', r.finish_reason || '-'],
        ['Tool calls', r.tool_calls?.join(', ') || '-'], ['Error', r.error || '-']]),
      h('div', { class: 'form-section' }, 'Output'), h('pre', { class: 'code wrap', style: { maxHeight: '40vh' } }, r.preview || '(no text output)')), { wide: true });
  }

  // ---- snippets ---------------------------------------------------------------
  const snipCode = h('pre', { class: 'code wrap', style: { maxHeight: '320px' } });
  let snipKind = 'curl';
  const snipSeg = seg([['curl', 'curl'], ['python', 'Python (openai)'], ['js', 'JavaScript'], ['vision', 'Vision (curl)'], ['lms', 'LM Studio REST']], snipKind,
    (v) => { snipKind = v; renderSnip(); }, { small: true });
  const snipGroup = group('Quick start', [h('div', { class: 'row snip-tabs' }, snipSeg, h('span', { class: 'spacer' }), copyBtn(() => snipCode.textContent, { label: 'Copy', small: true })), snipCode],
    { icon: 'log', collapsible: true });
  function renderSnip() {
    const s = st().server || {};
    const base = s.lan?.[0] || s.local || 'http://localhost:5070/v1';
    const root = base.replace(/\/v1$/, '');
    const model = store.primary?.model || 'MODEL_ID';
    const auth = store.settings?.server?.api_key ? ` \\\n  -H "Authorization: Bearer ${store.settings.server.api_key}"` : '';
    const key = store.settings?.server?.api_key || 'not-needed';
    const S = {
      curl: `curl ${base}/chat/completions \\\n  -H "Content-Type: application/json"${auth} \\\n  -d '{\n    "model": "${model}",\n    "messages": [\n      {"role": "system", "content": "You are a helpful assistant."},\n      {"role": "user", "content": "Explain KV caching in two sentences."}\n    ],\n    "stream": true\n  }'`,
      python: `from openai import OpenAI\n\nclient = OpenAI(base_url="${base}", api_key="${key}")\n\nstream = client.chat.completions.create(\n    model="${model}",\n    messages=[{"role": "user", "content": "Explain KV caching in two sentences."}],\n    stream=True,\n)\nfor chunk in stream:\n    delta = chunk.choices[0].delta if chunk.choices else None\n    if delta and delta.content:\n        print(delta.content, end="", flush=True)`,
      js: `const res = await fetch("${base}/chat/completions", {\n  method: "POST",\n  headers: { "Content-Type": "application/json"${store.settings?.server?.api_key ? `, Authorization: "Bearer ${key}"` : ''} },\n  body: JSON.stringify({\n    model: "${model}",\n    messages: [{ role: "user", content: "Hello!" }],\n  }),\n});\nconst data = await res.json();\nconsole.log(data.choices[0].message.content);`,
      vision: `# image as base64 data URI (JPEG, PNG, WebP, GIF, BMP, TIFF...)\nIMG=$(base64 -w0 photo.jpg)\ncurl ${base}/chat/completions \\\n  -H "Content-Type: application/json"${auth} \\\n  -d '{\n    "model": "${model}",\n    "messages": [{\n      "role": "user",\n      "content": [\n        {"type": "text", "text": "Describe this image."},\n        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$IMG"'"}}\n      ]\n    }]\n  }'`,
      lms: `# LM Studio REST API (beta) - model list with state, type (llm/vlm/embeddings) and context\ncurl ${root}/api/v0/models${auth}\n\n# chat with LM Studio statistics (tokens_per_second, time_to_first_token, stop_reason)\ncurl ${root}/api/v0/chat/completions \\\n  -H "Content-Type: application/json"${auth} \\\n  -d '{"model": "${model}", "messages": [{"role": "user", "content": "Hi"}]}'`,
    };
    snipCode.textContent = S[snipKind];
  }

  const refGroup = group('Endpoints', h('table', { class: 'tbl compact' }, h('tbody', null, [
    ['GET', '/v1/models', 'Models (all library models when JIT loading is on, otherwise loaded models)'],
    ['POST', '/v1/chat/completions', 'Chat with streaming, tools, JSON schema, vision (image_url), reasoning_content'],
    ['POST', '/v1/completions', 'Raw text completion'],
    ['POST', '/v1/responses', 'OpenAI Responses API (text, images, streaming events)'],
    ['POST', '/v1/messages', 'Anthropic Messages API format'],
    ['POST', '/v1/embeddings', 'Embeddings (embedding models)'],
    ['POST', '/v1/rerank', 'Reranking (reranker models)'],
    ['GET', '/api/v0/models', 'LM Studio REST: models with type, state, arch, quantization, context'],
    ['POST', '/api/v0/chat/completions', 'LM Studio REST: chat with stats, model_info and runtime'],
  ].map(([m, p, d]) => h('tr', null, h('td', { class: 'acc' }, m), h('td', { class: 'mono' }, p), h('td', { class: 'dim' }, d))))), { icon: 'list', collapsible: true, collapsed: true });

  root.append(h('div', { class: 'pg-head' }, h('h2', null, 'Server'), h('span', { class: 'sub' }, 'OpenAI / LM Studio compatible API')),
    epGroup, instGroup, reqGroup, snipGroup, refGroup);

  // ---- wiring ---------------------------------------------------------------
  const renderAll = () => { renderEndpoint(); renderInstances(); renderSnip(); };
  offs.push(store.on('status', renderAll));
  offs.push(store.on('settings', renderAll));
  offs.push(store.on('instance', renderInstances));
  offs.push(store.on('instance_progress', renderInstances));
  offs.push(store.on('instance_removed', renderInstances));
  offs.push(store.on('slots', (ev) => { const c = instCards.get(ev.iid); const inst = store.instances.get(ev.iid); if (c && inst) renderSlots(inst, c.slots); }));
  offs.push(store.on('request', ({ rec }) => reqRow(rec)));
  offs.push(store.on('hello', () => { clear(reqBody); rows.clear(); for (const r of store.requests.values()) reqRow(r); renderAll(); }));
  const timer = setInterval(renderInstances, 5000);
  api.get(wr('/hardware')).then((hw) => { devices = hw.engine_devices || []; renderInstances(); }).catch(() => {});
  for (const r of store.requests.values()) reqRow(r);
  renderAll();
  return () => { offs.forEach((f) => f()); clearInterval(timer); };
}
