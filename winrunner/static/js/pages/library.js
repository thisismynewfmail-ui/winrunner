// LIBRARY: models on disk, load configuration, memory plan, metadata, chat template, vision, presets.

import { h, clear, icon, setText, debounce, escapeHtml } from '../core/dom.js';
import { api, wr } from '../core/api.js';
import { store } from '../core/store.js';
import * as fmt from '../core/fmt.js';
import { group, btn, seg, select, input, toggle, toast, modal, kv, stat, empty, copyBtn, confirmBox } from '../components/ui.js';
import { loadForm } from '../components/loadform.js';
import { memBars, planDevices, actualDevices, layerMap } from '../components/memviz.js';
import { navigate, takeNavParams } from '../app.js';

let lastSelected = null;

export function capBadges(m) {
  return h('span', { class: 'caps-row' },
    m.vision ? h('span', { class: 'badge vision', 'data-tip': 'Vision: image input supported (mmproj paired)' }, icon('eye'), 'Vision') : null,
    m.audio ? h('span', { class: 'badge audio', 'data-tip': 'Audio input supported' }, icon('audio'), 'Audio') : null,
    m.tools ? h('span', { class: 'badge tools', 'data-tip': 'Chat template supports tool / function calling' }, icon('tool'), 'Tools') : null,
    m.reasoning ? h('span', { class: 'badge reason', 'data-tip': 'Reasoning / thinking model' }, icon('think'), 'Reasoning') : null,
    m.kind === 'embedding' ? h('span', { class: 'badge embed', 'data-tip': 'Embedding model' }, icon('embed'), 'Embed') : null,
    m.is_moe ? h('span', { class: 'badge moe', 'data-tip': `Mixture of experts: ${m.expert_used_count} of ${m.expert_count} experts active per token` }, `MoE ${m.expert_used_count}/${m.expert_count}`) : null,
  );
}

export function mount(root) {
  let models = [];
  let filter = 'all';
  let search = '';
  let sort = 'name';
  let selected = takeNavParams()?.model || lastSelected;
  let detail = null;
  let subtab = 'load';
  let devices = [];
  let engine = null;
  const offs = [];

  const count = h('span', { class: 'sub' });
  const head = h('div', { class: 'pg-head' }, h('h2', null, 'Library'), count, h('span', { class: 'spacer' }),
    btn('Download', () => openDownloads(), { icon: 'download', tip: 'Search and download GGUF models from Hugging Face' }),
    btn('Folders', () => openFolders(), { icon: 'folder', tip: 'Model folders scanned for GGUF files' }),
    btn('Rescan', async () => { await api.post(wr('/library/rescan')); await loadList(); }, { icon: 'refresh' }));
  const searchIn = input('', (v) => { search = v.toLowerCase(); renderList(); }, { placeholder: 'Search models...', live: true, cls: 'grow' });
  const toolbar = h('div', { class: 'toolbar' }, icon('search'), searchIn,
    seg([['all', 'All'], ['llm', 'Text'], ['vlm', 'Vision'], ['embeddings', 'Embedding'], ['loaded', 'Loaded']], filter, (v) => { filter = v; renderList(); }, { small: true }),
    select([['name', 'Sort: name'], ['size', 'Sort: size'], ['recent', 'Sort: last used'], ['params', 'Sort: parameters']], sort, (v) => { sort = v; renderList(); }));
  const tbody = h('tbody');
  const table = h('table', { class: 'tbl' }, h('thead', null, h('tr', null,
    h('th', { style: { width: '14px' } }), h('th', null, 'Model'), h('th', null, 'Arch'), h('th', { class: 'num' }, 'Params'),
    h('th', null, 'Quant'), h('th', { class: 'num' }, 'Size'), h('th', { class: 'num' }, 'Context'), h('th', null, 'Capabilities'))), tbody);
  const listWrap = h('div', { class: 'tbl-wrap lib-list' }, table);
  const detailHost = h('div', { class: 'lib-detail' });
  root.append(head, toolbar, listWrap, detailHost);

  async function loadList() {
    try {
      const d = await api.get(wr('/library'));
      models = d.models;
      const vis = models.filter((m) => m.vision).length;
      setText(count, `${models.length} models · ${vis} with vision · ${d.dirs.filter((x) => x.exists).length} folders`);
      renderList();
      if (!models.length) {
        clear(detailHost);
        detailHost.appendChild(group('No models found', empty('No GGUF models found', `Scanned: ${d.dirs.map((x) => x.path).join(' · ')}`,
          h('div', { class: 'row', style: { justifyContent: 'center' } }, btn('Download a model', () => openDownloads(), { icon: 'download', cls: 'primary' }),
            btn('Add a folder', () => openFolders(), { icon: 'folder' })))));
      } else if (!selected || !models.find((m) => m.id === selected)) {
        const loaded = models.find((m) => m.state === 'ready');
        select_(loaded?.id || models[0].id);
      } else if (!detail) select_(selected);
    } catch (e) {
      toast(e.message, 'err');
    }
  }

  function renderList() {
    clear(tbody);
    let list = models.filter((m) => (filter === 'all' || (filter === 'loaded' ? m.state : m.type === filter))
      && (!search || `${m.id} ${m.name} ${m.architecture} ${m.quant} ${m.publisher}`.toLowerCase().includes(search)));
    const cmp = { name: (a, b) => a.id.localeCompare(b.id), size: (a, b) => b.file_size - a.file_size,
      recent: (a, b) => (b.last_loaded || 0) - (a.last_loaded || 0), params: (a, b) => b.n_params - a.n_params }[sort];
    list = list.sort(cmp);
    for (const m of list) {
      const inst = [...store.instances.values()].find((i) => i.model === m.id);
      const st = inst?.state || m.state;
      const led = st === 'ready' ? 'ok' : st === 'loading' || st === 'starting' ? 'acc blink' : st === 'error' ? 'err' : '';
      const tr = h('tr', { class: m.id === selected ? 'sel' : '', onclick: () => select_(m.id), ondblclick: () => loadModel() },
        h('td', null, h('span', { class: `led ${led}`, 'data-tip': st || 'not loaded' })),
        h('td', null, h('div', { class: 'ellipsis', style: { fontWeight: 'bold', maxWidth: '360px' } }, m.name),
          h('div', { class: 'dim ellipsis', style: { fontSize: '10px', maxWidth: '360px' } }, `${m.id}${m.alias ? ` · alias "${m.alias}"` : ''}`)),
        h('td', null, m.architecture), h('td', { class: 'num' }, m.size_label || fmt.params(m.n_params)), h('td', null, m.quant),
        h('td', { class: 'num' }, fmt.bytes(m.file_size)), h('td', { class: 'num' }, fmt.ctx(m.context_length)), h('td', null, capBadges(m)));
      tbody.appendChild(tr);
    }
    if (!list.length && models.length) tbody.appendChild(h('tr', null, h('td', { colspan: 8, class: 'dim center' }, 'No models match the filter.')));
  }

  async function select_(id) {
    selected = id;
    lastSelected = id;
    for (const tr of tbody.children) tr.classList.remove('sel');
    renderList();
    try {
      const [d, eng, hw] = await Promise.all([api.get(wr(`/library/item?id=${encodeURIComponent(id)}`)), api.get(wr('/engine')),
        devices.length ? Promise.resolve(null) : api.get(wr('/hardware')).catch(() => null)]);
      detail = d;
      engine = eng.active;
      if (hw) devices = hw.engine_devices || [];
      renderDetail();
    } catch (e) {
      toast(e.message, 'err');
    }
  }

  // ------------------------------------------------------------------------------------
  // detail
  // ------------------------------------------------------------------------------------
  let form = null;
  let formValues = null;
  let formFor = null;
  let lastInstState = null;
  let planView = null;
  let actualView = null;
  let cmdBox = null;
  let loadBtn = null;

  function instFor() { return [...store.instances.values()].find((i) => i.model === detail?.id); }

  function renderDetail() {
    clear(detailHost);
    if (!detail) return;
    const d = detail;
    const inst = instFor();
    lastInstState = inst?.state || null;
    loadBtn = btn(inst?.state === 'ready' ? 'Reload' : 'Load model', () => loadModel(), { icon: 'play', cls: 'primary big' });
    const unloadBtn = inst ? btn('Unload', async () => { await api.post(wr('/models/unload'), { id: inst.id }); }, { icon: 'eject', cls: 'big' }) : null;
    const chatBtn = btn('Chat', () => navigate('chat', { model: d.id }), { icon: 'chat', cls: 'big', tip: 'Open the chat console with this model' });
    const header = h('div', { class: 'lib-head' },
      h('div', { class: 'grow' },
        h('div', { class: 'lib-title' }, d.name),
        h('div', { class: 'row wrap', style: { gap: '6px', marginTop: '4px' } },
          h('code', { class: 'acc' }, d.id), copyBtn(d.id, { tip: 'Copy model id (use as "model" in API requests)' }), capBadges(d),
          d.template_family ? h('span', { class: 'chip', 'data-tip': 'Prompt format family of the embedded chat template' }, d.template_family) : null)),
      h('div', { class: 'row' }, loadBtn, unloadBtn, chatBtn));

    const tabs = [['load', 'Load'], ['props', 'Properties'], ['template', 'Chat template'], ['vision', `Vision${d.vision ? '' : ' (none)'}`], ['preset', 'API preset']];
    const tabBar = h('div', { class: 'subtabs' }, tabs.map(([k, l]) => h('button', { class: k === subtab ? 'on' : '', onclick: () => { subtab = k; renderDetail(); } }, l)));
    const body = h('div');
    const g = group('Model', [tabBar, body], { icon: 'model', sub: d.file_name, tools: [h('span', { class: 'dim ellipsis', style: { maxWidth: '300px', fontSize: '10px' }, 'data-tip': d.path }, d.path)] });
    detailHost.append(header, g);
    ({ load: renderLoad, props: renderProps, template: renderTemplate, vision: renderVision, preset: renderPreset }[subtab])(body);
  }

  // ---- LOAD ----------------------------------------------------------------------------------
  function renderLoad(body) {
    const d = detail;
    if (formFor !== d.id || !formValues) { formValues = { ...d.effective }; formFor = d.id; }
    const saved = Object.keys(d.profile.load || {});
    form = loadForm({ values: formValues, defaults: d.defaults, saved, model: d, devices, engine, onChange: (vals) => { formValues = vals; schedulePlan(); } });
    const planHost = h('div');
    const planStatus = h('div', { class: 'plan-status' });
    const planStats = h('div', { class: 'stats wide' });
    const planNotes = h('div');
    const bars = memBars();
    const lmap = layerMap();
    const verifyBtn = btn('Verify with engine', () => runPlan(true), { icon: 'check', tip: 'Run llama-fit-params: the engine\'s own memory projection for these settings' });
    planView = { planStatus, planStats, planNotes, bars, lmap, verifyBtn };
    planHost.append(planStatus, planStats, h('div', { class: 'divider' }), bars, h('div', { class: 'lm-title dim' }, 'Layer placement'), lmap, planNotes);
    cmdBox = h('pre', { class: 'code wrap', style: { maxHeight: '140px' } }, '');
    const inst = instFor();
    actualView = h('div');
    const actions = h('div', { class: 'load-actions' },
      btn('Save as model defaults', async () => {
        await api.put(wr('/library/profile'), { id: d.id, load: diffFromDefaults(formValues, d.defaults) });
        toast('Load settings saved for this model', 'ok');
        formFor = null;
        select_(d.id);
      }, { icon: 'check', tip: 'Remember these settings for this model (used by the API for JIT loads too)' }),
      btn('Reset to global defaults', async () => {
        if (!(await confirmBox('Reset load settings', 'Discard this model\'s saved load settings and use the global defaults?', 'Reset'))) return;
        await api.put(wr('/library/profile'), { id: d.id, load: {} });
        formFor = null;
        select_(d.id);
      }, { icon: 'refresh' }),
      h('span', { class: 'spacer' }),
      btn(inst?.state === 'ready' ? 'Reload with these settings' : 'Load model', () => loadModel(), { icon: 'play', cls: 'primary big' }));
    body.append(
      inst ? group('Loaded instance', actualView, { icon: 'chip', sub: inst.state }) : null,
      h('div', { class: 'load-grid' },
        h('div', { class: 'plan-col' },
          group('Memory plan', planHost, { icon: 'mem', tools: [verifyBtn] }),
          group('Engine command line', [cmdBox, h('div', { class: 'row', style: { marginTop: '6px' } }, copyBtn(() => cmdBox.textContent, { label: 'Copy', small: true }),
            h('span', { class: 'dim', style: { fontSize: '10px' } }, 'Flags are generated for the installed engine build; port and API key are assigned at load.'))], { icon: 'log' })),
        h('div', { class: 'form-col' }, group('Load configuration', form, { icon: 'sliders' }))),
      actions);
    if (inst) renderActual();
    runPlan(false);
  }

  function renderActual() {
    const inst = instFor();
    if (!inst || !actualView) return;
    clear(actualView);
    const li = inst.load || {};
    const s = h('div', { class: 'stats' },
      stat('State', inst.state === 'loading' ? `${Math.round(inst.progress * 100)}%` : inst.state),
      stat('Context', fmt.num(inst.n_ctx_engine || inst.ctx)),
      stat('KV cache', li.kv ? `${li.kv.k_type === li.kv.v_type ? li.kv.k_type : `${li.kv.k_type}/${li.kv.v_type}`} ${fmt.mib(li.kv.mib)}` : '-'),
      stat('Flash attn', li.flash_attn === null || li.flash_attn === undefined ? '-' : li.flash_attn ? 'on' : 'off'),
      stat('GPU layers', li.offload ? `${li.offload.gpu_layers}/${li.offload.total_layers}` : '-'),
      stat('Slots', li.slots?.n_slots ?? '-'),
      stat('Load time', li.load_seconds ? `${li.load_seconds} s` : '-'),
      stat('Template', inst.template_verified === true ? 'GGUF ✓' : inst.template_verified === false ? 'differs' : inst.params?.chat_template_mode || '-',
        'Whether the engine is using the exact chat template embedded in the GGUF'));
    const bars = memBars();
    bars.update(actualDevices(inst, devices));
    const lm = layerMap();
    lm.update(inst.plan, { loadInfo: li, loading: inst.state !== 'ready', progress: inst.progress });
    actualView.append(s, h('div', { class: 'divider' }), bars, lm,
      inst.error ? h('div', { class: 'note err' }, inst.error) : null);
  }

  const schedulePlan = debounce(() => runPlan(false), 350);
  let planSeq = 0;
  async function runPlan(verify) {
    if (!planView || !detail) return;
    const seq = ++planSeq;
    const { planStatus, planStats, planNotes, bars, lmap, verifyBtn } = planView;
    if (verify) { verifyBtn.disabled = true; verifyBtn.querySelector('span').textContent = 'Running engine projection...'; }
    planStatus.classList.add('busy');
    try {
      const r = await api.post(wr('/plan'), { id: detail.id, overrides: diffFromDefaults(formValues, detail.defaults, true), verify });
      if (seq !== planSeq) return;
      const p = r.plan;
      if (r.devices?.length) devices = r.devices;
      clear(planStatus);
      const full = p.full_offload;
      const cls = !r.devices?.length ? 'warn' : full ? 'ok' : 'warn';
      const headline = !r.devices?.length ? 'CPU ONLY' : full ? 'FULL GPU OFFLOAD' : p.n_cpu_moe ? 'GPU + EXPERTS IN RAM' : 'PARTIAL OFFLOAD';
      planStatus.append(h('span', { class: `led ${cls === 'ok' ? 'ok' : 'warn'}` }), h('b', { class: cls }, headline),
        h('span', { class: 'dim' }, p.source === 'engine' ? 'verified by engine projection (llama-fit-params)' : p.use_engine_fit ? 'estimate · engine fits layers at load' : 'estimate'));
      clear(planStats);
      const vramUsed = p.devices.reduce((a, x) => a + x.used_mib, 0);
      const vramFree = p.devices.reduce((a, x) => a + x.free_mib - x.margin_mib, 0);
      planStats.append(
        stat('Context', fmt.num(p.ctx), p.ctx !== p.ctx_requested ? `Requested ${fmt.num(p.ctx_requested)}` : null),
        stat('KV cache', `${p.kv_k.toUpperCase()} · ${fmt.mib(p.totals.kv_mib)}`, `${fmt.bytes(p.kv_bytes_per_token)} per token`),
        stat('Flash attn', p.flash_attn),
        stat('GPU layers', `${Math.min(p.gpu_layers, p.n_layer + 1)}/${p.n_layer + 1}`),
        stat('VRAM', `${fmt.mib(vramUsed)} / ${fmt.mib(vramFree)}`, 'Planned use vs. free VRAM after safety margins'),
        stat('Max full-offload ctx', p.max_ctx_full_offload?.f16 !== undefined ? `${fmt.ctx(p.max_ctx_full_offload.f16)} F16 · ${fmt.ctx(p.max_ctx_full_offload.q8_0)} Q8_0` : '-',
          'Largest context that still fits entirely in VRAM with each KV cache type'),
        stat('Load mode', p.load_mode));
      bars.update(planDevices(p), { host: p.host });
      lmap.update(p);
      clear(planNotes);
      for (const w of p.warnings) planNotes.appendChild(h('div', { class: 'note warn' }, w));
      for (const n of p.notes) planNotes.appendChild(h('div', { class: 'note' }, n));
      if (r.device_error) planNotes.appendChild(h('div', { class: 'note warn' }, `Device query: ${r.device_error}`));
      if (r.evicting?.length) planNotes.appendChild(h('div', { class: 'note' }, `Loading will unload: ${r.evicting.join(', ')}`));
      if (!r.engine) planNotes.appendChild(h('div', { class: 'note err' }, 'No llama.cpp engine installed. Settings › Engine › Download.'));
      setText(cmdBox, r.command || '(no engine)');
    } catch (e) {
      if (seq === planSeq) { clear(planStatus); planStatus.append(h('span', { class: 'err' }, e.message)); }
    } finally {
      planStatus.classList.remove('busy');
      if (verify) { verifyBtn.disabled = false; verifyBtn.querySelector('span').textContent = 'Verify with engine'; }
    }
  }

  async function loadModel() {
    if (!detail) return;
    try {
      const overrides = formValues ? diffFromDefaults(formValues, detail.defaults, true) : null;
      await api.post(wr('/models/load'), { id: detail.id, overrides });
      toast(`Loading ${detail.id}...`, 'info', 2500);
    } catch (e) {
      toast(e.message, 'err', 6000, 'Load failed');
    }
  }

  // ---- PROPERTIES ------------------------------------------------------------------------------
  function renderProps(body) {
    const i = detail.info;
    const hkv = Array.isArray(i.n_head_kv) ? `${Math.min(...i.n_head_kv)}-${Math.max(...i.n_head_kv)} (per layer)` : i.n_head_kv;
    const main = kv([
      ['Architecture', i.architecture], ['Name', i.name || '-'], ['Organization', i.organization || '-'],
      ['Parameters', `${fmt.params(i.n_params)} (${fmt.num(i.n_params)})`], ['Size label', i.size_label || '-'],
      ['Quantization', `${i.quant}${i.file_type !== null ? ` (file type ${i.file_type})` : ''}`],
      ['File size', `${fmt.bytes(i.file_size, 2)}${i.split_files?.length ? ` in ${i.split_files.length} parts` : ''}`],
      ['Bits per weight', i.n_params ? (i.weights_bytes * 8 / i.n_params).toFixed(2) : '-'],
      ['Layers', i.n_layer], ['Embedding size', fmt.num(i.n_embd)], ['Feed-forward size', fmt.num(i.n_ff)],
      ['Attention heads', `${i.n_head} query · ${hkv} KV`], ['Head dimension', `${i.head_dim_k} K · ${i.head_dim_v} V`],
      ['Vocabulary', `${fmt.num(i.n_vocab)} (${i.tokenizer_model || '?'})`], ['Trained context', fmt.num(i.context_length)],
      ['RoPE', `${i.rope_freq_base ? `base ${fmt.num(i.rope_freq_base)}` : '-'}${i.rope_scaling ? ` · ${i.rope_scaling}${i.rope_scaling_factor ? ` ×${i.rope_scaling_factor}` : ''}` : ''}`],
      ['Experts', i.expert_count ? `${i.expert_used_count} of ${i.expert_count} active` : 'dense'],
      ['Sliding window', i.sliding_window ? `${i.sliding_window} tokens` : 'none'],
      ['Special tokens', `BOS ${i.bos_token ?? '-'} · EOS ${i.eos_token ?? '-'}${i.eot_token ? ` · EOT ${i.eot_token}` : ''}${i.add_bos !== null ? ` · add BOS: ${i.add_bos}` : ''}`],
      ['GGUF version', i.gguf_version], ['Path', h('code', null, i.path)],
    ], 'kv');
    const types = Object.entries(i.tensor_types || {}).sort((a, b) => b[1] - a[1]);
    const tot = types.reduce((a, [, b]) => a + b, 0) || 1;
    const colors = ['--c-1', '--c-2', '--c-3', '--c-4', '--c-5', '--c-other'];
    const typeBar = h('div', { class: 'membar' }, types.map(([t, b], k) => h('div', { class: 'mseg', 'data-tip': `${t}: ${fmt.bytes(b)}`,
      style: { width: `${(b / tot) * 100}%`, background: `var(${colors[k % colors.length]})` } })));
    const typeLegend = h('div', { class: 'membar-legend' }, types.map(([t, b], k) => h('span', null,
      h('i', { style: { background: `var(${colors[k % colors.length]})` } }), `${t} ${((b / tot) * 100).toFixed(1)}%`)));
    const samp = Object.entries(i.sampling || {});
    const mdFilter = input('', (v) => renderMd(v), { placeholder: 'Filter keys...', live: true, cls: 'grow' });
    const mdBody = h('tbody');
    const renderMd = (q = '') => {
      clear(mdBody);
      for (const [k, v] of Object.entries(detail.metadata || {})) {
        if (q && !k.toLowerCase().includes(q.toLowerCase())) continue;
        mdBody.appendChild(h('tr', null, h('td', { class: 'mono', style: { whiteSpace: 'nowrap' } }, k),
          h('td', { class: 'mono', style: { overflowWrap: 'anywhere' } }, typeof v === 'object' ? JSON.stringify(v) : String(v))));
      }
    };
    renderMd();
    body.append(
      group('Model', main, { icon: 'model' }),
      group('Tensor types', [typeBar, typeLegend], { icon: 'mem' }),
      samp.length ? group('Recommended sampling (GGUF)', [kv(samp.map(([k, v]) => [k, String(v)])),
        h('div', { class: 'dim', style: { marginTop: '6px' } }, 'The engine applies these as defaults for requests that do not set them.')], { icon: 'sliders' }) : null,
      group('GGUF metadata', [h('div', { class: 'toolbar' }, icon('search'), mdFilter),
        h('div', { class: 'tbl-wrap', style: { maxHeight: '420px' } }, h('table', { class: 'tbl compact' }, h('thead', null, h('tr', null, h('th', null, 'Key'), h('th', null, 'Value'))), mdBody))],
      { icon: 'list', collapsible: true, collapsed: true }),
    );
  }

  // ---- TEMPLATE ------------------------------------------------------------------------------
  function renderTemplate(body) {
    const d = detail;
    const a = d.template_analysis || {};
    const inst = instFor();
    const chips = h('div', { class: 'row wrap' },
      h('span', { class: 'chip' }, `Format: ${a.family}`),
      a.tools ? h('span', { class: 'badge tools' }, icon('tool'), 'tool calls') : h('span', { class: 'chip dim' }, 'no tool calls'),
      a.reasoning ? h('span', { class: 'badge reason' }, icon('think'), 'reasoning') : null,
      a.thinking_toggle ? h('span', { class: 'chip', 'data-tip': 'Template accepts enable_thinking (chat_template_kwargs)' }, 'enable_thinking') : null,
      a.reasoning_effort ? h('span', { class: 'chip' }, 'reasoning_effort') : null,
      a.system_role ? h('span', { class: 'chip' }, 'system role') : h('span', { class: 'chip warn' }, 'no system role'),
      h('span', { class: 'chip dim' }, `${fmt.num(a.length)} chars`));
    const status = [];
    if (inst?.state === 'ready') {
      if (inst.params?.chat_template_mode === 'gguf') {
        status.push(h('div', { class: `note ${inst.template_verified === false ? 'warn' : 'ok'}` },
          inst.template_verified === false ? 'The engine reports a different template than the GGUF (override or engine fallback).'
            : 'Verified: the running engine uses the exact Jinja template embedded in this GGUF.'));
      }
      const caps = inst.template_caps || {};
      if (Object.keys(caps).length) {
        status.push(h('div', { class: 'row wrap', style: { margin: '6px 0' } }, h('span', { class: 'dim' }, 'Engine capabilities:'),
          Object.entries(caps).map(([k, v]) => h('span', { class: `chip ${v ? '' : 'dim'}` }, `${v ? '✓' : '×'} ${k.replace('supports_', '').replace(/_/g, ' ')}`))));
      }
    }
    const code = h('pre', { class: 'code', style: { maxHeight: '420px' } });
    code.innerHTML = highlightJinja(d.chat_template || '(no chat template in this GGUF - the engine will use a generic fallback)');
    const named = Object.keys(d.named_templates || {});
    const sys = toggle('System message', true, () => {});
    const gen = toggle('Add generation prompt', true, () => {});
    const kwargs = input('', null, { cls: 'mono grow', placeholder: 'Template arguments JSON, e.g. {"enable_thinking": false}' });
    const out = h('pre', { class: 'code wrap', style: { maxHeight: '360px' } }, 'Press Render to see the exact prompt text for a sample conversation.');
    const src = h('span', { class: 'dim' });
    const render = async () => {
      let kw = {};
      if (kwargs.value.trim()) { try { kw = JSON.parse(kwargs.value); } catch { toast('Template arguments must be valid JSON', 'err'); return; } }
      const msgs = [
        ...(sys.input.checked ? [{ role: 'system', content: 'You are a helpful assistant.' }] : []),
        { role: 'user', content: 'What is the capital of France?' },
        { role: 'assistant', content: 'The capital of France is Paris.' },
        { role: 'user', content: 'And of Italy?' },
      ];
      try {
        const r = await api.post(wr('/template/preview'), { id: d.id, messages: msgs, kwargs: kw });
        if (r.error) { out.textContent = `Template error: ${r.error}`; setText(src, ''); return; }
        out.innerHTML = markSpecial(r.prompt + (gen.input.checked ? '' : ''));
        setText(src, r.source === 'engine' ? 'rendered by the running engine (/apply-template)' : 'rendered locally with Jinja2 (sandbox)');
      } catch (e) { out.textContent = e.message; }
    };
    body.append(
      group('Template', [chips, ...status, named.length ? h('div', { class: 'dim', style: { margin: '4px 0' } }, `Named templates in GGUF: ${named.join(', ')}`) : null,
        h('div', { class: 'row', style: { margin: '6px 0' } }, copyBtn(d.chat_template || '', { label: 'Copy template', small: true }),
          btn('Use as custom template', async () => {
            await api.put(wr('/library/profile'), { id: d.id, load: { ...d.profile.load, chat_template_mode: 'custom', chat_template_custom: d.chat_template } });
            toast('Custom template created from the GGUF template. Edit it under Load > Chat template.', 'ok');
            select_(d.id);
          }, { cls: 'small', icon: 'copy' })), code], { icon: 'chat' }),
      group('Prompt preview', [h('div', { class: 'row wrap', style: { marginBottom: '6px' } }, sys, kwargs, btn('Render', render, { icon: 'play', cls: 'primary' })), out, src],
        { icon: 'log' }));
  }

  // ---- VISION ------------------------------------------------------------------------------
  function renderVision(body) {
    const d = detail;
    const cands = d.mmproj_candidates || [];
    if (!cands.length) {
      body.append(group('Vision projector', [
        h('div', { class: 'note warn' }, 'No multimodal projector (mmproj) file was found next to this model, so image input is not available.'),
        h('p', null, 'Vision models on llama.cpp consist of the language model GGUF plus a separate projector GGUF (usually named mmproj-*.gguf) from the same repository. Place it in the model\'s folder:'),
        h('pre', { class: 'code wrap' }, d.path.replace(/[^\\/]+$/, '')),
        btn('Find projector on Hugging Face', () => openDownloads(guessRepo(d)), { icon: 'download' })], { icon: 'eye' }));
      return;
    }
    const current = d.profile.load?.mmproj || '';
    const tb = h('tbody');
    for (const p of cands) {
      const mm = d.mmproj_details?.[p] || {};
      const compat = mm.projection_dim && d.info.n_embd ? Number(mm.projection_dim) === Number(d.info.n_embd) : null;
      const isDefault = (current || d.mmproj) === p;
      tb.appendChild(h('tr', { class: isDefault ? 'sel' : '' },
        h('td', null, h('input', { type: 'radio', name: 'mmproj', checked: isDefault, onchange: async () => {
          await api.put(wr('/library/profile'), { id: d.id, load: { ...d.profile.load, mmproj: p === d.mmproj ? '' : p } });
          toast('Vision projector selection saved', 'ok');
          select_(d.id);
        } })),
        h('td', null, p.split(/[\\/]/).pop()), h('td', null, mm.projector_type || '-'),
        h('td', null, [mm.has_vision ? 'image' : null, mm.has_audio ? 'audio' : null].filter(Boolean).join(' + ') || '-'),
        h('td', { class: 'num' }, mm.image_size ? `${mm.image_size}px / ${mm.patch_size}` : '-'),
        h('td', { class: 'num' }, mm.projection_dim ?? '-'),
        h('td', null, compat === null ? h('span', { class: 'dim' }, 'n/a') : compat ? h('span', { class: 'ok' }, '✓ matches') : h('span', { class: 'err', 'data-tip': `Projector output ${mm.projection_dim} ≠ model embedding ${d.info.n_embd}` }, '× mismatch'))));
    }
    const inst = instFor();
    const li = inst?.load?.mmproj || {};
    body.append(
      group('Vision projector', [
        h('div', { class: 'tbl-wrap' }, h('table', { class: 'tbl' }, h('thead', null, h('tr', null, h('th'), h('th', null, 'File'), h('th', null, 'Projector'),
          h('th', null, 'Modalities'), h('th', { class: 'num' }, 'Image / patch'), h('th', { class: 'num' }, 'Proj. dim'), h('th', null, 'Model match'))), tb)),
        h('div', { class: 'dim', style: { marginTop: '6px' } }, `The projector's output dimension must equal the language model's embedding size (${d.info.n_embd}).`),
        inst?.state === 'ready' ? kv([['Loaded projector', inst.mmproj ? inst.mmproj.split(/[\\/]/).pop() : 'none'], ['Projector type', li.projector || '-'],
          ['Encoder backend', li.backend || '-'], ['Worst-case memory', li.est_mib ? fmt.mib(li.est_mib) : '-'], ['Vision active', inst.vision ? 'yes' : 'no']]) : null,
      ], { icon: 'eye' }),
      group('Image input handling', h('div', null,
        h('p', null, 'Images sent to /v1/chat/completions (image_url parts with base64 data URIs or http(s) URLs), /v1/responses (input_image) and /v1/messages (image blocks) are normalised before they reach the engine:'),
        h('ul', { class: 'bullets' },
          h('li', null, 'WebP, TIFF, HEIC/AVIF (with pillow-heif) and other formats are converted to PNG/JPEG.'),
          h('li', null, 'EXIF orientation from phone cameras is applied; CMYK images are converted to RGB.'),
          h('li', null, 'Remote URLs are downloaded by WinRunner (can be disabled in Settings › Network).'),
          h('li', null, 'Requests with images for a model without a projector are rejected with a clear error.'))), { icon: 'image', collapsible: true }),
    );
  }

  // ---- PRESET ------------------------------------------------------------------------------
  function renderPreset(body) {
    const d = detail;
    const s = { ...(d.profile.sampling || {}) };
    const alias = input(d.profile.alias || '', null, { placeholder: 'e.g. local-chat', cls: 'mono' });
    const f = h('div', { class: 'form' });
    const fields = [
      ['temperature', 'Temperature', 0.01, 'Randomness. Lower = more deterministic.'], ['top_p', 'Top P', 0.01, 'Nucleus sampling threshold.'],
      ['top_k', 'Top K', 1, 'Sample from the K most likely tokens (0 = off).'], ['min_p', 'Min P', 0.01, 'Minimum probability relative to the top token.'],
      ['repeat_penalty', 'Repeat penalty', 0.01, '1.0 = off.'], ['repeat_last_n', 'Repeat window', 1, 'Tokens considered for the repeat penalty.'],
      ['presence_penalty', 'Presence penalty', 0.01, ''], ['frequency_penalty', 'Frequency penalty', 0.01, ''],
      ['max_tokens', 'Max tokens', 1, 'Default response length limit (empty = until the model stops or the context is full).'],
      ['seed', 'Seed', 1, 'Fixed seed for reproducible sampling.'],
    ];
    const eng = instFor()?.generation_defaults || {};
    const gg = d.info.sampling || {};
    const inputs = {};
    for (const [k, label, step, hint] of fields) {
      const ph = eng[k] !== undefined ? `engine: ${typeof eng[k] === 'number' ? +eng[k].toFixed(3) : eng[k]}` : gg[k.replace('temperature', 'temp')] !== undefined ? `GGUF: ${gg[k.replace('temperature', 'temp')]}` : 'engine default';
      inputs[k] = input(s[k] ?? '', null, { type: 'number', cls: 'num', step, placeholder: ph, style: { width: '140px' } });
      f.append(h('label', null, label), h('div', { class: 'ctl' }, inputs[k], hint ? h('span', { class: 'dim' }, hint) : null));
    }
    body.append(
      group('Model alias', [h('div', { class: 'row' }, alias, h('span', { class: 'dim' }, `Extra name clients may use for this model besides "${d.id}".`))], { icon: 'key' }),
      group('API sampling preset', [
        h('p', { class: 'dim' }, 'Applied to API requests that do not specify these values. Leave empty to keep the model\'s own defaults (GGUF recommended values, then engine defaults).'),
        f,
        h('div', { class: 'row', style: { marginTop: '10px' } }, btn('Save preset', async () => {
          const out = {};
          for (const [k] of fields) { const v = inputs[k].value; if (v !== '') out[k] = Number(v); }
          await api.put(wr('/library/profile'), { id: d.id, alias: alias.value.trim(), sampling: out });
          toast('Preset saved', 'ok');
          select_(d.id);
        }, { icon: 'check', cls: 'primary' }), btn('Clear', () => { for (const k in inputs) inputs[k].value = ''; }))],
      { icon: 'sliders' }),
    );
  }

  // ------------------------------------------------------------------------------------
  offs.push(store.on('library_scan', (ev) => { if (ev.state === 'done') loadList(); }));
  offs.push(store.on('instance', ({ inst }) => {
    renderList();
    if (detail && inst.model === detail.id) {
      if (inst.state !== lastInstState) { lastInstState = inst.state; renderDetail(); } else renderActual();
    }
  }));
  offs.push(store.on('instance_progress', (ev) => { if (detail && ev.model === detail.id) renderActual(); }));
  offs.push(store.on('instance_removed', (ev) => {
    renderList();
    if (detail && ev.model === detail.id) { lastInstState = null; renderDetail(); }
  }));
  loadList();
  return () => offs.forEach((f) => f());
}

function guessRepo(d) {
  const pub = d.publisher && d.publisher !== 'models' ? d.publisher : '';
  return pub && d.repo ? `${pub}/${d.repo}` : d.name;
}

/** Only keep values that differ from global defaults (so profiles stay minimal). */
function diffFromDefaults(values, defaults, all = false) {
  const out = {};
  for (const [k, v] of Object.entries(values || {})) {
    if (all || JSON.stringify(v) !== JSON.stringify(defaults?.[k])) out[k] = v;
  }
  return out;
}

function highlightJinja(src) {
  const esc = escapeHtml(src);
  const lines = esc.split('\n');
  return lines.map((ln, i) => {
    let s = ln
      .replace(/(\{#.*?#\})/g, '<span class="j-cmt">$1</span>')
      .replace(/(\{%-?|-?%\}|\{\{-?|-?\}\})/g, '<span class="j-tag">$1</span>')
      .replace(/(&#39;[^&]*?&#39;|&quot;[^&]*?&quot;)/g, '<span class="j-str">$1</span>')
      .replace(/\b(if|elif|else|endif|for|endfor|in|set|not|and|or|is|macro|endmacro|namespace|defined|none|true|false)\b/g, '<span class="j-kw">$1</span>');
    return `<span class="ln">${i + 1}</span>${s}`;
  }).join('\n');
}

function markSpecial(text) {
  return escapeHtml(text).replace(/(&lt;\|[^|]{1,40}?\|&gt;|&lt;\/?(?:s|think|start_of_turn|end_of_turn|bos|eos)&gt;|\[\/?INST\]|&lt;｜[^｜]{1,30}｜&gt;)/g, '<span class="special">$1</span>');
}

// ------------------------------------------------------------------------------------
// Downloads dialog (Hugging Face)
// ------------------------------------------------------------------------------------
export function openDownloads(query = '') {
  const q = input(query, null, { placeholder: 'Search Hugging Face (e.g. qwen3 gguf, gemma-3 12b, unsloth)', cls: 'grow', onEnter: () => doSearch() });
  const results = h('div', { class: 'tbl-wrap', style: { maxHeight: '240px' } });
  const files = h('div');
  const jobs = h('div', { class: 'dl-jobs' });
  const body = h('div', { class: 'col' }, h('div', { class: 'row' }, icon('search'), q, btn('Search', () => doSearch(), { cls: 'primary' })), results, files,
    h('div', { class: 'form-section' }, 'Downloads'), jobs);
  const m = modal('Download models from Hugging Face', body, { wide: true, onClose: () => off() });
  async function doSearch() {
    clear(results);
    results.appendChild(h('div', { class: 'dim', style: { padding: '8px' } }, 'Searching...'));
    try {
      if (/^[\w.-]+\/[\w.-]+$/.test(q.value.trim())) { clear(results); return showRepo(q.value.trim()); }
      const r = await api.get(wr(`/hf/search?q=${encodeURIComponent(q.value)}`));
      clear(results);
      const tb = h('tbody');
      for (const x of r.results) {
        tb.appendChild(h('tr', { onclick: () => showRepo(x.repo), style: { cursor: 'pointer' } },
          h('td', null, h('b', null, x.repo)), h('td', { class: 'num' }, fmt.num(x.downloads)), h('td', { class: 'num' }, fmt.num(x.likes)),
          h('td', null, x.pipeline || ''), h('td', { class: 'dim' }, x.updated ? x.updated.slice(0, 10) : '')));
      }
      results.appendChild(h('table', { class: 'tbl' }, h('thead', null, h('tr', null, h('th', null, 'Repository'), h('th', { class: 'num' }, 'Downloads'),
        h('th', { class: 'num' }, 'Likes'), h('th', null, 'Type'), h('th', null, 'Updated'))), tb));
      if (!r.results.length) results.appendChild(h('div', { class: 'dim', style: { padding: '8px' } }, 'No GGUF repositories found.'));
    } catch (e) { clear(results); results.appendChild(h('div', { class: 'note err' }, e.message)); }
  }
  async function showRepo(repo) {
    clear(files);
    files.appendChild(h('div', { class: 'dim' }, `Listing ${repo}...`));
    try {
      const r = await api.get(wr(`/hf/files?repo=${encodeURIComponent(repo)}`));
      clear(files);
      const mm = r.files.filter((f) => f.mmproj);
      const tb = h('tbody');
      const groups = new Map();
      for (const f of r.files.filter((x) => !x.mmproj)) {
        const key = f.split ? f.path.replace(/-\d{5}-of-\d{5}\.gguf$/i, '') : f.path;
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(f);
      }
      const withMm = toggle(`Also download vision projector (${mm[0]?.name || ''})`, mm.length > 0, null, { disabled: !mm.length });
      for (const [key, parts] of groups) {
        const size = parts.reduce((a, b) => a + (b.size || 0), 0);
        tb.appendChild(h('tr', null, h('td', null, key.split('/').pop()), h('td', null, parts[0].quant), h('td', { class: 'num' }, fmt.bytes(size)),
          h('td', null, parts.length > 1 ? `${parts.length} parts` : ''),
          h('td', null, btn('Download', async () => {
            const list = parts.map((p) => p.path);
            const sizes = Object.fromEntries(parts.map((p) => [p.path, p.size]));
            if (withMm.input.checked && mm.length) { list.push(mm[0].path); sizes[mm[0].path] = mm[0].size; }
            await api.post(wr('/downloads'), { repo, files: list, sizes });
            toast(`Downloading ${key.split('/').pop()}`, 'ok');
          }, { cls: 'small', icon: 'download' }))));
      }
      files.append(h('div', { class: 'row', style: { margin: '6px 0' } }, h('b', null, repo), h('span', { class: 'spacer' }), withMm),
        mm.length ? null : h('div', { class: 'dim' }, 'No vision projector (mmproj) files in this repository.'),
        h('div', { class: 'tbl-wrap', style: { maxHeight: '260px' } }, h('table', { class: 'tbl' }, h('thead', null, h('tr', null, h('th', null, 'File'),
          h('th', null, 'Quant'), h('th', { class: 'num' }, 'Size'), h('th'), h('th'))), tb)));
    } catch (e) { clear(files); files.appendChild(h('div', { class: 'note err' }, e.message)); }
  }
  function renderJobs() {
    clear(jobs);
    const list = [...store.downloads.values()];
    if (!list.length) { jobs.appendChild(h('div', { class: 'dim' }, 'No downloads.')); return; }
    for (const j of list) {
      const pct = j.total ? (j.done / j.total) * 100 : 0;
      const bar = h('div', { class: `progress ${j.state === 'queued' ? 'indeterminate' : ''}` }, h('div', { class: 'fill', style: { width: `${pct}%` } }));
      jobs.appendChild(h('div', { class: 'dl-job' }, h('div', { class: 'row' }, h('b', { class: 'ellipsis grow' }, j.path.split('/').pop()),
        h('span', { class: j.state === 'error' ? 'err' : j.state === 'done' ? 'ok' : 'dim' },
          j.state === 'downloading' ? `${fmt.bytes(j.done)} / ${fmt.bytes(j.total)} · ${fmt.bytes(j.speed)}/s` : j.state + (j.error ? `: ${j.error}` : '')),
        btn('', async () => { await api.del(wr(`/downloads/${j.id}`)); store.downloads.delete(j.id); renderJobs(); }, { icon: 'x', cls: 'icon small', tip: j.state === 'downloading' ? 'Cancel' : 'Remove' })), bar));
    }
  }
  const off = store.on('download', renderJobs);
  renderJobs();
  if (query) doSearch();
  return m;
}

// ------------------------------------------------------------------------------------
// Folders dialog
// ------------------------------------------------------------------------------------
export async function openFolders() {
  const list = h('div', { class: 'col' });
  const path = input('', null, { placeholder: 'C:\\Models or D:\\LLM\\gguf', cls: 'grow mono' });
  const body = h('div', { class: 'col' }, h('p', { class: 'dim' }, 'Folders are scanned recursively for .gguf files. LM Studio\'s model folder layout (publisher/repository/file.gguf) is supported.'),
    list, h('div', { class: 'row' }, path, btn('Add folder', async () => {
      try { await api.post(wr('/library/folder'), { path: path.value.trim(), action: 'add' }); path.value = ''; await render(); } catch (e) { toast(e.message, 'err'); }
    }, { icon: 'plus', cls: 'primary' })));
  async function render() {
    const d = await api.get(wr('/library'));
    clear(list);
    for (const x of d.dirs) {
      list.appendChild(h('div', { class: 'row folder-row' }, h('span', { class: `led ${x.exists ? 'ok' : ''}`, 'data-tip': x.exists ? 'Folder exists' : 'Folder not found' }),
        h('code', { class: 'grow ellipsis' }, x.path), btn('', async () => { await api.post(wr('/library/folder'), { path: x.path, action: 'remove' }); render(); },
          { icon: 'trash', cls: 'icon small', tip: 'Remove from library (files are not deleted)' })));
    }
    list.appendChild(h('div', { class: 'dim' }, `Downloads are saved to: ${d.download_dir}`));
  }
  modal('Model folders', body);
  render();
}

