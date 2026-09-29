// SETTINGS: appearance, network, engine, hardware, model defaults, storage, startup.

import { h, clear, icon, debounce } from '../core/dom.js';
import { api, wr } from '../core/api.js';
import { store } from '../core/store.js';
import * as fmt from '../core/fmt.js';
import { THEMES, EDITABLE, currentTokens, toHex, applyUI } from '../core/theme.js';
import { group, btn, seg, toggle, select, input, toast, kv, confirmBox, formRow, slideNum, copyBtn } from '../components/ui.js';
import { loadForm } from '../components/loadform.js';
import { openFolders } from './library.js';

export function mount(root) {
  const offs = [];
  const S = () => store.settings;
  const save = async (patch, msg = 'Settings saved') => {
    try { await store.saveSettings(patch); toast(msg, 'ok', 1300); } catch (e) { toast(e.message, 'err', 6000); }
  };
  const sections = [['appearance', 'Appearance'], ['network', 'Network'], ['engine', 'Engine'], ['hardware', 'Hardware'],
    ['defaults', 'Model defaults'], ['storage', 'Storage'], ['startup', 'Startup'], ['about', 'About']];
  const hosts = Object.fromEntries(sections.map(([k]) => [k, h('div', { id: `set-${k}` })]));
  root.append(h('div', { class: 'pg-head' }, h('h2', null, 'Settings'), h('span', { class: 'sub' }, 'Saved automatically')),
    h('div', { class: 'settings-nav' }, sections.map(([k, l]) => btn(l, () => hosts[k].scrollIntoView({ behavior: 'smooth', block: 'start' }), { cls: 'small' }))),
    ...Object.values(hosts));

  // ================= APPEARANCE =================
  function renderAppearance() {
    const host = hosts.appearance;
    clear(host);
    const ui = S().ui;
    const cards = h('div', { class: 'theme-cards' });
    const all = [...THEMES.map((t) => ({ ...t, key: t.id })), ...Object.entries(ui.custom_themes || {}).map(([name, c]) => ({
      key: `custom:${name}`, name, desc: `Custom theme based on ${c.base || 'classic'}`, sw: [c['--bg-1'], c['--bg-2'], c['--acc'], c['--tx-0']] }))];
    for (const t of all) {
      cards.appendChild(h('div', { class: `theme-card ${ui.theme === t.key ? 'on' : ''}`, onclick: () => save({ ui: { theme: t.key } }, `Theme: ${t.name}`) },
        h('div', { class: 'sw' }, t.sw.map((c) => h('i', { style: { background: c } }))), h('div', { class: 'nm' }, t.name), h('div', { class: 'ds' }, t.desc)));
    }
    // custom theme editor
    const cur = currentTokens();
    const base = select(THEMES.map((t) => [t.id, t.name]), ui.theme.startsWith('custom:') ? (ui.custom_themes?.[ui.theme.slice(7)]?.base || 'classic') : ui.theme);
    const name = input(ui.theme.startsWith('custom:') ? ui.theme.slice(7) : '', null, { placeholder: 'Theme name' });
    const colors = {};
    const grid = h('div', { class: 'token-grid' });
    for (const [k, label] of EDITABLE) {
      colors[k] = h('input', { type: 'color', class: 'field', value: toHex(cur[k] || '#808080') });
      colors[k].addEventListener('input', () => preview());
      grid.appendChild(h('label', null, colors[k], label));
    }
    const collect = () => ({ base: base.value, ...Object.fromEntries(Object.entries(colors).map(([k, el]) => [k, el.value])) });
    const preview = () => applyUI({ ...S().ui, theme: 'custom:__preview', custom_themes: { __preview: collect() } });
    const editor = group('Custom theme', [
      h('div', { class: 'row wrap', style: { marginBottom: '8px' } }, h('span', { class: 'dim' }, 'Base'), base, h('span', { class: 'dim' }, 'Name'), name,
        btn('Load colours from base', () => { applyUI({ ...S().ui, theme: base.value }); const c = currentTokens(); for (const [k] of EDITABLE) colors[k].value = toHex(c[k]); preview(); }, { cls: 'small' })),
      grid,
      h('div', { class: 'row', style: { marginTop: '8px' } },
        btn('Save theme', () => {
          const n = name.value.trim();
          if (!/^[\w -]{1,32}$/.test(n)) { toast('Enter a theme name (letters, numbers, spaces)', 'warn'); return; }
          save({ ui: { theme: `custom:${n}`, custom_themes: { ...(S().ui.custom_themes || {}), [n]: collect() } } }, `Theme "${n}" saved`);
        }, { icon: 'check', cls: 'primary' }),
        btn('Revert preview', () => applyUI(S().ui), { icon: 'refresh' }),
        ui.theme.startsWith('custom:') ? btn('Delete this theme', async () => {
          const n = ui.theme.slice(7);
          const ct = { ...(S().ui.custom_themes || {}) };
          delete ct[n];
          await store.saveSettings({ ui: { theme: 'classic', custom_themes: ct } });
          render();
        }, { icon: 'trash', cls: 'danger' }) : null),
    ], { icon: 'sliders', collapsible: true, collapsed: true });
    const f = h('div', { class: 'form' });
    formRow(f, 'Display effects', h('div', { class: 'col', style: { gap: '4px' } },
      toggle('CRT scanlines and vignette', ui.crt_effect, (v) => save({ ui: { crt_effect: v } })),
      toggle('Glow on readouts and graphs', ui.glow, (v) => save({ ui: { glow: v } })),
      toggle('Start-up sequence', ui.boot_sequence, (v) => save({ ui: { boot_sequence: v } })),
      toggle('Show token boundaries in the token stream', ui.token_boundaries, (v) => save({ ui: { token_boundaries: v } })),
      toggle('24-hour clock', ui.clock_24h, (v) => save({ ui: { clock_24h: v } }))));
    formRow(f, 'Animation', seg([['full', 'Full'], ['reduced', 'Reduced'], ['off', 'Off']], ui.animations, (v) => save({ ui: { animations: v } })),
      'Reduced/Off lower CPU and GPU use of the control panel.');
    formRow(f, 'Interface scale', slideNum(ui.ui_scale, { min: 0.8, max: 1.6, step: 0.05, width: 70 }, (v) => save({ ui: { ui_scale: Math.round(v * 100) / 100 } })),
      'Scale text and controls (1.0 = 12 px base). Useful on high-DPI vertical monitors.');
    formRow(f, 'Activity panel', seg([['right', 'Right column'], ['bottom', 'Bottom']], ui.sidebar_position, (v) => save({ ui: { sidebar_position: v } })),
      'Right column suits portrait monitors wider than ~1000 px; bottom suits narrow windows.');
    host.append(group('Appearance', [cards, h('div', { class: 'divider' }), f], { icon: 'sliders' }), editor);
  }

  // ================= NETWORK =================
  function renderNetwork() {
    const host = hosts.network;
    clear(host);
    const s = S().server;
    const f = h('div', { class: 'form' });
    const hostSel = select([['0.0.0.0', '0.0.0.0 - all network interfaces (LAN access)'], ['127.0.0.1', '127.0.0.1 - this computer only']].concat(
      ['0.0.0.0', '127.0.0.1'].includes(s.host) ? [] : [[s.host, s.host]]), s.host, (v) => save({ server: { host: v } }, 'Bind address saved - restart WinRunner to apply'));
    formRow(f, 'Bind address', hostSel);
    formRow(f, 'Port', input(s.port, (v) => save({ server: { port: Number(v) } }, 'Port saved - restart WinRunner to apply'), { type: 'number', cls: 'num', min: 1, max: 65535 }),
      'Default 5070. Endpoint: http://<this-pc>:<port>/v1. Changes apply after a restart.');
    formRow(f, 'API', h('div', { class: 'col', style: { gap: '4px' } },
      toggle('API server running', s.api_enabled, (v) => save({ server: { api_enabled: v } })),
      toggle('Just-in-time model loading (load the model named in a request)', s.jit_loading, (v) => save({ server: { jit_loading: v } })),
      toggle('Unload other models when loading a model', s.auto_evict, (v) => save({ server: { auto_evict: v } })),
      toggle('Allow cross-origin (CORS) requests from web apps', s.cors, (v) => save({ server: { cors: v } })),
      toggle('Stream non-streaming requests internally (live telemetry for all clients)', s.internal_stream_aggregation, (v) => save({ server: { internal_stream_aggregation: v } }))));
    formRow(f, 'Idle unload', h('div', { class: 'row' }, input(s.idle_unload_minutes, (v) => save({ server: { idle_unload_minutes: Number(v) || 0 } }), { type: 'number', cls: 'num', min: 0 }),
      h('span', { class: 'dim' }, 'minutes (0 = never)')), 'Free VRAM when no request arrived for this long.');
    const keyBox = h('div', { class: 'row wrap' },
      h('code', null, s.api_key ? `${s.api_key.slice(0, 7)}…${s.api_key.slice(-4)}` : 'none (open access)'),
      btn(s.api_key ? 'Regenerate' : 'Generate key', () => {
        const k = `wr-${Array.from(crypto.getRandomValues(new Uint8Array(18))).map((b) => b.toString(16).padStart(2, '0')).join('')}`;
        save({ server: { api_key: k } }, 'API key set');
      }, { cls: 'small', icon: 'key' }),
      s.api_key ? copyBtn(s.api_key, { label: 'Copy' }) : null,
      s.api_key ? btn('Remove', () => save({ server: { api_key: '' } }, 'API key removed'), { cls: 'small danger' }) : null);
    formRow(f, 'API key', keyBox, 'Clients send it as "Authorization: Bearer <key>". The local control panel does not need it.');
    formRow(f, 'Control panel', toggle('Allow the control panel from other computers', s.lan_control_panel, (v) => save({ server: { lan_control_panel: v } })),
      'Off: only this PC can open the control panel (the API stays available on the network).');
    formRow(f, 'Images', h('div', { class: 'col', style: { gap: '4px' } },
      toggle('Download remote image URLs sent by clients', s.fetch_remote_images, (v) => save({ server: { fetch_remote_images: v } })),
      h('div', { class: 'row' }, h('span', { class: 'dim' }, 'Downscale images larger than'), input(s.max_image_edge, (v) => save({ server: { max_image_edge: Number(v) || 0 } }),
        { type: 'number', cls: 'num', min: 0, step: 64 }), h('span', { class: 'dim' }, 'px (0 = keep original)'))));
    formRow(f, 'Request history', input(s.request_history, (v) => save({ server: { request_history: Number(v) } }, 'Saved - applies after restart'), { type: 'number', cls: 'num', min: 50, max: 10000 }));
    host.append(group('Network and API', f, { icon: 'globe' }));
  }

  // ================= ENGINE =================
  const installBar = h('div', { class: 'progress' }, h('div', { class: 'fill' }));
  const installTxt = h('div', { class: 'dim' });
  async function renderEngine() {
    const host = hosts.engine;
    clear(host);
    let d;
    try { d = await api.get(wr('/engine')); } catch (e) { host.appendChild(h('div', { class: 'note err' }, e.message)); return; }
    const a = d.active;
    const es = S().engine;
    const activeBox = a ? h('div', null,
      kv([['Engine', `llama.cpp ${a.version}${a.build ? ` (build ${a.build})` : ''}${a.commit ? ` · ${a.commit}` : ''}`], ['Backend', a.backend_label],
        ['Executable', h('code', null, a.path)], ['Options detected', `${a.flag_count} command-line options`],
        ['Memory projection', a.fit_params ? 'llama-fit-params available' : 'not included in this build'],
        ['Benchmark tool', a.bench ? 'llama-bench available' : 'not included'], ['Flash attention flag', a.fa_tristate ? 'on / off / auto' : 'boolean (older build)']]),
      a.probe_error ? h('div', { class: 'note err' }, a.probe_error) : null,
      h('div', { class: 'row', style: { marginTop: '8px' } }, btn('Re-detect engine and devices', async () => {
        const r = await api.post(wr('/engine/refresh'));
        toast(`${r.devices.length} device(s): ${r.devices.map((x) => x.name).join(', ') || 'none'}`, 'ok');
        renderEngine(); renderHardware();
      }, { icon: 'refresh' })))
      : h('div', { class: 'note warn' }, 'No llama.cpp engine installed. Download one below (Vulkan is recommended for AMD Radeon GPUs on Windows).');
    const tb = h('tbody');
    for (const e of d.installed) {
      const isActive = a && a.path === e.server;
      tb.appendChild(h('tr', { class: isActive ? 'sel' : '' }, h('td', null, e.name), h('td', null, e.backend), h('td', null, e.tag || '-'),
        h('td', null, e.installed_at ? fmt.dateTime(e.installed_at) : '-'),
        h('td', null, isActive ? h('b', null, 'active') : btn('Use', async () => { await api.post(wr('/engine/select'), { name: e.name }); toast(`Engine ${e.name} selected`, 'ok'); renderEngine(); }, { cls: 'small' }),
          isActive ? null : btn('', async () => {
            if (await confirmBox('Remove engine', `Delete ${e.name} from disk?`, 'Delete')) { await api.del(wr(`/engine/${encodeURIComponent(e.name)}`)); renderEngine(); }
          }, { icon: 'trash', cls: 'icon small' }))));
    }
    const relBox = h('div');
    const relBtn = btn('Check for llama.cpp releases', async () => {
      relBtn.disabled = true;
      clear(relBox);
      relBox.appendChild(h('div', { class: 'dim' }, 'Querying GitHub...'));
      try {
        const r = await api.get(wr('/engine/releases'));
        clear(relBox);
        const rel = r.releases.find((x) => !x.prerelease) || r.releases[0];
        if (!rel) { relBox.appendChild(h('div', { class: 'dim' }, 'No releases found.')); return; }
        relBox.appendChild(h('div', { style: { marginBottom: '6px' } }, `Latest release: `, h('b', null, rel.tag), rel.published ? h('span', { class: 'dim' }, ` · ${rel.published.slice(0, 10)}`) : null));
        const labels = { vulkan: 'Vulkan (recommended for Radeon on Windows)', rocm: 'ROCm / HIP (needs the AMD HIP SDK installed)', cpu: 'CPU only' };
        for (const b of ['vulkan', 'rocm', 'cpu']) {
          const asset = rel.backends[b];
          relBox.appendChild(h('div', { class: 'row', style: { margin: '3px 0' } }, h('span', { style: { width: '300px' } }, labels[b]),
            asset ? h('span', { class: 'dim grow ellipsis' }, `${asset.name} · ${fmt.bytes(asset.size)}`) : h('span', { class: 'dim grow' }, 'not published for this platform'),
            asset ? btn('Install', async () => {
              try { await api.post(wr('/engine/install'), { tag: rel.tag, backend: b, asset }); toast(`Installing ${rel.tag} ${b}...`); } catch (e) { toast(e.message, 'err'); }
            }, { cls: 'small primary', icon: 'download' }) : null));
        }
      } catch (e) { clear(relBox); relBox.appendChild(h('div', { class: 'note err' }, e.message)); } finally { relBtn.disabled = false; }
    }, { icon: 'download' });
    const custom = input(es.engine_path, null, { placeholder: 'Path to llama-server.exe or its folder (e.g. a custom HIP build)', cls: 'grow mono' });
    const f = h('div', { class: 'form' });
    formRow(f, 'Process priority', select([['normal', 'Normal'], ['above_normal', 'Above normal (recommended)'], ['high', 'High']], es.process_priority,
      (v) => save({ engine: { process_priority: v } }, 'Priority saved - applies to the next load')));
    formRow(f, 'Engine log detail', select([[3, 'Normal'], [4, 'Detailed (recommended: buffer sizes, slots)'], [5, 'Debug']], es.log_verbosity,
      (v) => save({ engine: { log_verbosity: Number(v) } })));
    formRow(f, 'Engine memory fitting', toggle('Let llama.cpp fit layers to free VRAM in automatic mode', es.use_engine_fit, (v) => save({ engine: { use_engine_fit: v } })),
      'Uses the engine\'s exact allocator projection (--fit) within the configured VRAM margins.');
    formRow(f, 'Load timeout', h('div', { class: 'row' }, input(es.load_timeout_s, (v) => save({ engine: { load_timeout_s: Number(v) } }), { type: 'number', cls: 'num', min: 30 }),
      h('span', { class: 'dim' }, 'seconds')));
    host.append(group('Engine', [activeBox], { icon: 'server' }),
      group('Installed engines', d.installed.length ? h('div', { class: 'tbl-wrap' }, h('table', { class: 'tbl' }, h('thead', null, h('tr', null,
        h('th', null, 'Name'), h('th', null, 'Backend'), h('th', null, 'Release'), h('th', null, 'Installed'), h('th'))), tb)) : h('div', { class: 'dim' }, 'None.'),
      { icon: 'list' }),
      group('Download llama.cpp', [h('p', { class: 'dim' }, 'Official release builds from github.com/ggml-org/llama.cpp. The Vulkan build needs only the graphics driver. The ROCm build bundles the HIP runtime but loads rocBLAS from the AMD HIP SDK, which must be installed separately.'),
        relBtn, relBox, h('div', { style: { marginTop: '8px' } }, installBar, installTxt)], { icon: 'download' }),
      group('Custom engine', [h('div', { class: 'row' }, custom, btn('Use', async () => {
        try { await api.post(wr('/engine/select'), { path: custom.value.trim() }); toast('Custom engine selected', 'ok'); renderEngine(); } catch (e) { toast(e.message, 'err'); }
      }, { icon: 'check' }))], { icon: 'folder', collapsible: true, collapsed: !es.engine_path }),
      group('Engine options', f, { icon: 'sliders' }));
    renderInstall(store.installState);
  }
  function renderInstall(st) {
    if (!st || !st.phase) { installBar.classList.add('hidden'); installTxt.textContent = ''; return; }
    installBar.classList.remove('hidden');
    installBar.classList.toggle('indeterminate', st.phase === 'extract');
    installBar.firstChild.style.width = st.total ? `${(st.done / st.total) * 100}%` : '0%';
    installTxt.textContent = st.phase === 'download' ? `Downloading ${st.asset}: ${fmt.bytes(st.done)} / ${fmt.bytes(st.total)}`
      : st.phase === 'extract' ? 'Extracting...' : st.phase === 'done' ? `Installed ${st.tag} (${st.backend})` : st.phase === 'error' ? `Failed: ${st.error}` : st.phase;
    if (st.phase === 'done') setTimeout(renderEngine, 400);
  }

  // ================= HARDWARE =================
  async function renderHardware() {
    const host = hosts.hardware;
    clear(host);
    let d;
    try { d = await api.get(wr('/hardware')); } catch (e) { host.appendChild(h('div', { class: 'note err' }, e.message)); return; }
    const sys = d.system || {};
    const hw = S().hardware;
    const gtb = h('tbody');
    for (const g of sys.gpus || []) {
      const dev = Object.entries(d.device_map || {}).find(([, v]) => v === g.id)?.[0];
      gtb.appendChild(h('tr', null, h('td', null, g.id), h('td', null, g.name), h('td', null, g.vendor), h('td', { class: 'num' }, fmt.bytes(g.vram_total)),
        h('td', null, g.bus ?? '-'), h('td', null, g.driver || '-'), h('td', null, dev || '-')));
    }
    const dtb = h('tbody');
    for (const e of d.engine_devices || []) {
      const margin = hw.vram_margin_per_device?.[e.name];
      dtb.appendChild(h('tr', null, h('td', null, h('b', null, e.name)), h('td', null, e.description), h('td', { class: 'num' }, `${fmt.num(e.total_mib)} MiB`),
        h('td', { class: 'num' }, `${fmt.num(e.free_mib)} MiB`), h('td', { class: 'dim', style: { whiteSpace: 'normal', fontSize: '10px' } }, e.details || ''),
        h('td', null, input(margin ?? '', (v) => {
          const m = { ...(S().hardware.vram_margin_per_device || {}) };
          if (v === '' || v === null || isNaN(v)) delete m[e.name]; else m[e.name] = Number(v);
          save({ hardware: { vram_margin_per_device: m } });
        }, { type: 'number', cls: 'num', placeholder: String(hw.vram_margin_mib), style: { width: '80px' } }))));
    }
    const f = h('div', { class: 'form' });
    formRow(f, 'VRAM safety margin', h('div', { class: 'row' }, input(hw.vram_margin_mib, (v) => save({ hardware: { vram_margin_mib: Number(v) } }), { type: 'number', cls: 'num', min: 0, step: 128 }),
      h('span', { class: 'dim' }, 'MiB per GPU')), 'Kept free on every GPU for the desktop, browser and driver. Per-device values can be set in the table above.');
    formRow(f, 'Telemetry interval', select([[0.5, '0.5 s'], [1, '1 s'], [2, '2 s'], [5, '5 s']], hw.telemetry_interval_s, (v) => save({ hardware: { telemetry_interval_s: Number(v) } })));
    host.append(
      group('System', kv([['Operating system', sys.os], ['Processor', `${sys.cpu} · ${sys.cores_physical} cores / ${sys.cores_logical} threads`],
        ['Memory', fmt.bytes(sys.ram_total, 1)], ['Python', sys.python], ['Telemetry sources', Object.entries(sys.telemetry || {}).filter(([, v]) => v).map(([k]) => k.toUpperCase()).join(', ') || 'CPU / RAM only']]),
      { icon: 'chip' }),
      group('Graphics adapters (operating system)', (sys.gpus || []).length ? h('div', { class: 'tbl-wrap' }, h('table', { class: 'tbl' }, h('thead', null, h('tr', null,
        ['ID', 'Adapter', 'Vendor', 'VRAM', 'PCI bus', 'Driver', 'Engine device'].map((x, i) => h('th', { class: i === 3 ? 'num' : '' }, x)))), gtb))
        : h('div', { class: 'dim' }, 'No discrete GPUs reported.'), { icon: 'chip' }),
      group('Engine devices', [(d.engine_devices || []).length ? h('div', { class: 'tbl-wrap' }, h('table', { class: 'tbl' }, h('thead', null, h('tr', null,
        ['Device', 'Description', 'Total', 'Free now', 'Backend details', 'Margin MiB'].map((x, i) => h('th', { class: i === 2 || i === 3 ? 'num' : '' }, x)))), dtb))
        : h('div', { class: 'note warn' }, d.device_error || 'The engine reports no GPU devices. Check the graphics driver (Vulkan) or install the ROCm build.'), h('div', { class: 'divider' }), f],
      { icon: 'mem' }),
      group('Optimisation notes for this PC', (d.recommendations || []).map((r) => h('div', { class: 'rec' }, h('b', null, r.title), r.text)), { icon: 'bolt' }));
  }

  // ================= DEFAULTS =================
  async function renderDefaults() {
    const host = hosts.defaults;
    clear(host);
    let eng = null;
    let devs = [];
    try { eng = (await api.get(wr('/engine'))).active; devs = (await api.get(wr('/hardware'))).engine_devices || []; } catch { /* ignore */ }
    const saveDefaults = debounce((vals) => save({ defaults: vals }, 'Model defaults saved'), 700);
    const form = loadForm({ values: S().defaults, defaults: null, model: null, devices: devs, engine: eng, mode: 'defaults', onChange: (vals) => saveDefaults(vals) });
    host.append(group('Model load defaults', [h('p', { class: 'dim' }, 'Used for every model unless the model has its own saved load settings (Library › model › Load › Save as model defaults). The API uses them for just-in-time loads.'), form],
      { icon: 'sliders', tools: [btn('Restore factory defaults', async () => {
        if (await confirmBox('Restore defaults', 'Reset all model load defaults to factory values?', 'Reset')) {
          await api.post(wr('/settings/reset'), { section: 'defaults' });
          await store.loadSettings();
          renderDefaults();
        }
      }, { cls: 'small', icon: 'refresh' })] }));
  }

  // ================= STORAGE =================
  function renderStorage() {
    const host = hosts.storage;
    clear(host);
    const l = S().library;
    const f = h('div', { class: 'form' });
    formRow(f, 'Model folders', h('div', { class: 'col', style: { gap: '3px' } }, l.model_dirs.map((d) => h('code', null, d)),
      btn('Manage folders', () => openFolders(), { icon: 'folder', cls: 'small' })));
    formRow(f, 'Download folder', input(l.download_dir, (v) => save({ library: { download_dir: v } }), { cls: 'grow mono', placeholder: 'default: <install>\\models' }),
      'Models downloaded from Hugging Face are saved here (publisher\\repository\\file.gguf).');
    formRow(f, 'Hugging Face token', input(l.hf_token, (v) => save({ library: { hf_token: v } }), { type: 'password', cls: 'grow mono', placeholder: 'hf_... (only for gated repositories)' }));
    host.append(group('Storage', f, { icon: 'folder' }));
  }

  // ================= STARTUP =================
  function renderStartup() {
    const host = hosts.startup;
    clear(host);
    const s = S().startup;
    const f = h('div', { class: 'form' });
    formRow(f, 'Control panel', seg([['window', 'Native window'], ['browser', 'Web browser'], ['none', 'Do not open']], s.open_ui, (v) => save({ startup: { open_ui: v } })),
      'What happens when WinRunner starts. The panel is always available at http://127.0.0.1:<port>/.');
    formRow(f, 'Model', toggle(`Load the last used model on start${s.last_model ? ` (${s.last_model})` : ''}`, s.autoload_last_model, (v) => save({ startup: { autoload_last_model: v } })));
    host.append(group('Startup', f, { icon: 'power' }));
  }

  // ================= ABOUT =================
  function renderAbout() {
    const host = hosts.about;
    clear(host);
    const st = store.status || {};
    host.append(group('About', [
      h('div', { class: 'row', style: { gap: '14px', marginBottom: '8px' } }, h('svg', { viewBox: '0 0 48 48', style: { width: '56px', height: '56px' } }, h('use', { href: '#logo' })),
        h('div', null, h('div', { class: 'lib-title' }, `WinRunner ${st.version || ''}`), h('div', { class: 'dim' }, 'Local inference server with an OpenAI / LM Studio compatible API'))),
      kv([['Inference engine', st.engine ? `llama.cpp ${st.engine.version} (build ${st.engine.build}, ${st.engine.backend_label})` : 'not installed'],
        ['API', st.server ? `${st.server.local}${st.server.lan?.length ? ` · ${st.server.lan.join(' · ')}` : ''}` : '-'],
        ['Uptime', fmt.dur(store.now() - (st.started_at || store.now()))],
        ['Licenses', 'WinRunner: MIT. llama.cpp / ggml: MIT (github.com/ggml-org/llama.cpp).']]),
      h('div', { class: 'row', style: { marginTop: '10px' } }, btn('Reset appearance', async () => {
        if (await confirmBox('Reset appearance', 'Restore the default theme and display settings?', 'Reset')) { await api.post(wr('/settings/reset'), { section: 'ui' }); await store.loadSettings(); render(); }
      }, { cls: 'small' }), btn('Reset network settings', async () => {
        if (await confirmBox('Reset network', 'Restore default network and API settings (port 5070, all interfaces)?', 'Reset')) { await api.post(wr('/settings/reset'), { section: 'server' }); await store.loadSettings(); render(); }
      }, { cls: 'small' }))], { icon: 'model' }));
  }

  function render() {
    renderAppearance(); renderNetwork(); renderEngine(); renderHardware(); renderDefaults(); renderStorage(); renderStartup(); renderAbout();
  }
  render();
  let lastUi = JSON.stringify(S().ui);
  offs.push(store.on('settings', (s) => {
    const ui = JSON.stringify(s.ui);
    if (ui !== lastUi) { lastUi = ui; renderAppearance(); }
    renderNetwork(); renderStartup(); renderStorage();
  }));
  offs.push(store.on('engine_install', renderInstall));
  offs.push(store.on('engine_changed', () => renderEngine()));
  return () => offs.forEach((f) => f());
}
