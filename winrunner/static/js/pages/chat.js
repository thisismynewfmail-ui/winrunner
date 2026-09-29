// CHAT: test console using the same /v1/chat/completions endpoint as external clients.

import { h, clear, icon, setText } from '../core/dom.js';
import { api, wr } from '../core/api.js';
import { store } from '../core/store.js';
import * as fmt from '../core/fmt.js';
import { btn, toggle, toast, select, input, confirmBox, copyBtn } from '../components/ui.js';
import { renderMarkdown } from '../components/markdown.js';
import { takeNavParams } from '../app.js';

let lastChatId = null;

function newChat(model) {
  return { id: `c${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`, title: 'New conversation', model: model || '',
    system: '', messages: [], params: {}, updated: Date.now() / 1000 };
}

export function mount(root) {
  const nav = takeNavParams();
  let chat = newChat(nav?.model || store.primary?.model);
  let library = [];
  let busy = null; // AbortController
  let attachments = [];
  const offs = [];

  // ---- top bar -----------------------------------------------------------------------
  const modelSel = select([], '', (v) => { chat.model = v; updateVisionBadge(); save(); });
  modelSel.style.minWidth = '260px';
  const visBadge = h('span');
  const sideToggle = btn('Settings', () => side.classList.toggle('collapsed'), { icon: 'sliders' });
  const top = h('div', { class: 'chat-top' },
    h('span', { class: 'dim' }, 'Model'), modelSel, visBadge, h('span', { class: 'spacer' }),
    btn('New', () => { chat = newChat(chat.model); renderAll(); }, { icon: 'plus' }),
    btn('Clear', () => { chat.messages = []; renderLog(); save(); }, { icon: 'trash' }),
    sideToggle);

  // ---- log and composer --------------------------------------------------------------
  const log = h('div', { class: 'chat-log' });
  const ta = h('textarea', { class: 'field', placeholder: 'Message (Enter to send, Shift+Enter for a new line). Paste or drop images for vision models.', rows: 3 });
  const attachBox = h('div', { class: 'attach' });
  const fileIn = h('input', { type: 'file', accept: 'image/*', multiple: true, style: { display: 'none' } });
  const attachBtn = btn('Image', () => fileIn.click(), { icon: 'image', tip: 'Attach images (vision models only)' });
  const sendBtn = btn('Send', () => send(), { icon: 'send', cls: 'primary' });
  const stopBtn = btn('Stop', () => busy?.abort(), { icon: 'stop', cls: 'hidden danger' });
  const status = h('span', { class: 'dim' });
  const composer = h('div', { class: 'composer' }, attachBox, ta,
    h('div', { class: 'row' }, attachBtn, fileIn, status, h('span', { class: 'spacer' }), stopBtn, sendBtn));
  const col = h('div', { class: 'chat-col' }, log, composer);

  // ---- side panel ----------------------------------------------------------------------
  const listBox = h('div', { class: 'chat-list tbl-wrap', style: { maxHeight: '220px' } });
  const sysIn = h('textarea', { class: 'field', rows: 4, placeholder: 'System prompt (optional)' });
  sysIn.addEventListener('change', () => { chat.system = sysIn.value; save(); });
  const params = {};
  const pForm = h('div', { class: 'form' });
  const P = [['temperature', 'Temperature', 0.05], ['top_p', 'Top P', 0.01], ['top_k', 'Top K', 1], ['min_p', 'Min P', 0.01],
    ['repeat_penalty', 'Repeat penalty', 0.01], ['max_tokens', 'Max tokens', 1], ['seed', 'Seed', 1]];
  for (const [k, label, step] of P) {
    params[k] = input('', (v) => { if (v === '' || v === null) delete chat.params[k]; else chat.params[k] = Number(v); save(); },
      { type: 'number', step, placeholder: 'model default', cls: 'num', style: { width: '100%' } });
    pForm.append(h('label', null, label), params[k]);
  }
  const thinkT = toggle('Thinking enabled', true, (v) => { chat.params._think = v; save(); }, { tip: 'Sets chat_template_kwargs.enable_thinking for templates that support it' });
  const side = h('div', { class: `chat-side ${window.innerWidth < 1100 ? 'collapsed' : ''}` },
    h('section', { class: 'group' }, h('div', { class: 'gt' }, icon('list'), 'Conversations'), h('div', { class: 'gb' }, listBox)),
    h('section', { class: 'group' }, h('div', { class: 'gt' }, icon('chat'), 'System prompt'), h('div', { class: 'gb' }, sysIn)),
    h('section', { class: 'group' }, h('div', { class: 'gt' }, icon('sliders'), 'Sampling'), h('div', { class: 'gb' },
      h('div', { class: 'dim', style: { fontSize: '10px', marginBottom: '4px' } }, 'Empty fields use the model preset / GGUF defaults.'), pForm, thinkT)));

  root.append(h('div', { class: 'pg-head' }, h('h2', null, 'Chat'), h('span', { class: 'sub' }, 'Test console · requests go through the public API')),
    top, h('div', { class: 'chat-main' }, col, side));

  // ---- helpers --------------------------------------------------------------------------
  async function loadModels() {
    try {
      const d = await api.get(wr('/library'));
      library = d.models.filter((m) => m.kind !== 'embedding');
    } catch { library = []; }
    const loaded = [...store.instances.values()].filter((i) => i.state === 'ready' || i.state === 'loading').map((i) => i.model);
    clear(modelSel);
    const add = (v, l) => modelSel.appendChild(h('option', { value: v }, l));
    if (loaded.length) {
      const og = h('optgroup', { label: 'Loaded' });
      for (const m of loaded) og.appendChild(h('option', { value: m }, `● ${m}`));
      modelSel.appendChild(og);
    }
    const og2 = h('optgroup', { label: 'Library (loads on first message)' });
    for (const m of library) if (!loaded.includes(m.id)) og2.appendChild(h('option', { value: m.id }, `${m.id}${m.vision ? '  [vision]' : ''}`));
    modelSel.appendChild(og2);
    if (!library.length && !loaded.length) add('', 'No models');
    if (!chat.model) chat.model = loaded[0] || library[0]?.id || '';
    modelSel.value = chat.model;
    updateVisionBadge();
  }

  function modelInfo() { return library.find((m) => m.id === chat.model); }
  function updateVisionBadge() {
    clear(visBadge);
    const m = modelInfo();
    const inst = [...store.instances.values()].find((i) => i.model === chat.model);
    const vision = inst ? inst.vision : m?.vision;
    visBadge.append(vision ? h('span', { class: 'badge vision', 'data-tip': 'Image input available' }, icon('eye'), 'Vision')
      : h('span', { class: 'badge', 'data-tip': 'This model has no vision projector: images cannot be sent' }, 'Text only'));
    attachBtn.disabled = !vision;
    thinkT.classList.toggle('hidden', !(m?.reasoning));
  }

  async function loadList() {
    try {
      const d = await api.get(wr('/chats'));
      clear(listBox);
      for (const c of d.chats) {
        const it = h('div', { class: `item ${c.id === chat.id ? 'on' : ''}`, onclick: () => open(c.id) },
          h('div', { class: 'ellipsis' }, c.title || 'Untitled'), h('div', { class: 'dim', style: { fontSize: '10px' } }, `${fmt.ago(c.updated)} · ${c.messages} msgs`));
        it.addEventListener('contextmenu', async (e) => {
          e.preventDefault();
          if (await confirmBox('Delete conversation', `Delete "${c.title}"?`, 'Delete')) { await api.del(wr(`/chats/${c.id}`)); if (c.id === chat.id) chat = newChat(chat.model); renderAll(); }
        });
        listBox.appendChild(it);
      }
      if (!d.chats.length) listBox.appendChild(h('div', { class: 'dim', style: { padding: '6px' } }, 'No saved conversations.'));
    } catch { /* ignore */ }
  }

  async function open(id) {
    try { chat = await api.get(wr(`/chats/${id}`)); lastChatId = id; renderAll(); } catch (e) { toast(e.message, 'err'); }
  }

  let saveTimer = null;
  function save() {
    if (!chat.messages.length) return;
    clearTimeout(saveTimer);
    saveTimer = setTimeout(async () => {
      chat.updated = Date.now() / 1000;
      try { await api.put(wr(`/chats/${chat.id}`), chat); lastChatId = chat.id; loadList(); } catch { /* ignore */ }
    }, 600);
  }

  function renderAll() {
    sysIn.value = chat.system || '';
    for (const [k] of P) params[k].value = chat.params?.[k] ?? '';
    thinkT.set(chat.params?._think !== false);
    modelSel.value = chat.model;
    updateVisionBadge();
    renderLog();
    loadList();
  }

  function msgEl(m, idx) {
    const body = h('div', { class: 'msg-body' });
    const el = h('div', { class: `msg ${m.role}` },
      h('div', { class: 'msg-head' }, h('span', { class: 'who' }, m.role === 'assistant' ? (m.model || 'assistant') : m.role),
        m.t ? h('span', null, fmt.time(m.t)) : null, h('span', { class: 'spacer' }),
        copyBtn(() => m.content || '', { tip: 'Copy text' }),
        idx === chat.messages.length - 1 && m.role === 'assistant' ? btn('', () => regenerate(), { icon: 'refresh', cls: 'icon small', tip: 'Regenerate' }) : null,
        btn('', () => { chat.messages.splice(idx, 1); renderLog(); save(); }, { icon: 'x', cls: 'icon small', tip: 'Delete message' })),
      body, h('div', { class: 'msg-foot' }));
    fillMsg(el, m, false);
    return el;
  }

  function fillMsg(el, m, streaming) {
    const body = el.querySelector('.msg-body');
    clear(body);
    if (m.images?.length) body.appendChild(h('div', { class: 'msg-imgs' }, m.images.map((src) => h('img', { src, alt: 'attached image' }))));
    if (m.reasoning) {
      const det = h('details', { class: 'reason', open: streaming && !m.content },
        h('summary', null, `Reasoning · ${fmt.num(m.reasoning_tokens || 0)} tokens${streaming && !m.content ? ' · thinking...' : ''}`),
        h('div', { class: 'rt' }, m.reasoning));
      body.appendChild(det);
    }
    for (const tc of m.tool_calls || []) {
      body.appendChild(h('div', { class: 'toolcall' }, `tool call: ${tc.function?.name}(${tc.function?.arguments || ''})`));
    }
    if (m.role === 'assistant') {
      const c = h('div');
      c.innerHTML = renderMarkdown(m.content || (streaming ? '' : m.tool_calls?.length ? '' : '(empty)'), streaming);
      body.appendChild(c);
    } else {
      body.appendChild(h('div', { style: { whiteSpace: 'pre-wrap' } }, m.content));
    }
    const foot = el.querySelector('.msg-foot');
    clear(foot);
    const s = m.stats;
    if (s) {
      foot.append(
        s.prompt_n !== undefined ? h('span', null, `prompt ${fmt.num(s.prompt_n)} tok${s.cache_n ? ` (+${fmt.num(s.cache_n)} cached)` : ''} @ ${fmt.tps(s.prompt_per_second)} t/s`) : null,
        s.predicted_n !== undefined ? h('span', null, `output ${fmt.num(s.predicted_n)} tok @ ${fmt.tps(s.predicted_per_second)} t/s`) : null,
        s.ttft !== undefined ? h('span', null, `TTFT ${fmt.ms(s.ttft * 1000)}`) : null,
        s.finish ? h('span', null, `stop: ${s.finish}`) : null);
    } else if (m.status) foot.append(h('span', null, m.status));
  }

  function renderLog() {
    clear(log);
    if (!chat.messages.length) {
      log.appendChild(h('div', { class: 'empty' }, h('div', { class: 'big' }, 'Chat console'),
        h('div', null, 'Messages are sent to /v1/chat/completions exactly like an external client, so the sidebar shows live prompt processing and token streaming.')));
    }
    chat.messages.forEach((m, i) => log.appendChild(msgEl(m, i)));
    log.scrollTop = log.scrollHeight;
  }

  function toApiMessages() {
    const out = [];
    if (chat.system?.trim()) out.push({ role: 'system', content: chat.system });
    for (const m of chat.messages) {
      if (m.role === 'user') {
        if (m.images?.length) {
          out.push({ role: 'user', content: [{ type: 'text', text: m.content }, ...m.images.map((u) => ({ type: 'image_url', image_url: { url: u } }))] });
        } else out.push({ role: 'user', content: m.content });
      } else if (m.role === 'assistant' && !m.error) {
        const x = { role: 'assistant', content: m.content || '' };
        if (m.tool_calls?.length) x.tool_calls = m.tool_calls;
        out.push(x);
      }
    }
    return out;
  }

  async function send() {
    if (busy) return;
    const text = ta.value.trim();
    if (!text && !attachments.length) return;
    if (!chat.model) { toast('Select a model first', 'warn'); return; }
    chat.messages.push({ role: 'user', content: text, images: attachments.map((a) => a.url), t: Date.now() / 1000 });
    if (chat.title === 'New conversation') chat.title = text.slice(0, 60) || 'Image';
    ta.value = '';
    attachments = [];
    renderAttach();
    await generate();
  }

  async function regenerate() {
    if (busy) return;
    if (chat.messages[chat.messages.length - 1]?.role === 'assistant') chat.messages.pop();
    await generate();
  }

  async function generate() {
    const apiMessages = toApiMessages(); // before the pending reply is added (an empty assistant turn would be prefilled)
    const m = { role: 'assistant', content: '', reasoning: '', reasoning_tokens: 0, model: chat.model, t: Date.now() / 1000, status: 'sending...' };
    chat.messages.push(m);
    renderLog();
    const el = log.lastChild;
    busy = new AbortController();
    sendBtn.classList.add('hidden');
    stopBtn.classList.remove('hidden');
    const t0 = performance.now();
    let tFirst = null;
    const body = { model: chat.model, messages: apiMessages, stream: true, stream_options: { include_usage: true } };
    for (const [k] of P) if (chat.params?.[k] !== undefined) body[k] = chat.params[k];
    const info = modelInfo();
    if (info?.reasoning && chat.params?._think === false) body.chat_template_kwargs = { enable_thinking: false };
    let dirty = false;
    const raf = () => { if (dirty) { fillMsg(el, m, true); dirty = false; const nearBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 80; if (nearBottom) log.scrollTop = log.scrollHeight; } };
    const tick = setInterval(raf, 50);
    try {
      const res = await fetch('/v1/chat/completions', {
        method: 'POST', signal: busy.signal,
        headers: { 'Content-Type': 'application/json', 'X-WinRunner-UI': '1', ...(store.settings?.server?.api_key ? { Authorization: `Bearer ${store.settings.server.api_key}` } : {}) },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error?.message || `HTTP ${res.status}`);
      }
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      m.status = 'waiting for first token...';
      dirty = true;
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf('\n\n')) >= 0) {
          const ev = buf.slice(0, i);
          buf = buf.slice(i + 2);
          for (const line of ev.split('\n')) {
            if (line.startsWith(':')) { m.status = line.slice(1).trim().replace(/^winrunner /, ''); dirty = true; continue; }
            if (!line.startsWith('data:')) continue;
            const data = line.slice(5).trim();
            if (data === '[DONE]') continue;
            let obj;
            try { obj = JSON.parse(data); } catch { continue; }
            if (obj.error) throw new Error(obj.error.message || JSON.stringify(obj.error));
            const ch = obj.choices?.[0];
            const d = ch?.delta || {};
            if (d.content) { m.content += d.content; if (!tFirst) tFirst = performance.now(); }
            if (d.reasoning_content) { m.reasoning += d.reasoning_content; m.reasoning_tokens++; if (!tFirst) tFirst = performance.now(); }
            for (const tc of d.tool_calls || []) {
              m.tool_calls ||= [];
              const slot = m.tool_calls[tc.index || 0] ||= { id: tc.id || '', type: 'function', function: { name: '', arguments: '' } };
              if (tc.function?.name) slot.function.name += tc.function.name;
              if (tc.function?.arguments) slot.function.arguments += tc.function.arguments;
            }
            if (ch?.finish_reason) m.finish = ch.finish_reason;
            if (obj.timings) m.stats = { ...obj.timings, ttft: tFirst ? (tFirst - t0) / 1000 : undefined, finish: m.finish };
            dirty = true;
          }
        }
      }
      if (m.stats) m.stats.finish = m.finish;
      else m.stats = { ttft: tFirst ? (tFirst - t0) / 1000 : undefined, finish: m.finish };
    } catch (e) {
      if (e.name === 'AbortError') { m.stats = { ...(m.stats || {}), finish: 'stopped by user' }; } else {
        m.error = e.message;
        m.content = m.content || `**Error:** ${e.message}`;
        toast(e.message, 'err', 6000, 'Request failed');
      }
    } finally {
      clearInterval(tick);
      busy = null;
      delete m.status;
      stopBtn.classList.add('hidden');
      sendBtn.classList.remove('hidden');
      fillMsg(el, m, false);
      renderLog();
      save();
      loadModels();
    }
  }

  // ---- attachments ------------------------------------------------------------------------
  function addFiles(files) {
    for (const f of files) {
      if (!f.type.startsWith('image/')) continue;
      if (attachBtn.disabled) { toast('The selected model has no vision projector; images cannot be sent.', 'warn'); return; }
      const r = new FileReader();
      r.onload = () => { attachments.push({ name: f.name, url: r.result }); renderAttach(); };
      r.readAsDataURL(f);
    }
  }
  function renderAttach() {
    clear(attachBox);
    attachments.forEach((a, i) => attachBox.appendChild(h('div', { class: 'attach-item', 'data-tip': a.name },
      h('img', { src: a.url, alt: a.name }), btn('', () => { attachments.splice(i, 1); renderAttach(); }, { icon: 'x', cls: 'icon small' }))));
    setText(status, attachments.length ? `${attachments.length} image(s) attached` : '');
  }
  fileIn.addEventListener('change', () => { addFiles(fileIn.files); fileIn.value = ''; });
  ta.addEventListener('paste', (e) => { const files = [...(e.clipboardData?.files || [])]; if (files.length) { e.preventDefault(); addFiles(files); } });
  composer.addEventListener('dragover', (e) => { e.preventDefault(); composer.classList.add('drag'); });
  composer.addEventListener('dragleave', () => composer.classList.remove('drag'));
  composer.addEventListener('drop', (e) => { e.preventDefault(); composer.classList.remove('drag'); addFiles(e.dataTransfer.files); });
  ta.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } });

  offs.push(store.on('instance', () => { loadModels(); }));
  offs.push(store.on('instance_removed', () => loadModels()));
  (async () => {
    await loadModels();
    if (!nav?.model && lastChatId) await open(lastChatId); else renderAll();
  })();
  return () => { offs.forEach((f) => f()); busy?.abort(); };
}
