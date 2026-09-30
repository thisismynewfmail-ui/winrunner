// Model load configuration form (per-model and global defaults).

import { h, clear } from '../core/dom.js';
import * as fmt from '../core/fmt.js';
import { seg, toggle, select, input, slideNum, formRow, btn } from './ui.js';

const KV_OPTS = [
  ['auto', 'Auto (F16, Q8_0 when needed)'], ['f16', 'F16 (full precision)'], ['bf16', 'BF16'], ['q8_0', 'Q8_0 (near-lossless, 50%)'],
  ['q5_1', 'Q5_1'], ['q5_0', 'Q5_0'], ['q4_1', 'Q4_1'], ['q4_0', 'Q4_0 (lossy, 28%)'], ['iq4_nl', 'IQ4_NL'], ['f32', 'F32'],
];
const LOAD_MODES = [
  ['auto', 'Auto (full read when fully offloaded, else memory-map)'], ['mmap', 'Memory-map (mmap)'], ['none', 'Full read (no mmap)'],
  ['mlock', 'Lock in RAM (mlock)'], ['mmap+mlock', 'Memory-map + lock'], ['dio', 'Direct I/O'],
];

/**
 * @param opts {values, defaults, saved (keys set in the model profile), model (detail|null), devices, engine, onChange}
 */
export function loadForm(opts) {
  const v = { ...opts.values };
  const el = h('div', { class: 'loadform' });
  const model = opts.model;
  const info = model?.info || {};
  const nLayer = info.n_layer || 0;
  const eng = opts.engine || {};
  const has = (f) => !eng.flags || eng.flags.includes(f);
  const changed = (k) => opts.defaults && JSON.stringify(v[k]) !== JSON.stringify(opts.defaults[k]);

  function set(k, val) {
    v[k] = val;
    opts.onChange?.({ ...v }, k);
    refreshTags();
  }
  const tags = new Map();
  function tag(k) {
    const t = h('span', { class: 'src-tag hidden' });
    tags.set(k, t);
    return t;
  }
  function refreshTags() {
    for (const [k, t] of tags) {
      const isSaved = opts.saved?.includes?.(k);
      const differs = changed(k);
      t.classList.toggle('hidden', !differs && !isSaved);
      t.className = `src-tag ${differs ? 'user' : ''}${!differs && !isSaved ? ' hidden' : ''}`;
      t.textContent = differs ? (opts.mode === 'defaults' ? 'modified' : 'custom') : 'saved';
      t.dataset.tip = differs ? `Differs from the global default (${JSON.stringify(opts.defaults?.[k])})` : 'Saved in this model\'s profile';
    }
  }

  function build() {
    clear(el);
    tags.clear();
    const f = h('div', { class: 'form' });
    const sec = (t) => f.appendChild(h('div', { class: 'form-section' }, t));
    const row = (label, key, control, hint, tip) => formRow(f, label, control, hint, { tip, tag: key ? tag(key) : null });

    // ---- context ---------------------------------------------------------
    sec('Context');
    const train = info.context_length || 0;
    const maxCtx = Math.max(262144, train, v.context_length);
    const ctxCtl = slideNum(v.context_length, { min: 512, max: maxCtx, step: 256, log2: true, width: 96 }, (x) => { set('context_length', x); updateCtxHint(); });
    const presets = h('div', { class: 'row wrap' }, [4096, 8192, 16384, 32768, 65536, 131072].map((c) =>
      btn(fmt.ctx(c), () => { ctxCtl.set(c); set('context_length', c); updateCtxHint(); }, { cls: 'small' })),
    train ? btn(`Max (${fmt.ctx(train)})`, () => { ctxCtl.set(train); set('context_length', train); updateCtxHint(); }, { cls: 'small' }) : null);
    const ctxHint = h('span');
    const updateCtxHint = () => {
      clear(ctxHint);
      if (!train) { ctxHint.textContent = 'Tokens of context (prompt + generated). The KV cache grows linearly with it.'; return; }
      if (v.context_length > train && !v.allow_context_over_train) ctxHint.append(h('span', { class: 'warn' }, `Above the trained context (${fmt.num(train)}): the effective context will be clamped to ${fmt.num(train)}.`));
      else if (v.context_length > train) ctxHint.append(h('span', { class: 'warn' }, `Above the trained context (${fmt.num(train)}): uses RoPE scaling, quality may degrade.`));
      else ctxHint.append(`Trained context: ${fmt.num(train)} tokens.`);
    };
    updateCtxHint();
    row('Context length', 'context_length', h('div', { class: 'col grow', style: { gap: '4px' } }, ctxCtl, presets), ctxHint,
      'Maximum tokens per conversation (prompt + response). Default 65,536.');
    row('Above trained context', 'allow_context_over_train',
      toggle('Allow (RoPE scaling)', v.allow_context_over_train, (x) => { set('allow_context_over_train', x); updateCtxHint(); }),
      null, 'Without this, contexts above the model\'s trained length are clamped to it.');
    row('Context in VRAM', 'context_fit', seg([
      ['fill', 'Fill VRAM', 'Use the largest context that fits: the KV cache takes the free VRAM (up to the trained context). If the requested length does not fit, it is reduced so the whole model stays on the GPUs.'],
      ['fit', 'Up to requested', 'Use the requested length, reduced only when it does not fit, so the whole model stays on the GPUs.'],
      ['off', 'Exact', 'Always use the requested length. If it does not fit, part of the model runs from system RAM: prompt processing becomes many times slower.'],
    ], v.context_fit, (x) => set('context_fit', x)),
    'Fill VRAM (default) uses the GPUs up to the safety margin; the requested length decides the KV cache precision.',
    'Running any part of a model from system RAM is very slow in llama.cpp, so the context is fitted first (down to 4,096 tokens).');

    // ---- GPU offload ------------------------------------------------------
    sec('GPU offload');
    const manualBox = h('div', { class: 'col grow', style: { gap: '6px' } });
    row('Allocation', 'gpu_offload', seg([['auto', 'Automatic', 'Keep the whole model on the GPUs and fit the context and buffers to free VRAM'], ['manual', 'Manual']],
      v.gpu_offload, (x) => { set('gpu_offload', x); renderManual(); }), 'Automatic keeps the whole model in VRAM whenever it fits and verifies the layout with the engine\'s own projection.');
    formRow(f, '', manualBox);
    const renderManual = () => {
      clear(manualBox);
      if (v.gpu_offload !== 'manual') return;
      const maxL = (nLayer || 99) + 1;
      const ngl = v.n_gpu_layers < 0 ? maxL : v.n_gpu_layers;
      const lbl = h('span', { class: 'dim' });
      const upd = (x) => { lbl.textContent = x >= maxL ? `all ${maxL} (incl. output layer)` : `${x} of ${maxL}`; };
      upd(ngl);
      manualBox.append(h('div', { class: 'row' }, h('span', { style: { width: '120px' } }, 'GPU layers'),
        slideNum(ngl, { min: 0, max: maxL, step: 1, width: 70 }, (x) => { set('n_gpu_layers', x >= maxL ? -1 : x); upd(x); }), lbl));
      if (model?.is_moe || opts.mode === 'defaults') {
        manualBox.append(h('div', { class: 'row' }, h('span', { style: { width: '120px' }, 'data-tip': 'Expert weights of the first N layers stay in system RAM' }, 'MoE experts on CPU'),
          slideNum(v.n_cpu_moe, { min: 0, max: nLayer || 99, step: 1, width: 70 }, (x) => set('n_cpu_moe', x)), h('span', { class: 'dim' }, 'layers')));
      }
    };
    renderManual();

    const devs = opts.devices || [];
    if (devs.length) {
      const devBox = h('div', { class: 'col', style: { gap: '3px' } });
      for (const d of devs) {
        const on = !v.devices?.length || v.devices.includes(d.name);
        devBox.appendChild(toggle(`${d.name} · ${d.description} · ${fmt.num(d.free_mib)} / ${fmt.num(d.total_mib)} MiB free`, on, (x) => {
          let list = v.devices?.length ? [...v.devices] : devs.map((z) => z.name);
          list = x ? [...new Set([...list, d.name])] : list.filter((n) => n !== d.name);
          if (list.length === devs.length) list = [];
          set('devices', list);
          build();
        }));
      }
      row('Devices', 'devices', devBox, 'GPUs the model may use. All selected devices share the layers.');
    }
    if (devs.length > 1 || opts.mode === 'defaults') {
      row('Split mode', 'split_mode', seg([['layer', 'Layer', 'Contiguous layers per GPU (pipeline). Best for PCIe consumer systems.'],
        ['row', 'Row', 'Each weight split by rows across GPUs (ROCm/CUDA only, heavy PCIe traffic)'],
        ['none', 'Single GPU']], v.split_mode, (x) => { set('split_mode', x); build(); }));
      if (v.split_mode !== 'layer') {
        row('Main GPU', 'main_gpu', select(devs.map((d, i) => [i, `${i}: ${d.name}`]), v.main_gpu, (x) => set('main_gpu', Number(x))));
      }
      const tsBox = h('div', { class: 'row wrap' });
      const auto = toggle('Automatic', !v.tensor_split?.length, (x) => {
        set('tensor_split', x ? [] : devs.map(() => 1));
        build();
      }, { tip: 'Computed per model from free VRAM, output layer placement and KV cache' });
      tsBox.appendChild(auto);
      if (v.tensor_split?.length) {
        v.tensor_split.forEach((val, i) => {
          tsBox.appendChild(h('span', { class: 'dim' }, devs[i]?.name || `GPU${i}`));
          tsBox.appendChild(input(val, (x) => { const ts = [...v.tensor_split]; ts[i] = Number(x) || 0; set('tensor_split', ts); },
            { type: 'number', cls: 'num', min: 0, step: 1, style: { width: '60px' } }));
        });
      }
      row('Tensor split', 'tensor_split', tsBox, 'Relative share of layers per GPU (e.g. 1,1 or 30,34).');
    }

    // ---- attention & KV ------------------------------------------------------
    sec('Attention and KV cache');
    row('Flash attention', 'flash_attn', seg([['auto', 'Auto'], ['on', 'On'], ['off', 'Off']], v.flash_attn, (x) => set('flash_attn', x)),
      'Fused attention kernel: far smaller compute buffers at long context. Required for a quantized V cache.',
      'Supported by the Vulkan and ROCm backends on RDNA2.');
    row('KV cache type', 'kv_cache_type', select(KV_OPTS, v.kv_cache_type, (x) => set('kv_cache_type', x), { style: { minWidth: '240px' } }),
      'Precision of the attention key/value cache. Auto keeps F16 and uses Q8_0 only when F16 cannot reach the requested context in VRAM.');
    row('V cache type', 'kv_cache_type_v', select([['', 'Same as K'], ...KV_OPTS.filter(([k]) => k !== 'auto')], v.kv_cache_type_v,
      (x) => set('kv_cache_type_v', x)), null, 'Separate precision for the value cache (advanced).');
    row('KV cache on GPU', 'kv_offload', toggle('Offload KV cache', v.kv_offload, (x) => set('kv_offload', x)), null,
      'Keep the KV cache in VRAM (recommended). Disabling it saves VRAM at a large speed cost.');
    if (info.sliding_window || opts.mode === 'defaults') {
      row('Sliding window cache', 'swa_full', toggle('Full-size SWA cache', v.swa_full, (x) => set('swa_full', x)),
        info.sliding_window ? `Model uses a ${info.sliding_window}-token sliding window. Full size uses more VRAM but allows prompt cache reuse.` : null);
    }

    // ---- batching ------------------------------------------------------------
    sec('Batching and threads');
    row('Batch size', 'batch_size', select([512, 1024, 2048, 4096, 8192].map((x) => [x, String(x)]), v.batch_size, (x) => set('batch_size', Number(x))),
      null, 'Logical batch: maximum tokens submitted per decode call.');
    row('Micro-batch size', 'ubatch_size', select([128, 256, 512, 1024, 2048, 4096].map((x) => [x, String(x)]), v.ubatch_size, (x) => set('ubatch_size', Number(x))),
      'Physical batch per GPU pass. Larger values speed up prompt processing and use more compute buffer VRAM.');
    row('Parallel slots', 'parallel', select([[-1, 'Auto (4, shared context)'], [1, '1'], [2, '2'], [3, '3'], [4, '4'], [6, '6'], [8, '8']], v.parallel,
      (x) => set('parallel', Number(x))), 'Concurrent requests. Slots share one unified KV cache, so a single request can still use the full context.');
    row('Threads', 'threads', h('div', { class: 'row' },
      input(v.threads, (x) => set('threads', Number(x) || 0), { type: 'number', cls: 'num', min: 0, max: 256, style: { width: '64px' } }), h('span', { class: 'dim' }, 'generation'),
      input(v.threads_batch, (x) => set('threads_batch', Number(x) || 0), { type: 'number', cls: 'num', min: 0, max: 256, style: { width: '64px' } }), h('span', { class: 'dim' }, 'batch')),
      '0 = engine default (one thread per physical core). Only matters for layers running on the CPU.');
    row('Load mode', 'load_mode', select(LOAD_MODES, v.load_mode, (x) => set('load_mode', x), { style: { minWidth: '240px' } }));

    // ---- vision ----------------------------------------------------------------
    if (model?.mmproj_candidates?.length || opts.mode === 'defaults') {
      sec('Vision');
      if (model) {
        const cands = model.mmproj_candidates || [];
        row('Vision projector', 'mmproj', select([['', `Auto${model.mmproj ? ` (${model.mmproj.split(/[\\/]/).pop()})` : ''}`], ['none', 'Disabled'],
          ...cands.map((p) => [p, p.split(/[\\/]/).pop()])], v.mmproj, (x) => set('mmproj', x), { style: { maxWidth: '100%' } }),
        'The mmproj file is paired automatically from the model folder.');
      }
      row('Projector on GPU', 'mmproj_offload', toggle('Offload vision projector', v.mmproj_offload, (x) => set('mmproj_offload', x)), null,
        'Image encoding on the GPU (fast). Disable to save VRAM.');
      if (has('--image-max-tokens')) {
        row('Image tokens', 'image_max_tokens', h('div', { class: 'row' },
          input(v.image_min_tokens, (x) => set('image_min_tokens', Number(x) || 0), { type: 'number', cls: 'num', min: 0, style: { width: '70px' } }), h('span', { class: 'dim' }, 'min'),
          input(v.image_max_tokens, (x) => set('image_max_tokens', Number(x) || 0), { type: 'number', cls: 'num', min: 0, style: { width: '70px' } }), h('span', { class: 'dim' }, 'max')),
        '0 = model default. For dynamic-resolution vision models: more tokens = more detail and more compute.');
      }
    }

    // ---- template & reasoning ---------------------------------------------------------
    sec('Chat template and reasoning');
    row('Template source', 'chat_template_mode', seg([['gguf', 'GGUF embedded', 'Use the Jinja template stored in the model file (recommended)'],
      ['builtin', 'Built-in'], ['custom', 'Custom']], v.chat_template_mode, (x) => { set('chat_template_mode', x); build(); }),
    v.chat_template_mode === 'gguf' ? 'The model\'s own template is rendered by the engine\'s Jinja implementation.' : null);
    if (v.chat_template_mode === 'builtin') {
      row('Built-in template', 'chat_template_builtin', select([['', 'Select...'], ...(eng.builtin_templates || []).map((t) => [t, t])], v.chat_template_builtin,
        (x) => set('chat_template_builtin', x)));
    }
    if (v.chat_template_mode === 'custom') {
      const ta = h('textarea', { class: 'field mono', rows: 8, spellcheck: 'false', style: { width: '100%' } });
      ta.value = v.chat_template_custom || info.chat_template || '';
      ta.addEventListener('change', () => set('chat_template_custom', ta.value));
      row('Custom template', 'chat_template_custom', ta, 'Jinja template. Saved to the data folder and passed with --chat-template-file.');
    }
    row('Template arguments', 'chat_template_kwargs', input(v.chat_template_kwargs, (x) => set('chat_template_kwargs', x),
      { cls: 'mono grow', placeholder: '{"enable_thinking": false}' }), 'Extra JSON variables for the template (applies to every request).');
    row('Reasoning output', 'reasoning_format', select([['auto', 'Separate field reasoning_content (auto)'], ['deepseek', 'Separate field reasoning_content'],
      ['deepseek-legacy', 'Both: <think> in content and reasoning_content'], ['none', 'Inline <think> tags in content']], v.reasoning_format,
    (x) => set('reasoning_format', x)), 'How thinking text is returned to API clients.');
    row('Thinking', 'reasoning', seg([['auto', 'Auto'], ['on', 'On'], ['off', 'Off']], v.reasoning, (x) => set('reasoning', x)),
      'For models with a thinking mode (e.g. Qwen3, GLM, DeepSeek).');
    row('Reasoning budget', 'reasoning_budget', input(v.reasoning_budget, (x) => set('reasoning_budget', Number(x)), { type: 'number', cls: 'num', min: -1 }),
      '-1 = unrestricted; N = maximum thinking tokens before the answer is forced.');

    // ---- cache & context overflow ------------------------------------------------------
    sec('Prompt cache and context overflow');
    row('Context shift', 'context_shift', toggle('Discard oldest tokens when full', v.context_shift, (x) => set('context_shift', x)),
      'Off: generation stops at the context limit (safest). On: rolling window.');
    row('Prompt cache RAM', 'cache_ram_mib', h('div', { class: 'row' }, input(v.cache_ram_mib, (x) => set('cache_ram_mib', Number(x)),
      { type: 'number', cls: 'num', min: -1, step: 512 }), h('span', { class: 'dim' }, 'MiB')), 'Recent prompts kept in system RAM for instant reuse. 0 disables, -1 unlimited.');
    row('Cache reuse chunk', 'cache_reuse', input(v.cache_reuse, (x) => set('cache_reuse', Number(x)), { type: 'number', cls: 'num', min: 0 }),
      'Minimum chunk (tokens) reused from the cache via KV shifting. 0 = off.');

    // ---- speculative ---------------------------------------------------------------------
    sec('Speculative decoding');
    const drafts = model?.draft_candidates || [];
    row('Draft model', 'draft_model', select([['', 'None'], ...drafts.map((d) => [d, d])], v.draft_model, (x) => set('draft_model', x)),
      drafts.length ? 'A small model with the same vocabulary proposes tokens that the main model verifies.' : 'No compatible smaller model (same tokenizer) in the library.');
    row('Draft tokens', 'draft_max', input(v.draft_max, (x) => set('draft_max', Number(x) || 0), { type: 'number', cls: 'num', min: 0, max: 64 }), '0 = engine default.');
    if (eng.spec_types?.length) {
      row('Speculative type', 'spec_type', select([['', 'Engine default'], ...eng.spec_types.filter((t) => t !== 'none').map((t) => [t, t])], v.spec_type,
        (x) => set('spec_type', x)), 'ngram types need no draft model (useful for code and document editing).');
    }

    // ---- rope -----------------------------------------------------------------------------
    sec('RoPE (leave empty to use GGUF values)');
    row('Scaling', 'rope_scaling', select([['', 'From model'], ['none', 'None'], ['linear', 'Linear'], ['yarn', 'YaRN']], v.rope_scaling, (x) => set('rope_scaling', x)));
    row('Frequency base', 'rope_freq_base', input(v.rope_freq_base || '', (x) => set('rope_freq_base', Number(x) || 0), { type: 'number', cls: 'num', placeholder: info.rope_freq_base ? String(info.rope_freq_base) : 'model', style: { width: '120px' } }));
    row('Frequency scale', 'rope_freq_scale', input(v.rope_freq_scale || '', (x) => set('rope_freq_scale', Number(x) || 0), { type: 'number', cls: 'num', step: 0.01, placeholder: 'model' }));
    row('YaRN original context', 'yarn_orig_ctx', input(v.yarn_orig_ctx || '', (x) => set('yarn_orig_ctx', Number(x) || 0), { type: 'number', cls: 'num', placeholder: 'model' }));

    // ---- extra ------------------------------------------------------------------------------
    sec('Advanced');
    row('Embeddings only', 'embeddings', toggle('Serve /v1/embeddings (embedding models)', v.embeddings || info.kind === 'embedding', (x) => set('embeddings', x),
      { disabled: info.kind === 'embedding' }));
    row('Extra arguments', 'extra_args', input(v.extra_args, (x) => set('extra_args', x), { cls: 'mono grow', placeholder: 'e.g. --override-kv tokenizer.ggml.add_bos_token=bool:false' }),
      'Appended verbatim to the llama-server command line.');
    el.appendChild(f);
    refreshTags();
  }

  build();
  el.values = () => ({ ...v });
  el.rebuild = (nv) => { Object.assign(v, nv); build(); };
  return el;
}
