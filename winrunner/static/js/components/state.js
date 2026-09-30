// Global engine state summary shared by the header, sidebar and window title.

import { store } from '../core/store.js';

export function engineState() {
  if (!store.connected) return { key: 'offline', label: 'OFFLINE', led: 'err blink', detail: 'Connection to WinRunner lost' };
  const st = store.status;
  if (st && !st.engine) return { key: 'noengine', label: 'NO ENGINE', led: 'warn blink', detail: 'Install llama.cpp in Settings > Engine' };
  if (st && st.server && !st.server.api_enabled) return { key: 'stopped', label: 'API STOPPED', led: 'warn', detail: 'The API server is stopped' };
  const insts = [...store.instances.values()];
  const loading = insts.find((i) => i.state === 'loading' || i.state === 'starting');
  if (loading) {
    return { key: 'loading', label: `LOADING ${Math.round((loading.progress || 0) * 100)}%`, led: 'acc blink',
      detail: `${loading.model} · ${loading.phase_label || loading.phase}`, inst: loading };
  }
  const act = store.activeRequests;
  if (act.some((r) => r.phase === 'generating')) {
    const r = act.find((x) => x.phase === 'generating');
    return { key: 'gen', label: 'GENERATING', led: 'ok pulse', detail: r.model, req: r };
  }
  if (act.some((r) => r.phase === 'prompt' || r.phase === 'queued')) {
    const r = act.find((x) => x.phase === 'prompt' || x.phase === 'queued');
    return { key: 'prompt', label: r.phase === 'queued' ? 'QUEUED' : 'PROMPT EVAL', led: 'acc pulse', detail: r.model, req: r };
  }
  if (act.some((r) => r.phase === 'loading_model')) return { key: 'loading', label: 'LOADING', led: 'acc blink', detail: 'Just-in-time model load' };
  const ready = insts.find((i) => i.state === 'ready');
  if (ready) return { key: 'ready', label: 'READY', led: 'ok', detail: ready.model, inst: ready };
  const err = insts.find((i) => i.state === 'error');
  if (err?.recovering) return { key: 'loading', label: 'RESTARTING', led: 'warn blink', detail: `${err.model} · ${err.error}`, inst: err };
  if (err) return { key: 'error', label: 'ERROR', led: 'err', detail: err.error || err.model, inst: err };
  return { key: 'idle', label: 'IDLE', led: '', detail: 'No model loaded' };
}
