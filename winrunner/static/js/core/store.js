// Central client-side state fed by the telemetry websocket.

import { api, wr } from './api.js';

class Emitter {
  constructor() { this.h = new Map(); }
  on(type, fn) {
    if (!this.h.has(type)) this.h.set(type, new Set());
    this.h.get(type).add(fn);
    return () => this.h.get(type)?.delete(fn);
  }
  emit(type, data) {
    for (const fn of this.h.get(type) || []) {
      try { fn(data); } catch (e) { console.error(`handler for ${type} failed`, e); }
    }
    for (const fn of this.h.get('*') || []) {
      try { fn(type, data); } catch (e) { console.error(e); }
    }
  }
}

const METRICS_MAX = 3600;
const REQ_MAX = 300;

class Store extends Emitter {
  constructor() {
    super();
    this.status = null;
    this.settings = null;
    this.instances = new Map();
    this.metrics = [];
    this.activity = [];
    this.requests = new Map();
    this.tps = [];
    this.slots = new Map();
    this.downloads = new Map();
    this.installState = {};
    this.connected = false;
    this.focusRid = null;
    this.tokenBuf = new Map(); // rid -> pieces (recent requests only)
    this.clockSkew = 0;
    this._ws = null;
    this._retry = 0;
  }

  // ----- derived --------------------------------------------------------------------
  get primary() {
    const list = [...this.instances.values()];
    return list.find((i) => i.state === 'loading' || i.state === 'starting')
      || list.find((i) => i.state === 'ready')
      || list.find((i) => i.state === 'error') || null;
  }
  get activeRequests() { return [...this.requests.values()].filter((r) => !r.t_end); }
  get lastMetrics() { return this.metrics[this.metrics.length - 1] || null; }
  now() { return Date.now() / 1000 - this.clockSkew; }

  // ----- bootstrap --------------------------------------------------------------------
  async loadSettings() {
    this.settings = await api.get(wr('/settings'));
    this.emit('settings', this.settings);
    return this.settings;
  }

  async saveSettings(patch) {
    this.settings = await api.put(wr('/settings'), patch);
    this.emit('settings', this.settings);
    return this.settings;
  }

  async refreshStatus() {
    try {
      this.status = await api.get(wr('/status'));
      for (const i of this.status.instances || []) this.instances.set(i.id, i);
      this.emit('status', this.status);
    } catch { /* offline handled by websocket */ }
  }

  connect() {
    const url = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/wr/ws`;
    const ws = new WebSocket(url);
    this._ws = ws;
    ws.onopen = () => { this._retry = 0; };
    ws.onmessage = (m) => {
      let msg;
      try { msg = JSON.parse(m.data); } catch { return; }
      if (msg.type === 'hello') this._hello(msg);
      else if (msg.type === 'batch') for (const ev of msg.events) this._event(ev);
    };
    ws.onclose = () => {
      if (this.connected) { this.connected = false; this.emit('ws', false); }
      const delay = Math.min(8000, 500 * 2 ** this._retry++);
      setTimeout(() => this.connect(), delay);
    };
    ws.onerror = () => ws.close();
  }

  _hello(msg) {
    this.status = msg.status;
    this.clockSkew = Date.now() / 1000 - (msg.status?.started_at + msg.status?.uptime || Date.now() / 1000);
    this.instances = new Map((msg.status.instances || []).map((i) => [i.id, i]));
    this.metrics = msg.metrics || [];
    this.activity = msg.activity || [];
    this.requests = new Map();
    for (const r of (msg.requests || []).slice().reverse()) this.requests.set(r.id, r);
    this.tps = msg.tps || [];
    this.downloads = new Map((msg.downloads || []).map((j) => [j.id, j]));
    this.installState = msg.install_state || {};
    const act = this.activeRequests;
    this.focusRid = act.length ? act[act.length - 1].id : ([...this.requests.keys()].pop() || null);
    this.connected = true;
    this.emit('ws', true);
    this.emit('hello', msg);
    this.emit('status', this.status);
  }

  _event(ev) {
    switch (ev.type) {
      case 'metrics': {
        this.metrics.push(ev.sample);
        if (this.metrics.length > METRICS_MAX) this.metrics.splice(0, this.metrics.length - METRICS_MAX);
        this.emit('metrics', ev.sample);
        break;
      }
      case 'instance': {
        const prev = this.instances.get(ev.instance.id);
        this.instances.set(ev.instance.id, ev.instance);
        this.emit('instance', { inst: ev.instance, prev });
        break;
      }
      case 'instance_progress': {
        const i = this.instances.get(ev.iid);
        if (i) { i.progress = ev.progress; i.phase = ev.phase; i.phase_label = ev.label; }
        this.emit('instance_progress', ev);
        break;
      }
      case 'instance_removed':
        this.instances.delete(ev.iid);
        this.slots.delete(ev.iid);
        this.emit('instance_removed', ev);
        break;
      case 'activity':
        this.activity.push(ev);
        if (this.activity.length > 500) this.activity.splice(0, this.activity.length - 500);
        this.emit('activity', ev);
        break;
      case 'request': {
        const r = ev.record;
        const isNew = !this.requests.has(r.id);
        this.requests.set(r.id, r);
        if (isNew) {
          this.focusRid = r.id;
          this.tokenBuf.set(r.id, []);
          if (this.tokenBuf.size > 8) this.tokenBuf.delete(this.tokenBuf.keys().next().value);
          if (this.requests.size > REQ_MAX) this.requests.delete(this.requests.keys().next().value);
        }
        if (r.t_end && r.gen_tps) this.tps.push({ t: r.t_end, tg: r.gen_tps, pp: r.prompt_tps, n: r.tokens, id: r.id });
        this.emit('request', { rec: r, isNew });
        break;
      }
      case 'tokens': {
        const buf = this.tokenBuf.get(ev.rid);
        if (buf) {
          buf.push(...ev.pieces);
          if (buf.length > 6000) buf.splice(0, buf.length - 6000);
        }
        const r = this.requests.get(ev.rid);
        if (r && ev.count !== null && ev.count !== undefined) { r.tokens = ev.count; r.live_tps = ev.live_tps; }
        this.emit('tokens', ev);
        break;
      }
      case 'slots':
        this.slots.set(ev.iid, ev.slots);
        this.emit('slots', ev);
        break;
      case 'download':
        this.downloads.set(ev.job.id, ev.job);
        this.emit('download', ev.job);
        break;
      case 'engine_install':
        this.installState = ev;
        this.emit('engine_install', ev);
        break;
      case 'settings':
        this.settings = ev.settings;
        this.emit('settings', ev.settings);
        break;
      default:
        this.emit(ev.type, ev);
    }
  }
}

export const store = new Store();
