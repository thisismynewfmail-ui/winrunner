// Canvas time-series charts (oscilloscope style) with a shared render loop.

const charts = new Set();
let colorCache = null;
let animLevel = 'full';
let glowOn = true;
let rafId = null;
let lastFrame = 0;

export function setChartPrefs({ anim, glow }) {
  if (anim) animLevel = anim;
  if (glow !== undefined) glowOn = glow;
  colorCache = null;
  for (const c of charts) c.dirty = true;
}

export function invalidateColors() {
  colorCache = null;
  for (const c of charts) c.dirty = true;
}

function cssColor(name) {
  if (!name.startsWith('--')) return name;
  if (!colorCache) colorCache = new Map();
  if (!colorCache.has(name)) {
    colorCache.set(name, getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888');
  }
  return colorCache.get(name);
}

function withAlpha(color, a) {
  const c = color.trim();
  if (c.startsWith('#')) {
    let hex = c.slice(1);
    if (hex.length === 3) hex = hex.split('').map((x) => x + x).join('');
    const n = parseInt(hex.slice(0, 6), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }
  const m = c.match(/rgba?\(([^)]+)\)/);
  if (m) {
    const p = m[1].split(',').map((x) => x.trim());
    return `rgba(${p[0]},${p[1]},${p[2]},${a})`;
  }
  return c;
}

function loop(ts) {
  rafId = requestAnimationFrame(loop);
  if (document.hidden) return;
  const smooth = animLevel === 'full';
  if (ts - lastFrame < (smooth ? 100 : 250)) return; // 10 fps scrolling is plenty for 1 Hz telemetry
  lastFrame = ts;
  for (const c of charts) {
    if (!c.visible) continue;
    if (smooth || c.dirty) c.draw();
  }
}

function ensureLoop() {
  if (!rafId) rafId = requestAnimationFrame(loop);
}

const io = typeof IntersectionObserver !== 'undefined'
  ? new IntersectionObserver((entries) => {
    for (const e of entries) {
      const c = e.target.__chart;
      if (c) { c.visible = e.isIntersecting; c.dirty = true; }
    }
  }) : null;

const ro = typeof ResizeObserver !== 'undefined'
  ? new ResizeObserver((entries) => {
    for (const e of entries) {
      const c = e.target.__chart;
      if (c) c.resize();
    }
  }) : null;

export class TimeChart {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {object} opt series: [{key,color,label,fill,width,dash}], yMin, yMax (null=auto), window (s),
   *                     fmt (value -> label), unit, now (() => seconds)
   */
  constructor(canvas, opt) {
    this.c = canvas;
    this.ctx = canvas.getContext('2d');
    this.o = Object.assign({ yMin: 0, yMax: null, window: 300, fmt: (v) => v.toFixed(0), now: () => Date.now() / 1000,
      labels: true, legend: true, headroom: 1.15, minMax: 1 }, opt);
    this.data = [];
    this.visible = true;
    this.dirty = true;
    canvas.__chart = this;
    charts.add(this);
    io?.observe(canvas);
    ro?.observe(canvas);
    this.resize();
    ensureLoop();
  }

  destroy() {
    charts.delete(this);
    io?.unobserve(this.c);
    ro?.unobserve(this.c);
  }

  setWindow(sec) { this.o.window = sec; this.dirty = true; }
  setSeries(series) { this.o.series = series; this.dirty = true; }

  setData(points) { this.data = points.slice(); this.dirty = true; }

  push(t, values) {
    this.data.push({ t, v: values });
    const cutoff = t - 3700;
    while (this.data.length && this.data[0].t < cutoff) this.data.shift();
    this.dirty = true;
  }

  resize() {
    const r = this.c.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(10, Math.round(r.width * dpr));
    const hgt = Math.max(10, Math.round(r.height * dpr));
    if (this.c.width !== w || this.c.height !== hgt) {
      this.c.width = w;
      this.c.height = hgt;
    }
    this.dpr = dpr;
    this.dirty = true;
  }

  draw() {
    this.dirty = false;
    const { ctx } = this;
    const W = this.c.width, H = this.c.height, dpr = this.dpr || 1;
    if (!W || !H) return;
    const o = this.o;
    const now = o.now();
    const t0 = now - o.window;
    const padR = o.labels ? 44 * dpr : 2 * dpr;
    const padT = 4 * dpr, padB = 4 * dpr, padL = 2 * dpr;
    const pw = W - padL - padR, ph = H - padT - padB;
    ctx.clearRect(0, 0, W, H);

    // value range
    let max = o.yMax;
    if (max === null || max === undefined) {
      max = o.minMax;
      for (const p of this.data) {
        if (p.t < t0) continue;
        for (const s of o.series) {
          const v = p.v[s.key];
          if (v !== null && v !== undefined && v > max) max = v;
        }
      }
      max *= o.headroom;
    }
    const min = o.yMin;
    const span = max - min || 1;
    const x = (t) => padL + ((t - t0) / o.window) * pw;
    const y = (v) => padT + ph - ((v - min) / span) * ph;

    // grid (scrolls with time)
    const grid = cssColor('--grid');
    const txt = cssColor('--tx-2');
    ctx.lineWidth = 1;
    ctx.strokeStyle = grid;
    ctx.beginPath();
    for (let i = 0; i <= 4; i++) {
      const yy = Math.round(padT + (ph * i) / 4) + 0.5;
      ctx.moveTo(padL, yy); ctx.lineTo(padL + pw, yy);
    }
    const step = niceStep(o.window / 6);
    for (let t = Math.ceil(t0 / step) * step; t <= now; t += step) {
      const xx = Math.round(x(t)) + 0.5;
      ctx.moveTo(xx, padT); ctx.lineTo(xx, padT + ph);
    }
    ctx.stroke();

    if (o.labels) {
      ctx.fillStyle = txt;
      ctx.font = `${10 * dpr}px ${getComputedStyle(document.body).fontFamily}`;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(o.fmt(max), padL + pw + 5 * dpr, padT + 6 * dpr);
      ctx.fillText(o.fmt(min + span / 2), padL + pw + 5 * dpr, padT + ph / 2);
      ctx.fillText(o.fmt(min), padL + pw + 5 * dpr, padT + ph - 5 * dpr);
    }

    // series
    ctx.save();
    ctx.beginPath();
    ctx.rect(padL, 0, pw, H);
    ctx.clip();
    for (const s of o.series) {
      const color = cssColor(s.color);
      const pts = [];
      let lastPx = -1e9;
      for (const p of this.data) {
        if (p.t < t0 - 5) continue;
        const v = p.v[s.key];
        if (v === null || v === undefined || isNaN(v)) { pts.push(null); continue; }
        const px = x(p.t);
        if (px - lastPx < dpr * 0.75 && pts.length && pts[pts.length - 1]) { // same pixel column: keep the peak
          const prev = pts[pts.length - 1];
          prev[1] = Math.min(prev[1], y(Math.min(max, v)));
          continue;
        }
        lastPx = px;
        pts.push([px, y(Math.min(max, v))]);
      }
      if (!pts.some(Boolean)) continue;
      if (s.fill !== false) {
        let started = false;
        ctx.beginPath();
        let lastX = null;
        let firstX = null;
        for (const pt of pts) {
          if (!pt) continue;
          if (!started) { ctx.moveTo(pt[0], y(min)); ctx.lineTo(pt[0], pt[1]); started = true; firstX = pt[0]; } else ctx.lineTo(pt[0], pt[1]);
          lastX = pt[0];
        }
        if (started) {
          ctx.lineTo(lastX, y(min));
          ctx.lineTo(firstX, y(min));
          const g = ctx.createLinearGradient(0, padT, 0, padT + ph);
          g.addColorStop(0, withAlpha(color, s.fillAlpha ?? 0.28));
          g.addColorStop(1, withAlpha(color, 0.02));
          ctx.fillStyle = g;
          ctx.fill();
        }
      }
      ctx.beginPath();
      let pen = false;
      for (const pt of pts) {
        if (!pt) { pen = false; continue; }
        if (!pen) { ctx.moveTo(pt[0], pt[1]); pen = true; } else ctx.lineTo(pt[0], pt[1]);
      }
      ctx.setLineDash(s.dash ? s.dash.map((d) => d * dpr) : []);
      if (glowOn) { // cheap phosphor glow: wide translucent pass under the line (no shadowBlur)
        ctx.strokeStyle = withAlpha(color, 0.22);
        ctx.lineWidth = ((s.width || 1.5) + 3) * dpr;
        ctx.stroke();
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = (s.width || 1.5) * dpr;
      ctx.stroke();
      ctx.setLineDash([]);
      // head dot
      const last = [...pts].reverse().find(Boolean);
      if (last) {
        ctx.fillStyle = color;
        ctx.fillRect(last[0] - 2 * dpr, last[1] - 2 * dpr, 4 * dpr, 4 * dpr);
      }
    }
    ctx.restore();

    // legend with latest values
    if (o.legend) {
      const lastP = this.data[this.data.length - 1];
      ctx.font = `${10 * dpr}px ${getComputedStyle(document.body).fontFamily}`;
      ctx.textBaseline = 'top';
      ctx.textAlign = 'left';
      let lx = padL + 5 * dpr;
      for (const s of o.series) {
        if (!s.label) continue;
        const v = lastP ? lastP.v[s.key] : null;
        const label = `${s.label} ${v === null || v === undefined ? '-' : (s.fmt || o.fmt)(v)}`;
        ctx.fillStyle = cssColor(s.color);
        ctx.fillRect(lx, padT + 3 * dpr, 7 * dpr, 7 * dpr);
        ctx.fillStyle = cssColor('--tx-1');
        ctx.fillText(label, lx + 10 * dpr, padT + 1.5 * dpr);
        lx += ctx.measureText(label).width + 22 * dpr;
      }
    }
  }
}

function niceStep(v) {
  const steps = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800];
  return steps.find((s) => s >= v) || 1800;
}

/** Simple bar history (e.g. tokens/s per request). */
export class BarChart {
  constructor(canvas, opt) {
    this.c = canvas;
    this.ctx = canvas.getContext('2d');
    this.o = Object.assign({ max: null, color: '--c-1', color2: '--c-2', fmt: (v) => v.toFixed(1) }, opt);
    this.data = [];
    canvas.__chart = this;
    this.visible = true;
    this.dirty = true;
    charts.add(this);
    ro?.observe(canvas);
    io?.observe(canvas);
    this.resize();
    ensureLoop();
  }
  destroy() { charts.delete(this); ro?.unobserve(this.c); io?.unobserve(this.c); }
  resize() { TimeChart.prototype.resize.call(this); }
  setData(d) { this.data = d.slice(-60); this.dirty = true; }
  draw() {
    this.dirty = false;
    const { ctx } = this;
    const W = this.c.width, H = this.c.height, dpr = this.dpr || 1;
    ctx.clearRect(0, 0, W, H);
    const n = Math.max(20, this.data.length);
    const max = (this.o.max || Math.max(1, ...this.data.map((d) => d.a || 0))) * 1.1;
    const bw = W / n;
    ctx.strokeStyle = cssColor('--grid');
    ctx.beginPath();
    for (let i = 1; i < 4; i++) { const yy = Math.round((H * i) / 4) + 0.5; ctx.moveTo(0, yy); ctx.lineTo(W, yy); }
    ctx.stroke();
    this.data.forEach((d, i) => {
      const hA = ((d.a || 0) / max) * (H - 14 * dpr);
      const xx = W - (this.data.length - i) * bw;
      ctx.fillStyle = cssColor(this.o.color);
      ctx.fillRect(xx + 1 * dpr, H - hA, Math.max(1, bw - 3 * dpr), hA);
    });
    ctx.fillStyle = cssColor('--tx-2');
    ctx.font = `${10 * dpr}px ${getComputedStyle(document.body).fontFamily}`;
    ctx.textBaseline = 'top';
    ctx.fillText(`max ${this.o.fmt(max / 1.1)}`, 4 * dpr, 3 * dpr);
  }
}
