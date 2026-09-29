// Number / size / time formatting.

const MiB = 1024 * 1024;
const GiB = 1024 * MiB;

export function bytes(n, digits = 1) {
  if (n === null || n === undefined || isNaN(n)) return '-';
  const a = Math.abs(n);
  if (a >= 1024 ** 4) return `${(n / 1024 ** 4).toFixed(digits)} TiB`;
  if (a >= GiB) return `${(n / GiB).toFixed(digits)} GiB`;
  if (a >= MiB) return `${(n / MiB).toFixed(a >= 100 * MiB ? 0 : digits)} MiB`;
  if (a >= 1024) return `${(n / 1024).toFixed(0)} KiB`;
  return `${n} B`;
}

export function mib(m, digits = 1) {
  if (m === null || m === undefined || isNaN(m)) return '-';
  return Math.abs(m) >= 1024 ? `${(m / 1024).toFixed(digits)} GiB` : `${Math.round(m)} MiB`;
}

export function gib(n, digits = 1) {
  if (n === null || n === undefined || isNaN(n)) return '-';
  return (n / GiB).toFixed(digits);
}

export function num(n, digits = 0) {
  if (n === null || n === undefined || isNaN(n)) return '-';
  return Number(n).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function params(n) {
  if (!n) return '-';
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 1e11 ? 0 : 2)}B`;
  if (n >= 1e6) return `${Math.round(n / 1e6)}M`;
  return String(n);
}

export function ctx(n) {
  if (!n) return '-';
  if (n >= 1024 && n % 1024 === 0) return `${n / 1024}K`;
  return num(n);
}

export function dur(sec) {
  if (sec === null || sec === undefined || isNaN(sec)) return '-';
  sec = Math.max(0, sec);
  if (sec < 1) return `${Math.round(sec * 1000)} ms`;
  if (sec < 60) return `${sec.toFixed(sec < 10 ? 2 : 1)} s`;
  const d = Math.floor(sec / 86400), hh = Math.floor((sec % 86400) / 3600), mm = Math.floor((sec % 3600) / 60), ss = Math.floor(sec % 60);
  if (d) return `${d}d ${hh}h ${mm}m`;
  if (hh) return `${hh}h ${String(mm).padStart(2, '0')}m`;
  return `${mm}m ${String(ss).padStart(2, '0')}s`;
}

export function uptime(sec) {
  sec = Math.max(0, Math.floor(sec || 0));
  const d = Math.floor(sec / 86400), hh = Math.floor((sec % 86400) / 3600), mm = Math.floor((sec % 3600) / 60), ss = sec % 60;
  const t = `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
  return d ? `${d}d ${t}` : t;
}

export function ms(v) {
  if (v === null || v === undefined || isNaN(v)) return '-';
  return v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${Math.round(v)} ms`;
}

export function tps(v) {
  if (v === null || v === undefined || isNaN(v) || v === 0) return '-';
  return v >= 100 ? v.toFixed(0) : v.toFixed(1);
}

let h24 = true;
export function set24h(v) { h24 = v; }

export function time(t, withSec = true) {
  const d = new Date(t * 1000);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: withSec ? '2-digit' : undefined, hour12: !h24 });
}

export function dateTime(t) {
  const d = new Date(t * 1000);
  return `${d.toLocaleDateString()} ${time(t, false)}`;
}

export function ago(t) {
  if (!t) return 'never';
  const s = Date.now() / 1000 - t;
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return `${Math.floor(s / 86400)} d ago`;
}

export function pct(v, digits = 0) {
  if (v === null || v === undefined || isNaN(v)) return '-';
  return `${Number(v).toFixed(digits)}%`;
}
