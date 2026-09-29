// WinRunner control panel bootstrap and tab router.

import { h, $, clear, icon, initTooltips } from './core/dom.js';
import { store } from './core/store.js';
import { applyUI } from './core/theme.js';
import { mountHeader } from './components/header.js';
import { mountSidebar } from './components/sidebar.js';
import { mountStatusBar } from './components/statusbar.js';
import { runBoot } from './components/boot.js';
import { toast } from './components/ui.js';

const PAGES = [
  { id: 'library', label: 'Library', icon: 'model', load: () => import('./pages/library.js') },
  { id: 'server', label: 'Server', icon: 'server', load: () => import('./pages/server.js') },
  { id: 'chat', label: 'Chat', icon: 'chat', load: () => import('./pages/chat.js') },
  { id: 'monitor', label: 'Monitor', icon: 'monitor', load: () => import('./pages/monitor.js') },
  { id: 'bench', label: 'Benchmark', icon: 'bench', load: () => import('./pages/bench.js') },
  { id: 'logs', label: 'Logs', icon: 'log', load: () => import('./pages/logs.js') },
  { id: 'settings', label: 'Settings', icon: 'gear', load: () => import('./pages/settings.js') },
];

let current = null;
let unmount = null;

async function show(id) {
  const page = PAGES.find((p) => p.id === id) || PAGES[0];
  if (current === page.id) return;
  current = page.id;
  for (const t of document.querySelectorAll('.tab')) t.classList.toggle('on', t.dataset.page === page.id);
  const host = $('#page');
  if (unmount) { try { unmount(); } catch (e) { console.error(e); } unmount = null; }
  clear(host);
  const wrap = h('div', { class: `pg pg-${page.id}` });
  host.appendChild(wrap);
  host.scrollTop = 0;
  try {
    const mod = await page.load();
    if (current !== page.id) return;
    unmount = mod.mount(wrap) || null;
  } catch (e) {
    console.error(e);
    wrap.appendChild(h('div', { class: 'note err' }, `Failed to open ${page.label}: ${e.message}`));
  }
}

export function navigate(id, params) {
  if (params) sessionStorageSet('nav-params', JSON.stringify(params));
  location.hash = `#${id}`;
}

function sessionStorageSet(k, v) { try { sessionStorage.setItem(k, v); } catch { /* ignore */ } }
export function takeNavParams() {
  try { const v = sessionStorage.getItem('nav-params'); sessionStorage.removeItem('nav-params'); return v ? JSON.parse(v) : null; } catch { return null; }
}

async function main() {
  initTooltips();
  let settings = null;
  try {
    settings = await store.loadSettings();
    applyUI(settings.ui);
  } catch (e) {
    toast(`Could not load settings: ${e.message}`, 'err', 8000);
  }
  store.on('settings', (s) => applyUI(s.ui));

  const tabs = $('#tabs');
  for (const p of PAGES) {
    tabs.appendChild(h('div', { class: 'tab', role: 'tab', tabindex: '0', 'data-page': p.id,
      onclick: () => navigate(p.id), onkeydown: (e) => { if (e.key === 'Enter') navigate(p.id); } }, icon(p.icon), p.label));
  }
  mountHeader($('#hdr'));
  mountSidebar($('#side'));
  mountStatusBar($('#status'));

  let offline = null;
  store.on('ws', (up) => {
    if (!up && !offline) { offline = h('div', { id: 'offline' }, 'CONNECTION TO WINRUNNER LOST - RECONNECTING'); document.body.appendChild(offline); }
    if (up && offline) { offline.remove(); offline = null; }
  });
  store.on('load_error', (ev) => toast(ev.error, 'err', 9000, `Load failed: ${ev.model}`));
  store.on('instance', ({ inst, prev }) => {
    if (inst.state === 'ready' && prev && prev.state !== 'ready') toast(`${inst.model} is ready (${inst.load?.load_seconds ?? '?'} s)`, 'ok', 4000, 'Model loaded');
  });
  store.connect();
  const vis = () => document.documentElement.classList.toggle('hidden-window', document.hidden);
  document.addEventListener('visibilitychange', vis);
  vis();
  window.addEventListener('hashchange', () => show(location.hash.slice(1)));
  show(location.hash.slice(1) || 'library');
  runBoot($('#boot'), settings?.ui?.boot_sequence !== false && !sessionStorage.getItem('booted'));
  try { sessionStorage.setItem('booted', '1'); } catch { /* ignore */ }
}

main();
