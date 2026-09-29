// Keyboard shortcuts: F2 hides/shows the title bar and page tabs, F11 toggles full screen.

import { api, wr } from '../core/api.js';
import { store } from '../core/store.js';
import { toast } from './ui.js';

const CHROME_KEY = 'wr-chrome-hidden';

function chromeHidden() { return document.documentElement.classList.contains('chrome-hidden'); }

export function setChromeHidden(hidden, announce = true) {
  document.documentElement.classList.toggle('chrome-hidden', hidden);
  try { localStorage.setItem(CHROME_KEY, hidden ? '1' : '0'); } catch { /* storage unavailable */ }
  if (announce && hidden) toast('Press F2 to show them again.', 'info', 4000, 'Title bar and tabs hidden');
}

async function toggleFullscreen() {
  try {
    const r = await api.post(wr('/app/fullscreen'));
    if (r.fullscreen) toast('Press F11 to leave full screen.', 'info', 3000, 'Full screen');
  } catch (e) {
    toast(e.message, 'err', 5000, 'Full screen');
  }
}

export function initShortcuts() {
  let hidden = false;
  try { hidden = localStorage.getItem(CHROME_KEY) === '1'; } catch { /* storage unavailable */ }
  if (hidden) setChromeHidden(true);
  document.addEventListener('keydown', (e) => {
    if (e.repeat || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
    if (e.key === 'F2') {
      e.preventDefault();
      setChromeHidden(!chromeHidden());
    } else if (e.key === 'F11' && store.status?.window) {
      // The app window has no full-screen mode of its own; in a browser, F11 stays the browser's.
      e.preventDefault();
      toggleFullscreen();
    }
  });
}
