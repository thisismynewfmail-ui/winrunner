// Themes and display preferences.

import { invalidateColors, setChartPrefs } from './charts.js';
import { set24h } from './fmt.js';

export const THEMES = [
  { id: 'classic', name: 'Classic Olive', desc: 'Olive and khaki panels with beveled controls. The default.',
    sw: ['#3e4637', '#4c5844', '#c4b550', '#dee5d7'] },
  { id: 'amber', name: 'Amber', desc: 'Charcoal panels with amber readouts. High contrast at night.',
    sw: ['#1b1a17', '#27251f', '#ffb000', '#f0e2c6'] },
  { id: 'graphite', name: 'Graphite', desc: 'Neutral dark grey with cyan accents.',
    sw: ['#23272b', '#2f3439', '#5fb6dc', '#e3e8ec'] },
  { id: 'steel', name: 'Steel', desc: 'Light grey classic desktop palette with navy selection.',
    sw: ['#d4d0c8', '#ffffff', '#0a246a', '#000000'] },
  { id: 'phosphor', name: 'Phosphor', desc: 'Green-on-black monochrome monitor.',
    sw: ['#031003', '#062006', '#46ff66', '#8dff9a'] },
];

// Tokens exposed in the custom theme editor.
export const EDITABLE = [
  ['--bg-0', 'Backdrop'], ['--bg-1', 'Panel'], ['--bg-2', 'Control face'], ['--bg-3', 'Field / well'],
  ['--bd-hi', 'Bevel light'], ['--bd-lo', 'Bevel shadow'], ['--tx-0', 'Text'], ['--tx-1', 'Text secondary'],
  ['--tx-2', 'Text dim'], ['--acc', 'Accent'], ['--acc-2', 'Selection'], ['--ok', 'Good'], ['--warn', 'Warning'],
  ['--err', 'Error'], ['--c-1', 'Chart 1'], ['--c-2', 'Chart 2'], ['--c-3', 'Chart 3'],
];

let appliedCustom = [];

export function currentTokens() {
  const cs = getComputedStyle(document.documentElement);
  const out = {};
  for (const [k] of EDITABLE) out[k] = cs.getPropertyValue(k).trim();
  return out;
}

export function toHex(color) {
  const c = color.trim();
  if (/^#[0-9a-f]{6}$/i.test(c)) return c;
  if (/^#[0-9a-f]{3}$/i.test(c)) return `#${c.slice(1).split('').map((x) => x + x).join('')}`;
  const m = c.match(/rgba?\(([^)]+)\)/);
  if (m) {
    const [r, g, b] = m[1].split(',').map((x) => parseInt(x, 10));
    return `#${[r, g, b].map((v) => v.toString(16).padStart(2, '0')).join('')}`;
  }
  return '#808080';
}

export function applyUI(ui) {
  const root = document.documentElement;
  for (const k of appliedCustom) root.style.removeProperty(k);
  appliedCustom = [];
  let theme = ui.theme || 'classic';
  if (theme.startsWith('custom:')) {
    const name = theme.slice(7);
    const custom = ui.custom_themes?.[name];
    if (custom) {
      root.dataset.theme = custom.base || 'classic';
      for (const [k, v] of Object.entries(custom)) {
        if (k.startsWith('--')) { root.style.setProperty(k, v); appliedCustom.push(k); }
      }
      if (custom['--acc']) { root.style.setProperty('--glow', `${custom['--acc']}88`); appliedCustom.push('--glow'); }
    } else {
      theme = 'classic';
      root.dataset.theme = theme;
    }
  } else {
    root.dataset.theme = theme;
  }
  root.dataset.crt = ui.crt_effect ? 'on' : 'off';
  root.dataset.glow = ui.glow ? 'on' : 'off';
  root.dataset.anim = ui.animations || 'full';
  root.dataset.side = ui.sidebar_position || 'right';
  root.style.setProperty('--ui-scale', String(ui.ui_scale || 1));
  set24h(ui.clock_24h !== false);
  setChartPrefs({ anim: ui.animations || 'full', glow: !!ui.glow });
  invalidateColors();
}
