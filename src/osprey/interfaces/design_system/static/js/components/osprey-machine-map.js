// @ts-check
/**
 * <osprey-machine-map> — the machine-map device selector.
 *
 * Draws a closed machine as a circle with every device of one family placed at
 * its position along the machine, plus one row of cells per region below it.
 * Each device is in use (`in`) or out of use (`out`), and may carry an
 * uncommitted change (`pending`). A family with two planes (say horizontal and
 * vertical correctors) puts its second lane on an inner track and its region
 * rows split into two plane rows.
 *
 * The element is generic over any family with a position and a region: every
 * label (region names, centre text, the marker) comes from the model the host
 * hands it, so nothing about a facility is written here. Styling lives in
 * components.css (`.map-wrap`, `svg.machine-map`, `.map-key`, `.sector-rows`).
 *
 * Usage:
 *
 *     import '/design-system/js/components/osprey-machine-map.js';
 *     const map = document.querySelector('osprey-machine-map');
 *     map.model = {
 *       label: 'BPMs',                // aria-label of the map
 *       length: 196.8,                // total length; positions are 0..length
 *       regions: [{ id: 1, label: 'Cell 1', mapLabel: '01', start: 0, end: 16.4 }, ...],
 *       devices: [{ id: 'bpm-1', label: 'BPM 1', position: 1.2, region: 1,
 *                   lane: 0, state: 'in', pending: false }, ...],
 *       focus: 1,                     // the region the host's detail shows
 *       marker: { position: 0.2, label: 'INJ' },          // optional
 *       center: { value: '68 / 72', caption: 'in use', note: '' },  // optional
 *       key: true,                    // optional: draw the in / out / pending key
 *     };
 *     map.addEventListener('machine-map:toggle', (e) => toggle(e.detail.device));
 *     map.addEventListener('machine-map:focus', (e) => show(e.detail.region));
 *
 * The element never changes a device's state itself: a dot click only asks
 * (`machine-map:toggle`), and the host answers by setting a new model. A
 * region click moves the focus at once (it is view state) and announces it
 * (`machine-map:focus`). Light DOM, no shadow root, registration guarded —
 * the conventions <osprey-drawer> set for every component.
 *
 * @module components/osprey-machine-map
 */

import { escapeHtml } from '/design-system/js/dom.js';

/**
 * @typedef {object} MapRegion
 * @property {string|number} id
 * @property {string} label       Row label (and map label when `mapLabel` is absent).
 * @property {string} [mapLabel]  Short label drawn on the map.
 * @property {number} start       Position where the region starts.
 * @property {number} end         Position where the region ends.
 */

/**
 * @typedef {object} MapDevice
 * @property {string} id
 * @property {string} label
 * @property {number} position    Position along the machine, 0..length.
 * @property {string|number} region
 * @property {number} [lane]      0 (outer track, default) or 1 (inner track).
 * @property {'in'|'out'} state
 * @property {boolean} [pending]
 */

/**
 * @typedef {object} MapModel
 * @property {string} [label]
 * @property {number} length
 * @property {MapRegion[]} regions
 * @property {MapDevice[]} devices
 * @property {string|number} [focus]
 * @property {{position: number, label?: string}} [marker]
 * @property {{value: string, caption?: string, note?: string}} [center]
 * @property {boolean} [key]
 */

/** Drawing size of the square viewBox. */
const SIZE = 300;
/** Radius of the outer track. */
const RADIUS = 115;
/** Distance from one lane's track to the next one in. */
const LANE_STEP = 14;

/**
 * The point at `position` on a circle of `radius` around `center`. Position 0
 * sits at the top; positions run clockwise and `length` comes back to the top.
 *
 * @param {number} position
 * @param {number} length
 * @param {number} radius
 * @param {[number, number]} center
 * @returns {[number, number]}
 */
export function mapPoint(position, length, radius, center) {
  const angle = -Math.PI / 2 + (2 * Math.PI * position) / length;
  return [center[0] + radius * Math.cos(angle), center[1] + radius * Math.sin(angle)];
}

/**
 * How many of a region's devices are in use.
 *
 * @param {MapDevice[]} devices
 * @param {string|number} region
 * @returns {{used: number, total: number}}
 */
export function regionSummary(devices, region) {
  const list = devices.filter((device) => String(device.region) === String(region));
  return { used: list.filter((device) => device.state !== 'out').length, total: list.length };
}

/** @param {number} n */
const fix = (n) => n.toFixed(1);

/**
 * @param {MapModel} model
 * @returns {string}
 */
function svgMarkup(model) {
  const center = /** @type {[number, number]} */ ([SIZE / 2, SIZE / 2]);
  const lanes = model.devices.some((device) => (device.lane ?? 0) > 0) ? 2 : 1;
  const at = (/** @type {number} */ s, /** @type {number} */ r) => mapPoint(s, model.length, r, center);
  const focus = model.focus === undefined ? null : String(model.focus);
  const label = escapeHtml(model.label ?? 'Machine map');
  let out = `<svg class="machine-map" viewBox="0 0 ${SIZE} ${SIZE}" role="group" aria-label="${label}">`;

  for (const region of model.regions) {
    const pad = Math.min(0.4, (region.end - region.start) / 10);
    const [x0, y0] = at(region.start + pad, RADIUS);
    const [x1, y1] = at(region.end - pad, RADIUS);
    const large = region.end - region.start > model.length / 2 ? 1 : 0;
    const isFocus = String(region.id) === focus ? ' is-focus' : '';
    out +=
      `<path class="sector-arc${isFocus}" data-region="${escapeHtml(region.id)}"` +
      ` d="M${fix(x0)},${fix(y0)} A${RADIUS},${RADIUS} 0 ${large} 1 ${fix(x1)},${fix(y1)}">` +
      `<title>${escapeHtml(region.label)}</title></path>`;
  }
  for (let lane = 0; lane < lanes; lane++) {
    out += `<circle class="map-path" cx="${center[0]}" cy="${center[1]}" r="${RADIUS - lane * LANE_STEP}"/>`;
  }
  for (const region of model.regions) {
    const [a, b] = at(region.start, RADIUS - 5);
    const [c, d] = at(region.start, RADIUS + 5);
    out += `<line class="tick" x1="${fix(a)}" y1="${fix(b)}" x2="${fix(c)}" y2="${fix(d)}"/>`;
    const [lx, ly] = at((region.start + region.end) / 2, RADIUS + 22);
    const isFocus = String(region.id) === focus ? ' is-focus' : '';
    out +=
      `<text class="sector-label${isFocus}" x="${fix(lx)}" y="${fix(ly)}">` +
      `${escapeHtml(region.mapLabel ?? region.label)}</text>`;
  }
  if (model.marker) {
    const [ix, iy] = at(model.marker.position, RADIUS + 10);
    out += `<path class="inj" d="M${fix(ix - 4)},${fix(iy - 7)} L${fix(ix + 4)},${fix(iy - 7)} L${fix(ix)},${fix(iy - 1)} Z"/>`;
    if (model.marker.label) {
      out += `<text class="inj-label" x="${fix(ix + 7)}" y="${fix(iy - 2)}">${escapeHtml(model.marker.label)}</text>`;
    }
  }
  for (const device of model.devices) {
    const lane = device.lane ?? 0;
    const [x, y] = at(device.position, RADIUS - lane * LANE_STEP);
    const inFocus = String(device.region) === focus;
    const r = (lanes > 1 ? 3 : 3.6) + (inFocus ? 1 : 0);
    out +=
      `<circle class="dev" cx="${fix(x)}" cy="${fix(y)}" r="${r}"` +
      ` data-device="${escapeHtml(device.id)}" data-state="${device.state === 'out' ? 'out' : 'in'}"` +
      ` data-pending="${device.pending ? 'true' : 'false'}">` +
      `<title>${escapeHtml(device.label)} · s = ${device.position.toFixed(1)}</title></circle>`;
  }
  if (model.center) {
    const { value, caption, note } = model.center;
    out += `<text class="center-big" x="${center[0]}" y="${center[1] + 2}">${escapeHtml(value)}</text>`;
    if (caption) {
      out += `<text class="center-small" x="${center[0]}" y="${center[1] + 20}">${escapeHtml(caption)}</text>`;
    }
    if (note) {
      out += `<text class="center-small center-note" x="${center[0]}" y="${center[1] - 30}">${escapeHtml(note)}</text>`;
    }
  }
  return `${out}</svg>`;
}

/**
 * @param {MapDevice} device
 * @returns {string}
 */
function cellMarkup(device) {
  return (
    `<span class="cell" data-device="${escapeHtml(device.id)}"` +
    ` data-state="${device.state === 'out' ? 'out' : 'in'}"` +
    ` data-pending="${device.pending ? 'true' : 'false'}" title="${escapeHtml(device.label)}"></span>`
  );
}

/**
 * @param {MapModel} model
 * @returns {string}
 */
function rowsMarkup(model) {
  const lanes = model.devices.some((device) => (device.lane ?? 0) > 0) ? 2 : 1;
  const focus = model.focus === undefined ? null : String(model.focus);
  let out = '<div class="sector-rows">';
  for (const region of model.regions) {
    const list = model.devices
      .filter((device) => String(device.region) === String(region.id))
      .sort((a, b) => a.position - b.position);
    const { used, total } = regionSummary(model.devices, region.id);
    let cells;
    if (lanes > 1) {
      const plane = (/** @type {number} */ lane) =>
        `<span class="plane">${list.filter((d) => (d.lane ?? 0) === lane).map(cellMarkup).join('')}</span>`;
      cells = `<span class="cells planes">${plane(0)}${plane(1)}</span>`;
    } else {
      cells = `<span class="cells">${list.map(cellMarkup).join('')}</span>`;
    }
    const isFocus = String(region.id) === focus;
    out +=
      `<div class="sector-row${isFocus ? ' is-focus' : ''}" data-region="${escapeHtml(region.id)}"` +
      ` role="button" tabindex="0" aria-pressed="${isFocus ? 'true' : 'false'}">` +
      `<span class="sec">${escapeHtml(region.label)}</span>${cells}` +
      `<span class="cnt">${used}/${total}</span></div>`;
  }
  return `${out}</div>`;
}

const KEY_MARKUP =
  '<div class="map-key"><span><i class="in"></i>in use</span>' +
  '<span><i class="out"></i>out</span><span><i class="pend"></i>pending</span></div>';

export class OspreyMachineMap extends HTMLElement {
  constructor() {
    super();
    /** @type {MapModel|null} */
    this._model = null;
    this._onClick = this._onClick.bind(this);
    this._onKeydown = this._onKeydown.bind(this);
  }

  /** @returns {MapModel|null} */
  get model() {
    return this._model;
  }

  /** @param {MapModel|null} value */
  set model(value) {
    this._model = value;
    this.render();
  }

  connectedCallback() {
    this.addEventListener('click', this._onClick);
    this.addEventListener('keydown', this._onKeydown);
    this.render();
  }

  disconnectedCallback() {
    this.removeEventListener('click', this._onClick);
    this.removeEventListener('keydown', this._onKeydown);
  }

  /** Redraw from the current model; an element without a model stays empty. */
  render() {
    if (!this._model) {
      this.replaceChildren();
      return;
    }
    const model = this._model;
    this.innerHTML =
      `<div class="map-wrap">${svgMarkup(model)}</div>` +
      (model.key ? KEY_MARKUP : '') +
      rowsMarkup(model);
  }

  /**
   * Move the focus to the region whose id reads `raw` and announce it.
   * @param {string} raw
   */
  _focus(raw) {
    if (!this._model) return;
    const region = this._model.regions.find((s) => String(s.id) === raw);
    if (!region) return;
    this._model = { ...this._model, focus: region.id };
    this.render();
    this.dispatchEvent(
      new CustomEvent('machine-map:focus', {
        detail: { region: region.id },
        bubbles: true,
        composed: true,
      }),
    );
  }

  /** @param {Event} event */
  _onClick(event) {
    if (!(event.target instanceof Element)) return;
    const dot = event.target.closest('[data-device]');
    if (dot && dot.classList.contains('dev')) {
      this.dispatchEvent(
        new CustomEvent('machine-map:toggle', {
          detail: { device: dot.getAttribute('data-device') },
          bubbles: true,
          composed: true,
        }),
      );
      return;
    }
    const region = event.target.closest('[data-region]');
    if (region) this._focus(region.getAttribute('data-region') ?? '');
  }

  /** @param {KeyboardEvent} event */
  _onKeydown(event) {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    if (!(event.target instanceof Element)) return;
    const row = event.target.closest('.sector-row');
    if (!row) return;
    event.preventDefault();
    this._focus(row.getAttribute('data-region') ?? '');
  }
}

if (!customElements.get('osprey-machine-map')) {
  customElements.define('osprey-machine-map', OspreyMachineMap);
}
