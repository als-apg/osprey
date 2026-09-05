// @ts-check
/**
 * Unit tests for the graph-finder pure markup builders (graph-finder-render.js).
 * Pure string builders, no DOM writes of their own — every assertion parses the
 * returned markup into a detached element and inspects the result:
 *   npx vitest run tests/interfaces/channel_finder/graph-finder-render.test.mjs
 *
 * Runs under happy-dom (vitest.config.js); imports resolve '/design-system/js'
 * via the configured alias.
 */

import { test, expect } from 'vitest';

import {
  facetRailHtml,
  chipsHtml,
  resultsHtml,
  footerHtml,
  deviceCardHtml,
  deviceCardErrorHtml,
  directionOf,
  fmt,
} from '../../../src/osprey/interfaces/channel_finder/static/js/graph-finder-render.js';

const PAYLOAD = '<img src=x onerror=alert(1)>';

/**
 * Parse a markup string into a detached container.
 * @param {string} html
 * @returns {HTMLElement}
 */
function parse(html) {
  const container = document.createElement('div');
  container.innerHTML = html;
  return container;
}

/**
 * Assert a payload reached the sink as inert text: no live node it would have
 * created, and no event-handler attribute anywhere in the parsed subtree.
 * @param {string} html
 * @param {string} label
 */
function assertInert(html, label) {
  // The payload must never survive as markup in the string itself. The word
  // "onerror" does survive, as text inside an escaped `&lt;img …&gt;` — that is
  // the point of escaping, so inertness is asserted on the parsed result below.
  expect(html.includes('<img'), `${label}: no raw <img in markup`).toBe(false);
  expect(html.includes('&lt;img'), `${label}: payload reached the sink escaped`).toBe(true);

  const container = parse(html);
  expect(container.querySelector('img'), `${label}: no live <img> node`).toBeNull();
  expect(container.querySelector('script'), `${label}: no live <script> node`).toBeNull();
  const hasOnAttr = [...container.querySelectorAll('*')].some(el =>
    [...el.attributes].some(attr => attr.name.startsWith('on'))
  );
  expect(hasOnAttr, `${label}: no on* event-handler attribute`).toBe(false);
  // Nothing named onerror survived as an attribute anywhere.
  expect(container.querySelector('[onerror]'), `${label}: no onerror attribute`).toBeNull();
}

/**
 * A finder state with empty facet selections.
 * @param {Partial<{section: Set<string>, system: Set<string>, signal: Set<string>,
 *   dir: Set<string>, cls: string | null, selected: Set<string>}>} [over]
 */
function makeState(over = {}) {
  return {
    section: new Set(),
    system: new Set(),
    signal: new Set(),
    dir: new Set(),
    cls: null,
    selected: new Set(),
    ...over,
  };
}

/** A facets payload covering all five groups. */
function makeFacets() {
  return {
    section: [{ value: 'SR', count: 12 }, { value: 'BR', count: 3 }],
    system: [{ value: 'MAG', count: 9 }],
    class: [{ value: 'http://x/Quadrupole', count: 4 }],
    signal: [{ value: 'current', count: 7 }],
    dir: [{ value: 'R', count: 5 }, { value: 'W', count: 2 },
      { value: 'RW', count: 1 }, { value: 'none', count: 6 }],
  };
}

/** A two-level ontology forest. */
function makeTree() {
  return [{
    uri: 'http://x/Magnet',
    name: 'Magnet',
    abstract: true,
    children: [
      { uri: 'http://x/Quadrupole', name: 'Quadrupole', abstract: false, children: [] },
      { uri: 'http://x/Dipole', name: 'Dipole', abstract: false, children: [] },
    ],
  }];
}

/**
 * One result row.
 * @param {Partial<Record<string, any>>} [over]
 */
function makeRow(over = {}) {
  return {
    fullPv: 'SR:QF1:current',
    description: 'Quadrupole current',
    device: 'QF1',
    device_uri: 'http://x/dev/QF1',
    section: 'SR',
    system: 'MAG',
    edges: ['READSSIGNAL'],
    signals: [{ uri: 'http://x/sig/current', name: 'current' }],
    ...over,
  };
}

// ---------------------------------------------------------------------------
// Direction derivation
// ---------------------------------------------------------------------------

test('fmt localises a count and falls back to zero for a non-number', () => {
  expect(fmt(1234)).toBe((1234).toLocaleString());
  expect(fmt(0)).toBe('0');
  expect(fmt(undefined)).toBe('0');
  expect(fmt(NaN)).toBe('0');
  expect(fmt('not a number')).toBe('0');
});

test('directionOf derives R / W / RW / none from the row edges', () => {
  expect(directionOf(makeRow({ edges: ['READSSIGNAL'] }))).toBe('R');
  expect(directionOf(makeRow({ edges: ['WRITESSIGNAL'] }))).toBe('W');
  expect(directionOf(makeRow({ edges: ['READSSIGNAL', 'WRITESSIGNAL'] }))).toBe('RW');
  expect(directionOf(makeRow({ edges: ['WRITESSIGNAL', 'READSSIGNAL'] }))).toBe('RW');
  expect(directionOf(makeRow({ edges: [] }))).toBe('none');
  expect(directionOf(makeRow({ edges: undefined }))).toBe('none');
  // An edge type that is neither read nor write leaves the row undirected.
  expect(directionOf(makeRow({ edges: ['HASBINDING'] }))).toBe('none');
});

// ---------------------------------------------------------------------------
// facetRailHtml
// ---------------------------------------------------------------------------

test('facetRailHtml renders every facet group as buttons carrying the click contract', () => {
  const container = parse(facetRailHtml(makeFacets(), makeTree(), makeState()));

  const items = [...container.querySelectorAll('.facet-item')];
  expect(items.length).toBeGreaterThan(0);
  for (const el of items) {
    expect(el.tagName).toBe('BUTTON');
    expect(el.getAttribute('type')).toBe('button');
    expect(el.getAttribute('data-facet')).toBeTruthy();
    expect(el.hasAttribute('data-value')).toBe(true);
  }

  const facetsUsed = new Set(items.map(el => el.getAttribute('data-facet')));
  expect(facetsUsed).toEqual(new Set(['section', 'system', 'cls', 'signal', 'dir']));

  const section = container.querySelector('[data-facet="section"][data-value="SR"]');
  expect(section).not.toBeNull();
  expect(section?.textContent).toContain('SR');
  expect(section?.textContent).toContain('12');
});

test('facetRailHtml renders the direction facet as R / W / RW / dash, keeping none as the value', () => {
  const container = parse(facetRailHtml(makeFacets(), makeTree(), makeState()));

  const labelFor = (/** @type {string} */ value) =>
    container.querySelector(`[data-facet="dir"][data-value="${value}"] .k`)?.textContent?.trim();

  expect(labelFor('R')).toBe('R');
  expect(labelFor('W')).toBe('W');
  expect(labelFor('RW')).toBe('RW');
  expect(labelFor('none')).toBe('—');
});

test('facetRailHtml marks the active multi-select values with .on', () => {
  const state = makeState({
    section: new Set(['SR']),
    signal: new Set(['current']),
    dir: new Set(['none']),
  });
  const container = parse(facetRailHtml(makeFacets(), makeTree(), state));

  const on = (/** @type {string} */ sel) =>
    container.querySelector(sel)?.classList.contains('on');

  expect(on('[data-facet="section"][data-value="SR"]')).toBe(true);
  expect(on('[data-facet="section"][data-value="BR"]')).toBe(false);
  expect(on('[data-facet="signal"][data-value="current"]')).toBe(true);
  expect(on('[data-facet="dir"][data-value="none"]')).toBe(true);
  expect(on('[data-facet="dir"][data-value="R"]')).toBe(false);
});

test('facetRailHtml marks the single-select class facet from state.cls', () => {
  const state = makeState({ cls: 'http://x/Quadrupole' });
  const container = parse(facetRailHtml(makeFacets(), makeTree(), state));

  const quad = container.querySelector('[data-facet="cls"][data-value="http://x/Quadrupole"]');
  const dipole = container.querySelector('[data-facet="cls"][data-value="http://x/Dipole"]');
  expect(quad?.classList.contains('on')).toBe(true);
  expect(dipole?.classList.contains('on')).toBe(false);
});

test('facetRailHtml renders the ontology tree indented, merged with the class counts', () => {
  const container = parse(facetRailHtml(makeFacets(), makeTree(), makeState()));

  const clsItems = [...container.querySelectorAll('[data-facet="cls"]')];
  // Forest order is pre-order: parent first, then its children.
  expect(clsItems.map(el => el.getAttribute('data-value'))).toEqual([
    'http://x/Magnet', 'http://x/Quadrupole', 'http://x/Dipole',
  ]);
  expect(clsItems[0].getAttribute('data-depth')).toBe('0');
  expect(clsItems[1].getAttribute('data-depth')).toBe('1');

  // The facet count for Quadrupole is merged in by uri.
  expect(clsItems[1].querySelector('.n')?.textContent).toContain('4');
});

test('facetRailHtml leaves the tree indent to the stylesheet', () => {
  const html = facetRailHtml(makeFacets(), makeTree(), makeState());
  // The stylesheet indents from data-depth. An inline padding here would win
  // over it and put the geometry in two places.
  expect(html).not.toContain('padding-left');

  const container = parse(html);
  const clsItems = [...container.querySelectorAll('[data-facet="cls"]')];
  for (const el of clsItems) {
    expect(el.getAttribute('style')).toBeNull();
  }
  // A leaf keeps the twisty gutter but shows nothing in it.
  const magnet = container.querySelector('[data-facet="cls"][data-value="http://x/Magnet"]');
  const quad = container.querySelector('[data-facet="cls"][data-value="http://x/Quadrupole"]');
  expect(magnet?.querySelector('.tw')?.classList.contains('leaf')).toBe(false);
  expect(magnet?.querySelector('.tw')?.textContent?.trim()).toBe('▾');
  expect(quad?.querySelector('.tw')?.classList.contains('leaf')).toBe(true);
  expect(quad?.querySelector('.tw')?.textContent?.trim()).toBe('');

  // Flat facets carry no depth and no twisty.
  const section = container.querySelector('[data-facet="section"][data-value="SR"]');
  expect(section?.hasAttribute('data-depth')).toBe(false);
  expect(section?.querySelector('.tw')).toBeNull();
});

test('facetRailHtml dims a class with a zero count and flags an abstract class', () => {
  const container = parse(facetRailHtml(makeFacets(), makeTree(), makeState()));

  const magnet = container.querySelector('[data-facet="cls"][data-value="http://x/Magnet"]');
  const quad = container.querySelector('[data-facet="cls"][data-value="http://x/Quadrupole"]');
  const dipole = container.querySelector('[data-facet="cls"][data-value="http://x/Dipole"]');

  // Magnet and Dipole carry no count under the active filters.
  expect(magnet?.className).toContain('facet-item');
  expect(magnet?.classList.contains('zero')).toBe(true);
  expect(magnet?.classList.contains('abstract')).toBe(true);
  expect(dipole?.classList.contains('zero')).toBe(true);
  expect(dipole?.classList.contains('abstract')).toBe(false);
  // Quadrupole has 4 and is not dimmed.
  expect(quad?.classList.contains('zero')).toBe(false);
});

test('facetRailHtml never renders a class the server pruned out of the tree', () => {
  const facets = makeFacets();
  facets.class.push({ value: 'http://x/PrunedSignalClass', count: 99 });
  const container = parse(facetRailHtml(facets, makeTree(), makeState()));

  expect(container.querySelector('[data-value="http://x/PrunedSignalClass"]')).toBeNull();
});

test('facetRailHtml escapes every store-sourced facet value and tree name', () => {
  const facets = {
    section: [{ value: PAYLOAD, count: 1 }],
    system: [{ value: PAYLOAD, count: 1 }],
    class: [{ value: PAYLOAD, count: 1 }],
    signal: [{ value: PAYLOAD, count: 1 }],
    dir: [{ value: PAYLOAD, count: 1 }],
  };
  const tree = [{ uri: PAYLOAD, name: PAYLOAD, abstract: false, children: [] }];
  const html = facetRailHtml(facets, tree, makeState({ cls: PAYLOAD }));

  assertInert(html, 'facetRailHtml');
  expect(parse(html).textContent).toContain(PAYLOAD);
});

// ---------------------------------------------------------------------------
// chipsHtml
// ---------------------------------------------------------------------------

test('chipsHtml renders one removal button per active filter', () => {
  const container = parse(chipsHtml([
    { key: 'q', label: '"qfa current"' },
    { key: 'section:SR', label: 'section SR' },
  ]));

  const chips = [...container.querySelectorAll('.active-filter')];
  expect(chips.length).toBe(2);
  expect(chips.map(el => el.getAttribute('data-chip'))).toEqual(['q', 'section:SR']);
  for (const chip of chips) {
    expect(chip.tagName).toBe('BUTTON');
    expect(chip.getAttribute('type')).toBe('button');
  }
  expect(chips[1].textContent).toContain('section SR');
});

test('chipsHtml renders nothing for no active filters', () => {
  expect(chipsHtml([])).toBe('');
  expect(chipsHtml(undefined)).toBe('');
});

test('chipsHtml escapes the chip key and label', () => {
  const html = chipsHtml([{ key: PAYLOAD, label: PAYLOAD }]);
  assertInert(html, 'chipsHtml');
  expect(parse(html).textContent).toContain(PAYLOAD);
});

// ---------------------------------------------------------------------------
// resultsHtml
// ---------------------------------------------------------------------------

test('resultsHtml renders the row columns and the click contract', () => {
  const container = parse(resultsHtml([makeRow()], makeState()));

  const row = container.querySelector('tbody tr');
  expect(row).not.toBeNull();
  expect(row?.querySelector('td.dev button')?.getAttribute('data-uri')).toBe('http://x/dev/QF1');
  expect(row?.querySelector('td.dev button')?.getAttribute('type')).toBe('button');
  expect(row?.querySelector('td.dev button')?.classList.contains('dev')).toBe(true);
  expect(row?.querySelector('td.sec')?.textContent).toContain('SR');
  expect(row?.querySelector('td.pv')?.textContent).toContain('SR:QF1:current');
  expect(row?.querySelector('.dir')?.classList.contains('dir-R')).toBe(true);
  expect(row?.querySelector('td.sig')?.textContent).toContain('current');
  expect(row?.querySelector('td.desc')?.textContent).toContain('Quadrupole current');

  const copy = row?.querySelector('button.copy-btn');
  expect(copy?.getAttribute('data-copy')).toBe('SR:QF1:current');
  expect(copy?.getAttribute('type')).toBe('button');
});

test('resultsHtml renders a directionless row as an em dash', () => {
  const container = parse(resultsHtml([makeRow({ edges: [] })], makeState()));

  const pill = container.querySelector('.dir');
  expect(pill?.classList.contains('dir-none')).toBe(true);
  expect(pill?.textContent?.trim()).toBe('—');
});

test('resultsHtml joins multiple signal names', () => {
  const row = makeRow({
    signals: [
      { uri: 'http://x/sig/a', name: 'current' },
      { uri: 'http://x/sig/b', name: 'setpoint' },
    ],
  });
  const container = parse(resultsHtml([row], makeState()));
  expect(container.querySelector('td.sig')?.textContent).toContain('current, setpoint');
});

test('resultsHtml reflects the selection in the row checkboxes', () => {
  const rows = [makeRow(), makeRow({ fullPv: 'SR:QF2:current', device: 'QF2' })];
  const state = makeState({ selected: new Set(['SR:QF2:current']) });
  const container = parse(resultsHtml(rows, state));

  const boxes = [...container.querySelectorAll('tbody input[type="checkbox"]')];
  expect(boxes.length).toBe(2);
  expect(boxes.map(el => el.getAttribute('data-pv')))
    .toEqual(['SR:QF1:current', 'SR:QF2:current']);
  expect(/** @type {HTMLInputElement} */ (boxes[0]).checked).toBe(false);
  expect(/** @type {HTMLInputElement} */ (boxes[1]).checked).toBe(true);
});

test('resultsHtml checks the header box only when every row on the page is selected', () => {
  const rows = [makeRow(), makeRow({ fullPv: 'SR:QF2:current', device: 'QF2' })];

  const head = (/** @type {Set<string>} */ selected) => {
    const container = parse(resultsHtml(rows, makeState({ selected })));
    const box = container.querySelector('thead input[type="checkbox"][data-select-page]');
    expect(box, 'header checkbox carries data-select-page').not.toBeNull();
    return /** @type {HTMLInputElement} */ (box).checked;
  };

  expect(head(new Set())).toBe(false);
  expect(head(new Set(['SR:QF1:current']))).toBe(false);
  expect(head(new Set(['SR:QF1:current', 'SR:QF2:current']))).toBe(true);
});

test('resultsHtml leaves the header box unchecked on an empty page', () => {
  const container = parse(resultsHtml([], makeState()));
  const box = /** @type {HTMLInputElement | null} */ (
    container.querySelector('thead input[data-select-page]'));
  expect(box?.checked).toBe(false);
  expect(container.querySelector('tbody')?.textContent).toContain('No channels match');
});

test('resultsHtml escapes every store-sourced row field', () => {
  const row = makeRow({
    fullPv: PAYLOAD,
    description: PAYLOAD,
    device: PAYLOAD,
    device_uri: PAYLOAD,
    section: PAYLOAD,
    system: PAYLOAD,
    signals: [{ uri: PAYLOAD, name: PAYLOAD }],
  });
  const html = resultsHtml([row], makeState({ selected: new Set([PAYLOAD]) }));

  assertInert(html, 'resultsHtml');
  expect(parse(html).textContent).toContain(PAYLOAD);
});

// ---------------------------------------------------------------------------
// footerHtml
// ---------------------------------------------------------------------------

test('footerHtml disables every selection action at zero selection', () => {
  const container = parse(footerHtml(120, 30, 1, 3, 0, true));

  for (const action of ['copy', 'send', 'clear']) {
    const btn = /** @type {HTMLButtonElement | null} */ (
      container.querySelector(`[data-action="${action}"]`));
    expect(btn, `${action} button present`).not.toBeNull();
    expect(btn?.disabled, `${action} disabled at zero selection`).toBe(true);
    expect(btn?.getAttribute('type')).toBe('button');
  }
});

test('footerHtml enables the selection actions once something is selected', () => {
  const container = parse(footerHtml(120, 30, 1, 3, 2, true));

  for (const action of ['copy', 'send', 'clear']) {
    const btn = /** @type {HTMLButtonElement} */ (
      container.querySelector(`[data-action="${action}"]`));
    expect(btn.disabled, `${action} enabled with a selection`).toBe(false);
  }
  expect(container.textContent).toContain('2 selected');
});

test('footerHtml renders Send only when embedded', () => {
  const standalone = parse(footerHtml(120, 30, 1, 3, 2, false));
  expect(standalone.querySelector('[data-action="send"]')).toBeNull();
  expect(standalone.querySelector('[data-action="copy"]')).not.toBeNull();
  expect(standalone.querySelector('[data-action="clear"]')).not.toBeNull();

  const embedded = parse(footerHtml(120, 30, 1, 3, 2, true));
  expect(embedded.querySelector('[data-action="send"]')).not.toBeNull();
});

test('footerHtml disables the pager at both bounds', () => {
  const prevOf = (/** @type {HTMLElement} */ c) => /** @type {HTMLButtonElement} */ (
    c.querySelector('[data-page="prev"]'));
  const nextOf = (/** @type {HTMLElement} */ c) => /** @type {HTMLButtonElement} */ (
    c.querySelector('[data-page="next"]'));

  const first = parse(footerHtml(120, 30, 1, 3, 0, true));
  expect(prevOf(first).disabled).toBe(true);
  expect(nextOf(first).disabled).toBe(false);

  const middle = parse(footerHtml(120, 30, 2, 3, 0, true));
  expect(prevOf(middle).disabled).toBe(false);
  expect(nextOf(middle).disabled).toBe(false);

  const last = parse(footerHtml(120, 30, 3, 3, 0, true));
  expect(prevOf(last).disabled).toBe(false);
  expect(nextOf(last).disabled).toBe(true);

  const empty = parse(footerHtml(0, 0, 1, 0, 0, true));
  expect(prevOf(empty).disabled).toBe(true);
  expect(nextOf(empty).disabled).toBe(true);
});

test('footerHtml states the channel and device counts', () => {
  const container = parse(footerHtml(120, 30, 1, 3, 0, true));
  const text = container.textContent ?? '';
  expect(text).toContain('120');
  expect(text).toContain('30');
  expect(text).toContain('channels');
  expect(text).toContain('devices');
  expect(text).toContain('1 / 3');
});

// ---------------------------------------------------------------------------
// deviceCardHtml / deviceCardErrorHtml
// ---------------------------------------------------------------------------

/**
 * A device-card payload.
 * @param {Partial<Record<string, any>>} [over]
 */
function makeDevice(over = {}) {
  return {
    device: 'QF1',
    device_uri: 'http://x/dev/QF1',
    section: 'SR',
    system: 'MAG',
    class: 'Quadrupole',
    rawType: 'QUAD',
    sPositionM: 12.5,
    ordinalInSection: 4,
    systemDescription: 'Magnet system',
    familyDescription: 'Storage-ring quadrupole family',
    signals: [
      {
        uri: 'http://x/sem/current',
        name: 'current',
        bindings: [
          {
            fullPv: 'SR:QF1:current',
            edges: ['READSSIGNAL'],
            subfieldDescription: 'Measured current',
            fieldDescription: 'Current',
          },
          {
            fullPv: 'SR:QF1:currentSet',
            edges: ['WRITESSIGNAL'],
            fieldDescription: 'Current setpoint',
          },
        ],
      },
      {
        uri: 'http://x/sem/status',
        name: 'status',
        bindings: [
          {
            fullPv: 'SR:QF1:status',
            edges: ['READSSIGNAL', 'WRITESSIGNAL'],
          },
        ],
      },
    ],
    ...over,
  };
}

test('deviceCardHtml renders the device fields and a close control', () => {
  const container = parse(deviceCardHtml(makeDevice()));

  expect(container.querySelector('.device-card')).not.toBeNull();
  expect(container.querySelector('.device-card-head .name')?.textContent).toContain('QF1');

  const close = /** @type {HTMLButtonElement | null} */ (
    container.querySelector('[data-action="close-card"]'));
  expect(close).not.toBeNull();
  expect(close?.tagName).toBe('BUTTON');
  expect(close?.getAttribute('type')).toBe('button');

  const text = container.textContent ?? '';
  expect(text).toContain('SR');
  expect(text).toContain('MAG');
  expect(text).toContain('Quadrupole');
  expect(text).toContain('QUAD');
  expect(text).toContain('12.5');
  expect(text).toContain('4');
  expect(text).toContain('Magnet system');
  expect(text).toContain('Storage-ring quadrupole family');
});

test('deviceCardHtml draws each signal group as the endpoint sends it', () => {
  const container = parse(deviceCardHtml(makeDevice()));

  const rows = [...container.querySelectorAll('.sig-table tbody tr')];
  expect(rows.length).toBe(3);

  // The signal name is written once per group, on the group's first row.
  const sigCells = rows.map(r => r.querySelector('td.sig')?.textContent?.trim());
  expect(sigCells).toEqual(['current', '', 'status']);

  // Each binding keeps its own direction and address.
  expect(rows[0].querySelector('.dir')?.classList.contains('dir-R')).toBe(true);
  expect(rows[1].querySelector('.dir')?.classList.contains('dir-W')).toBe(true);
  expect(rows[2].querySelector('.dir')?.classList.contains('dir-RW')).toBe(true);
  expect(rows[0].querySelector('td.pv')?.textContent).toContain('SR:QF1:current');

  // Subfield description wins over the field description; neither leaves a gap.
  expect(rows[0].querySelector('td.sub')?.textContent).toContain('Measured current');
  expect(rows[1].querySelector('td.sub')?.textContent).toContain('Current setpoint');
});

test('deviceCardHtml copies an address per binding', () => {
  const container = parse(deviceCardHtml(makeDevice()));
  const copies = [...container.querySelectorAll('.sig-table button.copy-btn')];
  expect(copies.length).toBe(3);
  expect(copies[0].getAttribute('data-copy')).toBe('SR:QF1:current');
  expect(copies[0].getAttribute('type')).toBe('button');
  // In the card the control is always visible, which is a class and not an
  // inline opacity, so the stylesheet keeps owning the hover behaviour.
  for (const btn of copies) {
    expect(btn.classList.contains('copy-btn-static')).toBe(true);
    expect(btn.getAttribute('style')).toBeNull();
  }
});

test('resultsHtml keeps the copy control hover-revealed', () => {
  const container = parse(resultsHtml([makeRow()], makeState()));
  const btn = container.querySelector('button.copy-btn');
  expect(btn?.classList.contains('copy-btn-static')).toBe(false);
});

test('deviceCardHtml omits absent optional fields rather than printing undefined', () => {
  const container = parse(deviceCardHtml({
    device: 'QF1', section: 'SR', signals: [],
  }));
  const text = container.textContent ?? '';
  expect(text).not.toContain('undefined');
  expect(text).not.toContain('null');
  expect(text).not.toContain('NaN');
});

test('deviceCardHtml escapes every store-sourced device field', () => {
  const html = deviceCardHtml(makeDevice({
    device: PAYLOAD,
    device_uri: PAYLOAD,
    section: PAYLOAD,
    system: PAYLOAD,
    class: PAYLOAD,
    rawType: PAYLOAD,
    systemDescription: PAYLOAD,
    familyDescription: PAYLOAD,
    signals: [{
      uri: PAYLOAD,
      name: PAYLOAD,
      bindings: [{
        fullPv: PAYLOAD,
        edges: ['READSSIGNAL'],
        subfieldDescription: PAYLOAD,
        fieldDescription: PAYLOAD,
      }],
    }],
  }));

  assertInert(html, 'deviceCardHtml');
  expect(parse(html).textContent).toContain(PAYLOAD);
});

test('deviceCardErrorHtml shows the server detail and keeps the close control', () => {
  const container = parse(deviceCardErrorHtml('Device not found in the store.'));

  expect(container.querySelector('.device-card')).not.toBeNull();
  expect(container.textContent).toContain('Device not found in the store.');

  const close = /** @type {HTMLButtonElement | null} */ (
    container.querySelector('[data-action="close-card"]'));
  expect(close).not.toBeNull();
  expect(close?.getAttribute('type')).toBe('button');
});

test('deviceCardErrorHtml escapes the server detail', () => {
  const html = deviceCardErrorHtml(PAYLOAD);
  assertInert(html, 'deviceCardErrorHtml');
  expect(parse(html).textContent).toContain(PAYLOAD);
});
