// The orbit-bump panel: one selection object (what the operator is building) and
// a state loop that polls the backend and renders from state plus selection.
// Every address is relative: the page is served under the terminal's /panel/<id>/.

const STATE_URL = 'api/state';
const RECOMPUTE_URL = 'api/recompute';
const DISCARD_URL = 'api/discard';
const COMPUTE_URL = 'api/compute';
const APPLY_URL = 'api/apply';
const REFRESH_MS = 2000;

/**
 * @typedef {'x' | 'y'} Plane
 * @typedef {'auto' | 'mml:[-1 1]' | 'mml:[-2 -1 1 2]' | 'mml:[-3 -1 1 3]' | 'manual'} Rule
 * @typedef {'idle' | 'building' | 'ready' | 'failed'} OrmStatus
 */

/**
 * @typedef {object} Member
 * @property {string} element
 * @property {Plane} plane
 * @property {number} index
 * @property {number} s_m
 * @property {string | null} [unit]
 * @property {string} [position_addr]
 * @property {string | null} [golden_addr]
 * @property {string} [setpoint_addr]
 * @property {string} [readback_addr]
 * @property {number | null} [tolerance]
 */

/**
 * @typedef {object} OrmEntry
 * @property {OrmStatus} status
 * @property {number | null} age_s
 * @property {number} version
 * @property {string | null} error
 * @property {string[]} unsolved
 */

/**
 * @typedef {object} Model
 * @property {string} name
 * @property {number} length_m
 * @property {OrmEntry} orm
 * @property {{bpms: Member[], correctors: Member[]}} members
 */

/**
 * The target's connector judgement of one new setpoint.
 * @typedef {object} Verdict
 * @property {boolean} allowed
 * @property {string | null} reason
 * @property {boolean} declared
 * @property {boolean} enforced
 * @property {'optional' | 'exclusive' | null} mode
 * @property {{min: number | null, max: number | null, max_step: number | null} | null} limits
 */

/**
 * @typedef {'confirmed' | 'mismatch' | 'refused' | 'failed' | 'unconfirmed' | 'not_attempted'} Outcome
 */

/**
 * One corrector row of a plan. Only `element` is guaranteed; the rest is what
 * the backend's Compute fills in.
 * @typedef {object} PlanCorrector
 * @property {string} element
 * @property {string} [setpoint_addr]
 * @property {string} [readback_addr]
 * @property {number} [now] the setpoint read at Compute
 * @property {number | null} [readback_at_compute]
 * @property {number} [new]
 * @property {number} [delta]
 * @property {string | null} [unit]
 * @property {Verdict} [verdict]
 * @property {number | null} [tolerance]
 * @property {Outcome} [outcome]
 * @property {number | null} [readback]
 * @property {boolean} [settled]
 */

/**
 * The plan as the backend returns it.
 * @typedef {object} Plan
 * @property {string} id
 * @property {string} model
 * @property {Plane} plane
 * @property {string} status
 * @property {'writing' | 'settling'} [phase]
 * @property {string} [reason] why a plan is not applicable
 * @property {Record<string, number>} targets
 * @property {string} [setpoint_digest]
 * @property {PlanCorrector[]} correctors
 * @property {Record<string, number | null>} [predicted] position per BPM element after the bump
 * @property {{element: string, predicted_residual: number}[]} [leakage]
 * @property {number | null} [predicted_closure_max]
 * @property {number} [sigma_kept]
 * @property {number} [sigma_total]
 * @property {string[]} [skipped_unsolved] correctors the rule skipped because the ORM could not solve them
 * @property {string[]} [dropped_bpms] non-bump BPMs left out as unreadable
 * @property {Record<string, number>} [residual_after]
 */

/**
 * @typedef {object} BumpState
 * @property {Model[]} models
 * @property {{allowed: boolean, reason?: string | null, phrase?: string | null}} posture
 * @property {{position: Record<string, number | null>, golden: Record<string, number | null>}} orbit
 * @property {Plan | null} plan
 */

/**
 * What the operator is building. Targets are offsets from the golden orbit, in
 * the facility's unit, keyed by BPM element in pick order.
 * @typedef {object} Selection
 * @property {Plane} plane
 * @property {string | null} model
 * @property {Record<string, number>} targets
 * @property {Rule} rule
 * @property {string[]} correctors manual correctors, by element
 */

/**
 * Everything the panel keeps between polls, one per root.
 * @typedef {object} Panel
 * @property {Selection} selection
 * @property {BumpState | null} state
 * @property {string | null} lastSeenPlanId the newest plan id this panel has taken in
 * @property {number} generation the counter of the newest state request sent
 * @property {number} shownGeneration the counter of the newest state response shown
 * @property {number} computesInFlight
 * @property {boolean} applying whether this panel's own apply is in flight
 * @property {boolean} readOnly set by a 403 on Compute: Compute and Apply are hidden
 * @property {{s0: number, s1: number} | null} view the strip's s-window in metres; null is the whole ring
 * @property {typeof fetch} fetcher what the panel's own controls send through
 */

/** @type {WeakMap<Element, Panel>} */
const panels = new WeakMap();

/** @returns {Selection} */
function emptySelection() {
  return { plane: 'x', model: null, targets: {}, rule: 'auto', correctors: [] };
}

/**
 * The panel kept for one root, made on first use.
 * @param {Element} root
 * @returns {Panel}
 */
export function panelOf(root) {
  let panel = panels.get(root);
  if (!panel) {
    panel = {
      selection: emptySelection(),
      state: null,
      lastSeenPlanId: null,
      generation: 0,
      shownGeneration: 0,
      computesInFlight: 0,
      applying: false,
      readOnly: false,
      view: null,
      fetcher: (input, init) => fetch(input, init),
    };
    panels.set(root, panel);
  }
  return panel;
}

/**
 * @param {Element} root
 * @param {string} name
 * @returns {HTMLElement}
 */
function field(root, name) {
  const element = root.querySelector(`[data-field="${name}"]`);
  if (!(element instanceof HTMLElement)) throw new Error(`panel has no ${name} field`);
  return element;
}

/**
 * Replace the selection. This is the one setter for selection fields; it only
 * changes what the panel shows and never discards a plan on the backend.
 * @param {Element} root
 * @param {Partial<Selection>} changes
 */
export function setSelection(root, changes) {
  const panel = panelOf(root);
  panel.selection = { ...panel.selection, ...changes };
  render(root);
}

/**
 * Empty the local selection of BPMs, targets and correctors, keeping plane and
 * model. Discarding the backend's plan is the caller's business.
 * @param {Element} root
 */
export function clearSelection(root) {
  setSelection(root, { targets: {}, rule: 'auto', correctors: [] });
}

/**
 * @param {Element} root
 * @returns {boolean}
 */
function confirmOpen(root) {
  return !field(root, 'confirm').hidden;
}

/**
 * @param {Element} root
 * @param {string} message
 */
function showNotice(root, message) {
  const notice = field(root, 'notice');
  notice.textContent = message;
  notice.hidden = false;
}

/**
 * Take in a plan this panel has not seen: one the agent computed, or one left
 * from before the page loaded. Never runs while the panel's own Compute is in
 * flight, and a null plan never re-hydrates.
 * @param {Element} root
 * @param {Plan | null} plan
 */
function hydrate(root, plan) {
  const panel = panelOf(root);
  if (!plan || plan.id == null) return;
  if (plan.id === panel.lastSeenPlanId) return;
  if (panel.computesInFlight > 0) return;
  panel.lastSeenPlanId = plan.id;
  if (confirmOpen(root)) {
    const confirm = field(root, 'confirm');
    confirm.hidden = true;
    confirm.replaceChildren();
    showNotice(root, 'A new plan arrived from the agent; the confirm card was closed.');
  }
  setSelection(root, {
    plane: plan.plane,
    model: plan.model,
    targets: { ...plan.targets },
    rule: 'manual',
    correctors: plan.correctors.map((row) => row.element),
  });
}

/**
 * The selected model's entry in state, or null when none is offered.
 * @param {Panel} panel
 * @returns {Model | null}
 */
export function selectedModel(panel) {
  const models = panel.state?.models ?? [];
  return models.find((model) => model.name === panel.selection.model) ?? null;
}

/**
 * @param {number | null | undefined} ageS
 * @returns {string}
 */
function formatAge(ageS) {
  if (ageS == null || !Number.isFinite(ageS)) return '';
  if (ageS < 60) return `${Math.round(ageS)} s ago`;
  if (ageS < 3600) return `${Math.round(ageS / 60)} min ago`;
  return `${Math.round(ageS / 3600)} h ago`;
}

/**
 * @param {Model} model
 * @returns {string}
 */
function ormText(model) {
  const { orm } = model;
  const parts = [`ORM · ${model.name}`, orm.status];
  if (orm.status === 'ready') parts.push(formatAge(orm.age_s));
  if (orm.status === 'failed' && orm.error) parts.push(orm.error);
  return `${parts.filter(Boolean).join(' · ')} · `;
}

/** The ORM chip's tone per ORM status; an idle ORM keeps the plain chip. */
const ORM_TONES = /** @type {Record<string, string>} */ ({
  building: 'osprey-chip--connecting',
  ready: 'osprey-chip--ok',
  failed: 'osprey-chip--err',
});

/**
 * @param {Element} root
 * @param {Panel} panel
 */
function renderHeader(root, panel) {
  const models = panel.state?.models ?? [];
  const { selection } = panel;

  for (const button of field(root, 'plane').querySelectorAll('[data-plane]')) {
    const on = button.getAttribute('data-plane') === selection.plane;
    button.setAttribute('aria-pressed', String(on));
  }

  const select = /** @type {HTMLSelectElement} */ (field(root, 'model'));
  select.hidden = models.length < 2;
  const names = models.map((model) => model.name);
  if ([...select.options].map((option) => option.value).join('\n') !== names.join('\n')) {
    select.replaceChildren(
      ...names.map((name) => {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        return option;
      })
    );
  }
  if (selection.model) select.value = selection.model;

  field(root, 'no-model').hidden = !(panel.state && models.length === 0);

  const chip = field(root, 'orm');
  const model = selectedModel(panel);
  chip.hidden = !model;
  for (const tone of Object.values(ORM_TONES)) chip.classList.remove(tone);
  if (model) {
    const tone = ORM_TONES[model.orm.status];
    if (tone) chip.classList.add(tone);
    field(root, 'orm-text').textContent = ormText(model);
  }

  const posture = field(root, 'posture');
  const phrase = panel.state?.posture?.phrase ?? '';
  posture.textContent = phrase;
  posture.hidden = !phrase;
}

const SVG_NS = 'http://www.w3.org/2000/svg';

// Strip geometry in SVG user units, after the mockup's draw(): a plot band over
// a corrector row, a left margin for the vertical labels. A plan's Δ bars stand
// on the corrector row, up for a positive step and down for a negative one.
const STRIP_W = 1000;
const STRIP_H = 222;
const STRIP_LEFT = 34;
const STRIP_RIGHT = 12;
const STRIP_TOP = 10;
const PLOT_BOTTOM = 150;
const DELTA_H = 20;
const COR_Y = PLOT_BOTTOM + DELTA_H + 4;
const COR_H = 8;
const STRIP_MID = (STRIP_TOP + PLOT_BOTTOM) / 2;
const HEADROOM = 1.15;
const S_TICKS = 4;

/**
 * @param {string} tag
 * @param {Record<string, string | number>} [attrs]
 * @param {string} [text]
 * @returns {SVGElement}
 */
function svgNode(tag, attrs = {}, text) {
  const node = /** @type {SVGElement} */ (document.createElementNS(SVG_NS, tag));
  for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, String(value));
  if (text != null) node.textContent = text;
  return node;
}

/**
 * @param {SVGElement} node
 * @param {string} text
 * @returns {SVGElement}
 */
function titled(node, text) {
  node.append(svgNode('title', {}, text));
  return node;
}

/**
 * A number for an axis label: two significant digits, no trailing noise.
 * @param {number} value
 * @returns {string}
 */
function axisNumber(value) {
  return String(Number(value.toPrecision(2)));
}

// Zoom, after the mockup's gestures but in metres: the narrowest window as a
// fraction of the ring, the wheel's zoom step, the shortest drag (user units)
// that counts as a range.
const MIN_SPAN_FRACTION = 0.01;
const WHEEL_STEP = 1.25;
const MIN_DRAG = 4;

/**
 * A zoom drag in flight on one strip: a range being selected, or a pan.
 * @typedef {object} Drag
 * @property {'select' | 'pan'} mode
 * @property {number} x0 where the drag started, in user units
 * @property {number} x1 where the pointer is now, in user units
 * @property {{s0: number, s1: number}} view0 the window when the drag started
 */

/** @type {WeakMap<Element, Drag>} */
const drags = new WeakMap();

/**
 * A view clamped inside a ring of the given length: at least the minimum span,
 * shifted (span kept) to lie in [0, length]; null when it covers the ring.
 * @param {{s0: number, s1: number} | null} view
 * @param {number} length
 * @returns {{s0: number, s1: number} | null}
 */
function clampView(view, length) {
  if (!view || !(length > 0)) return null;
  if (!Number.isFinite(view.s0) || !Number.isFinite(view.s1)) return null;
  const span = Math.max(length * MIN_SPAN_FRACTION, view.s1 - view.s0);
  if (span >= length) return null;
  const s0 = Math.min(Math.max(view.s0, 0), length - span);
  return { s0, s1: s0 + span };
}

/**
 * The window the strip shows for the model: the panel's view, or the whole ring.
 * @param {Panel} panel
 * @param {Model} model
 * @returns {{s0: number, s1: number}}
 */
function shownView(panel, model) {
  return panel.view ?? { s0: 0, s1: model.length_m };
}

/**
 * Set the strip's window (clamped to the selected model's ring) and redraw it.
 * @param {Element} root
 * @param {{s0: number, s1: number} | null} view
 */
function setView(root, view) {
  const panel = panelOf(root);
  const model = selectedModel(panel);
  if (!model) return;
  panel.view = clampView(view, model.length_m);
  renderStrip(root, panel);
}

/**
 * Back to the whole ring, ending any drag.
 * @param {Element} root
 */
function resetView(root) {
  drags.delete(root);
  setView(root, null);
}

/**
 * The pointer's x in the strip's user units, or null when the strip has no
 * laid-out width (a hidden tile) and no position can be read.
 * @param {SVGElement} svg
 * @param {MouseEvent} event
 * @returns {number | null}
 */
function svgX(svg, event) {
  const box = svg.getBoundingClientRect();
  if (!(box.width > 0)) return null;
  const x = ((event.clientX - box.left) * STRIP_W) / box.width;
  return Number.isFinite(x) ? x : null;
}

/**
 * The s-position under user-unit x for a window.
 * @param {number} x
 * @param {{s0: number, s1: number}} view
 * @returns {number}
 */
function sAt(x, view) {
  return view.s0 + ((x - STRIP_LEFT) * (view.s1 - view.s0)) / (STRIP_W - STRIP_LEFT - STRIP_RIGHT);
}

/**
 * Wire the zoom gestures on the strip: the wheel zooms around the cursor; a
 * drag from the empty plot selects a range (or pans, when zoomed and no Shift);
 * a double-click returns to the whole ring.
 * @param {Element} root
 * @param {SVGElement} svg
 */
function bindZoom(root, svg) {
  const current = () => {
    const panel = panelOf(root);
    const model = selectedModel(panel);
    return model ? shownView(panel, model) : null;
  };
  svg.addEventListener(
    'wheel',
    (event) => {
      const view = current();
      const x = svgX(svg, event);
      if (!view || x == null || event.deltaY === 0) return;
      event.preventDefault();
      const factor = event.deltaY > 0 ? WHEEL_STEP : 1 / WHEEL_STEP;
      const anchor = sAt(x, view);
      const span = (view.s1 - view.s0) * factor;
      const s0 = anchor - (anchor - view.s0) * factor;
      setView(root, { s0, s1: s0 + span });
    },
    { passive: false }
  );
  svg.addEventListener('pointerdown', (event) => {
    if (!(event.target instanceof Element) || !event.target.classList.contains('ob-bg')) return;
    const view = current();
    const x = svgX(svg, event);
    if (!view || x == null) return;
    try {
      svg.setPointerCapture(event.pointerId);
    } catch {
      // No active pointer to capture (a synthetic event): the drag works without it.
    }
    const zoomed = panelOf(root).view != null;
    drags.set(root, {
      mode: zoomed && !event.shiftKey ? 'pan' : 'select',
      x0: x,
      x1: x,
      view0: view,
    });
  });
  svg.addEventListener('pointermove', (event) => {
    const drag = drags.get(root);
    const x = svgX(svg, event);
    if (!drag || x == null) return;
    drag.x1 = x;
    if (drag.mode === 'select') {
      renderStrip(root, panelOf(root));
      return;
    }
    const shift = sAt(x, drag.view0) - sAt(drag.x0, drag.view0);
    setView(root, { s0: drag.view0.s0 - shift, s1: drag.view0.s1 - shift });
  });
  const finish = (/** @type {PointerEvent} */ event) => {
    const drag = drags.get(root);
    if (!drag) return;
    drags.delete(root);
    const x = event.type === 'pointerup' ? svgX(svg, event) : null;
    if (x != null) drag.x1 = x;
    if (
      drag.mode === 'select' &&
      event.type === 'pointerup' &&
      Math.abs(drag.x1 - drag.x0) >= MIN_DRAG
    ) {
      const a = sAt(Math.min(drag.x0, drag.x1), drag.view0);
      const b = sAt(Math.max(drag.x0, drag.x1), drag.view0);
      setView(root, { s0: a, s1: b });
      return;
    }
    renderStrip(root, panelOf(root));
  };
  svg.addEventListener('pointerup', finish);
  svg.addEventListener('pointercancel', finish);
  svg.addEventListener('dblclick', () => resetView(root));
}

/**
 * The header chip naming the strip's window, with its "whole ring" reset link;
 * made on first use, after the ORM chip (or first in the panel without one).
 * @param {Element} root
 * @returns {HTMLElement}
 */
function rangeChip(root) {
  const found = root.querySelector('[data-field="range-chip"]');
  if (found instanceof HTMLElement) return found;
  const chip = document.createElement('span');
  chip.className = 'osprey-chip';
  chip.setAttribute('data-field', 'range-chip');
  const text = document.createElement('span');
  text.setAttribute('data-field', 'range');
  const link = document.createElement('button');
  link.type = 'button';
  link.className = 'ob-link';
  link.setAttribute('data-field', 'whole-ring');
  link.textContent = '· whole ring';
  link.hidden = true;
  link.addEventListener('click', () => resetView(root));
  chip.append(text, link);
  const orm = root.querySelector('[data-field="orm"]');
  if (orm) orm.after(chip);
  else root.prepend(chip);
  return chip;
}

/**
 * A metre position for the range chip: at most one decimal.
 * @param {number} s
 * @returns {string}
 */
function metres(s) {
  return String(Number(s.toFixed(1)));
}

/**
 * Name the window in the header chip: "whole ring", or the first and last BPM
 * in view with the s-range in metres.
 * @param {Element} root
 * @param {{s0: number, s1: number} | null} view null for the whole ring
 * @param {string[]} names the BPMs in view, in s order
 */
function renderRangeChip(root, view, names) {
  const chip = rangeChip(root);
  chip.hidden = false;
  const link = /** @type {HTMLElement} */ (chip.querySelector('[data-field="whole-ring"]'));
  link.hidden = view == null;
  const text = /** @type {HTMLElement} */ (chip.querySelector('[data-field="range"]'));
  if (!view) {
    text.textContent = 'whole ring';
    return;
  }
  const range = `${metres(view.s0)}–${metres(view.s1)} m`;
  const first = names[0];
  const last = names[names.length - 1];
  const bpms = first == null ? '' : first === last ? `${first} · ` : `${first}–${last} · `;
  text.textContent = `${bpms}${range}`;
}

/**
 * The strip's svg inside the body field, made (with its click and zoom
 * handlers) on first use.
 * @param {Element} root
 * @returns {SVGElement}
 */
function stripSvg(root) {
  const body = field(root, 'body');
  const found = body.querySelector('svg[data-field="strip"]');
  if (found instanceof SVGElement) return found;
  const svg = svgNode('svg', {
    'data-field': 'strip',
    class: 'ob-strip',
    viewBox: `0 0 ${STRIP_W} ${STRIP_H}`,
    width: '100%',
  });
  svg.addEventListener('click', (event) => pickOnStrip(root, event));
  bindZoom(root, svg);
  body.prepend(svg);
  return svg;
}

/**
 * A click on the strip: a BPM dot joins the bump with a zero offset; a
 * corrector tick toggles under the manual rule. Hollow dots are never picked.
 * @param {Element} root
 * @param {Event} event
 */
function pickOnStrip(root, event) {
  const hit = event.target instanceof Element ? event.target.closest('[data-element]') : null;
  const element = hit?.getAttribute('data-element');
  if (!hit || !element || hit.classList.contains('hollow')) return;
  const { selection } = panelOf(root);
  if (hit.classList.contains('ob-bpm')) {
    if (element in selection.targets) return;
    setSelection(root, { targets: { ...selection.targets, [element]: 0 } });
  } else if (hit.classList.contains('ob-cor') && selection.rule === 'manual') {
    const correctors = selection.correctors.includes(element)
      ? selection.correctors.filter((name) => name !== element)
      : [...selection.correctors, element];
    setSelection(root, { correctors });
  }
}

/**
 * A difference of two readings, or null when either is missing or non-finite.
 * @param {number | null | undefined} value
 * @param {number | null | undefined} reference
 * @returns {number | null}
 */
function offset(value, reference) {
  if (value == null || reference == null) return null;
  const difference = value - reference;
  return Number.isFinite(difference) ? difference : null;
}

/**
 * A polyline through the points, lifted at every null so a missing reading
 * breaks the line.
 * @param {{x: number, y: number | null}[]} points
 * @returns {string}
 */
function brokenPath(points) {
  let d = '';
  let pen = false;
  for (const { x, y } of points) {
    if (y == null) {
      pen = false;
      continue;
    }
    d += `${pen ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)} `;
    pen = true;
  }
  return d.trim();
}

/**
 * Draw the strip for the selected model and plane from the state and the
 * selection: the orbit as position − golden (golden is the zero line), the
 * targets, the plan's predicted orbit, BPM dots and corrector ticks at their
 * s-positions. Values are in the BPM members' own unit.
 * @param {Element} root
 * @param {Panel} panel
 */
function renderStrip(root, panel) {
  const svg = stripSvg(root);
  const model = selectedModel(panel);
  if (!model) {
    svg.replaceChildren();
    rangeChip(root).hidden = true;
    return;
  }
  const { selection, state } = panel;
  const { plane } = selection;
  const bpms = model.members.bpms.filter((member) => member.plane === plane);
  const correctors = model.members.correctors.filter((member) => member.plane === plane);
  panel.view = clampView(panel.view, model.length_m);
  const zoomed = panel.view != null;
  const { s0, s1 } = shownView(panel, model);
  const span = s1 - s0 > 0 ? s1 - s0 : 1;
  const position = state?.orbit?.position ?? {};
  const golden = state?.orbit?.golden ?? {};
  const plan = state?.plan;
  const planHere = plan && plan.model === model.name && plan.plane === plane ? plan : null;
  const predicted = planHere?.predicted ?? null;
  /** @type {Map<string, number>} */
  const steps = new Map();
  for (const row of planHere?.correctors ?? []) {
    if (row.delta != null && Number.isFinite(row.delta)) steps.set(row.element, row.delta);
  }
  const largestStep = Math.max(0, ...[...steps.values()].map(Math.abs));

  const rows = bpms.map((member) => {
    const gold = member.golden_addr ? golden[member.golden_addr] : null;
    const pos = member.position_addr ? position[member.position_addr] : null;
    return {
      member,
      orbit: offset(pos, gold),
      target: member.element in selection.targets ? selection.targets[member.element] : null,
      predicted: predicted ? offset(predicted[member.element], gold) : null,
    };
  });

  const onView = (/** @type {number} */ s) => s >= s0 && s <= s1;
  let largest = 0;
  for (const row of rows) {
    if (!onView(row.member.s_m)) continue;
    for (const value of [row.orbit, row.target, row.predicted]) {
      if (value != null && Number.isFinite(value)) largest = Math.max(largest, Math.abs(value));
    }
  }
  const yMax = largest > 0 ? largest * HEADROOM : 1;
  const px = (/** @type {number} */ s) =>
    STRIP_LEFT + ((s - s0) * (STRIP_W - STRIP_LEFT - STRIP_RIGHT)) / span;
  const py = (/** @type {number} */ v) => STRIP_MID - (v * (STRIP_MID - STRIP_TOP)) / yMax;
  const right = STRIP_W - STRIP_RIGHT;

  svg.setAttribute('data-s0', String(s0));
  svg.setAttribute('data-s1', String(s1));
  /** @type {SVGElement[]} */
  const nodes = [];

  const clip = svgNode('clipPath', { id: 'ob-strip-plot' });
  clip.append(
    svgNode('rect', {
      x: STRIP_LEFT,
      y: 0,
      width: STRIP_W - STRIP_LEFT - STRIP_RIGHT,
      height: STRIP_H,
    })
  );
  const defs = svgNode('defs');
  defs.append(clip);
  nodes.push(defs);
  // The empty plot, under everything: a drag that starts here zooms or pans,
  // while dots and ticks drawn on top keep their clicks.
  nodes.push(
    svgNode('rect', {
      class: 'ob-bg',
      x: STRIP_LEFT,
      y: 0,
      width: STRIP_W - STRIP_LEFT - STRIP_RIGHT,
      height: COR_Y - 4,
      fill: 'transparent',
      style: `cursor:${zoomed ? 'grab' : 'crosshair'}`,
    })
  );

  const unit = bpms.find((member) => member.unit)?.unit ?? '';
  const tick = largest > 0 ? largest / 2 : yMax / 2;
  for (const value of [tick, -tick]) {
    nodes.push(
      svgNode('line', {
        class: 'ob-grid',
        x1: STRIP_LEFT,
        y1: py(value),
        x2: right,
        y2: py(value),
        stroke: 'var(--border-default)',
      }),
      svgNode('text', { x: 2, y: py(value) + 3 }, `${value > 0 ? '+' : '−'}${axisNumber(tick)}`)
    );
  }
  nodes.push(
    svgNode('line', {
      class: 'ob-golden',
      x1: STRIP_LEFT,
      y1: STRIP_MID,
      x2: right,
      y2: STRIP_MID,
      stroke: 'var(--color-accent-secondary)',
      'stroke-dasharray': '3 3',
    }),
    svgNode('text', { x: 10, y: STRIP_MID + 3 }, '0'),
    svgNode('text', { x: 2, y: 12, 'data-field': 'unit' }, unit)
  );
  for (let k = 0; k <= S_TICKS; k += 1) {
    const s = s0 + (span * k) / S_TICKS;
    const anchor = k === 0 ? 'start' : k === S_TICKS ? 'end' : 'middle';
    nodes.push(
      svgNode(
        'text',
        { class: 'ob-s', x: px(s), y: STRIP_H - 3, 'text-anchor': anchor },
        `${axisNumber(s)} m`
      )
    );
  }

  const plot = svgNode('g', { 'clip-path': 'url(#ob-strip-plot)' });
  const pointsOf = (/** @type {'orbit' | 'predicted'} */ key) =>
    rows.map((row) => ({
      x: px(row.member.s_m),
      y: row[key] == null ? null : py(row[key]),
    }));
  if (predicted) {
    plot.append(
      svgNode('path', {
        class: 'ob-predicted',
        d: brokenPath(pointsOf('predicted')),
        fill: 'none',
        stroke: 'var(--art-violet-light)',
        'stroke-width': 1.5,
        'stroke-dasharray': '4 3',
      })
    );
  }
  plot.append(
    svgNode('path', {
      class: 'ob-orbit',
      d: brokenPath(pointsOf('orbit')),
      fill: 'none',
      stroke: 'var(--color-accent)',
      'stroke-width': 1.5,
    })
  );

  for (const row of rows) {
    const { member } = row;
    if (!onView(member.s_m)) continue;
    const hollow = row.orbit == null;
    const on = row.target != null;
    const classes = ['ob-bpm', on ? 'on' : '', hollow ? 'hollow' : ''].filter(Boolean);
    const colour = on ? 'var(--color-warning)' : 'var(--color-accent)';
    plot.append(
      titled(
        svgNode('circle', {
          class: classes.join(' '),
          'data-element': member.element,
          cx: px(member.s_m),
          cy: py(row.orbit ?? 0),
          r: on ? 4.5 : 3,
          fill: hollow ? 'none' : colour,
          stroke: colour,
          style: `cursor:${hollow ? 'not-allowed' : 'pointer'}`,
        }),
        hollow ? `${member.element}: no reading or no golden` : member.element
      )
    );
  }

  const targets = rows.filter((row) => row.target != null && onView(row.member.s_m));
  if (targets.length > 1) {
    plot.append(
      svgNode('path', {
        class: 'ob-target-line',
        d: brokenPath(targets.map((row) => ({ x: px(row.member.s_m), y: py(row.target ?? 0) }))),
        fill: 'none',
        stroke: 'var(--color-warning)',
        'stroke-width': 1.2,
        opacity: 0.7,
      })
    );
  }
  for (const row of targets) {
    const x = px(row.member.s_m);
    const y = py(row.target ?? 0);
    const marker = svgNode('g', {
      class: 'ob-target',
      'data-element': row.member.element,
      'data-y': y,
    });
    marker.append(
      svgNode('line', {
        x1: x,
        y1: py(row.orbit ?? 0),
        x2: x,
        y2: y,
        stroke: 'var(--color-warning)',
        'stroke-dasharray': '2 2',
      }),
      svgNode('path', {
        d: `M${x} ${y - 6}l6 6-6 6-6-6z`,
        fill: 'var(--color-warning)',
      }),
      svgNode(
        'text',
        { x: x + 9, y: y - 5, fill: 'var(--color-warning)' },
        `${(row.target ?? 0) >= 0 ? '+' : '−'}${Math.abs(row.target ?? 0)}`
      )
    );
    plot.append(marker);
  }

  const visibleCorrectors = correctors.filter((member) => onView(member.s_m));
  const width = Math.min(
    14,
    Math.max(
      2,
      ((STRIP_W - STRIP_LEFT - STRIP_RIGHT) / Math.max(1, visibleCorrectors.length)) * 0.45
    )
  );
  const manual = selection.rule === 'manual';
  const unsolved = new Set(model.orm.unsolved ?? []);
  for (const member of visibleCorrectors) {
    const on = selection.correctors.includes(member.element);
    const blocked = unsolved.has(member.element);
    const classes = ['ob-cor', on ? 'on' : '', blocked ? 'unsolved hollow' : ''].filter(Boolean);
    const cursor = blocked ? 'not-allowed' : manual ? 'pointer' : 'default';
    plot.append(
      titled(
        svgNode('rect', {
          class: classes.join(' '),
          'data-element': member.element,
          x: px(member.s_m) - width / 2,
          y: COR_Y,
          width,
          height: COR_H,
          fill: blocked ? 'none' : on ? 'var(--color-warning)' : 'var(--text-muted)',
          stroke: blocked ? 'var(--text-muted)' : 'none',
          style: `cursor:${cursor}`,
        }),
        blocked ? `${member.element}: the ORM could not solve it` : member.element
      )
    );
    const step = steps.get(member.element);
    if (step && largestStep > 0) {
      const height = (Math.abs(step) * DELTA_H) / largestStep;
      plot.append(
        svgNode('rect', {
          class: 'ob-delta',
          'data-element': member.element,
          x: px(member.s_m) - width / 2,
          y: step > 0 ? COR_Y - height : COR_Y,
          width,
          height,
          fill: 'var(--art-violet-light)',
          style: 'pointer-events:none',
        })
      );
    }
  }
  // Zoomed in, every BPM and corrector in view is named: BPMs between the plot
  // and the corrector row, correctors under their ticks.
  if (zoomed) {
    const label = (/** @type {Member} */ member, /** @type {number} */ y) =>
      svgNode(
        'text',
        {
          class: 'ob-label',
          'data-label': member.element,
          x: px(member.s_m),
          y,
          'text-anchor': 'middle',
          style: 'pointer-events:none',
        },
        member.element
      );
    for (const row of rows) {
      if (onView(row.member.s_m)) plot.append(label(row.member, COR_Y - 8));
    }
    for (const member of visibleCorrectors) plot.append(label(member, COR_Y + DELTA_H + 10));
  }
  nodes.push(plot);

  const drag = drags.get(root);
  if (drag?.mode === 'select') {
    const a = Math.max(STRIP_LEFT, Math.min(drag.x0, drag.x1));
    const b = Math.min(right, Math.max(drag.x0, drag.x1));
    nodes.push(
      svgNode('rect', {
        class: 'ob-zoom-select',
        x: a,
        y: 0,
        width: Math.max(0, b - a),
        height: COR_Y - 4,
        fill: 'var(--color-accent)',
        'fill-opacity': 0.15,
        stroke: 'var(--color-accent)',
        'stroke-dasharray': '3 2',
        style: 'pointer-events:none',
      })
    );
  }
  svg.replaceChildren(...nodes);

  const inView = rows
    .map((row) => row.member)
    .filter((member) => onView(member.s_m))
    .sort((a, b) => a.s_m - b.s_m)
    .map((member) => member.element);
  renderRangeChip(root, panel.view, inView);
}

/** @type {[Rule, string][]} */
const RULES = [
  ['auto', 'auto · N+2'],
  ['mml:[-1 1]', '[-1 1]'],
  ['mml:[-2 -1 1 2]', '[-2 -1 1 2]'],
  ['mml:[-3 -1 1 3]', '[-3 -1 1 3]'],
  ['manual', 'manual (click ticks)'],
];

/**
 * A reading for a table cell, signed, in the members' own unit; a dash when missing.
 * @param {number | null | undefined} value
 * @returns {string}
 */
function cellNumber(value) {
  if (value == null || !Number.isFinite(value)) return '—';
  const magnitude = Math.abs(value);
  return `${value < 0 ? '−' : '+'}${magnitude.toFixed(magnitude >= 1000 ? 0 : 3)}`;
}

/**
 * @param {string} tag
 * @param {Record<string, string>} [attrs]
 * @param {string} [text]
 * @returns {HTMLElement}
 */
function htmlNode(tag, attrs = {}, text) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
  if (text != null) node.textContent = text;
  return node;
}

/**
 * The controls under the strip (bump table, corrector rule, actions), made with
 * their handlers on first use inside the body field.
 * @param {Element} root
 * @returns {HTMLElement}
 */
function controls(root) {
  const body = field(root, 'body');
  const found = body.querySelector('[data-field="controls"]');
  if (found instanceof HTMLElement) return found;

  const box = htmlNode('div', { class: 'ob-cols', 'data-field': 'controls' });

  const bump = htmlNode('div', { class: 'osprey-card ob-box' });
  const bumpHead = htmlNode('h3', { class: 'osprey-section-label' });
  const title = htmlNode('span', {}, 'Bump · ');
  title.append(htmlNode('span', { 'data-field': 'bump-count' }, '0'), ' BPM');
  const add = htmlNode('label', { class: 'ob-add' }, 'add: click a BPM, or ');
  const addSelect = htmlNode('select', { 'data-field': 'add-bpm', 'aria-label': 'add BPM' });
  add.append(addSelect);
  bumpHead.append(title, add);
  const table = htmlNode('table', { class: 'ob-table', 'data-field': 'bump' });
  const headRow = htmlNode('tr');
  for (const [name, text] of [
    ['bpm', 'BPM'],
    ['now', 'now'],
    ['target', 'target'],
    ['predicted', 'predicted'],
    ['remove', ''],
  ]) {
    headRow.append(htmlNode('th', { 'data-head': name }, text));
  }
  const head = htmlNode('thead');
  head.append(headRow);
  table.append(head, htmlNode('tbody'));
  const empty = htmlNode(
    'p',
    { class: 'osprey-list-empty', 'data-field': 'bump-empty' },
    'No BPM in the bump yet — click one on the strip.'
  );
  bump.append(bumpHead, table, empty);

  const cors = htmlNode('div', { class: 'osprey-card ob-box' });
  const corsHead = htmlNode('h3', { class: 'osprey-section-label' });
  const rule = htmlNode('select', { 'data-field': 'rule', 'aria-label': 'corrector rule' });
  for (const [value, text] of RULES) rule.append(htmlNode('option', { value }, text));
  corsHead.append(htmlNode('span', {}, 'Correctors'), rule);
  const unsolved = htmlNode('p', { class: 'ob-meta', 'data-field': 'unsolved' });
  unsolved.hidden = true;
  const plan = htmlNode('div', { 'data-field': 'card' });
  plan.hidden = true;
  cors.append(corsHead, unsolved, plan);

  const actions = htmlNode('div', { class: 'ob-actions', 'data-field': 'actions' });
  actions.append(
    htmlNode(
      'button',
      { type: 'button', class: 'osprey-btn primary', 'data-field': 'compute' },
      'Compute'
    ),
    htmlNode(
      'button',
      { type: 'button', class: 'osprey-btn ob-apply', 'data-field': 'apply' },
      'Apply…'
    ),
    htmlNode('button', { type: 'button', class: 'osprey-btn', 'data-field': 'clear' }, 'Clear'),
    htmlNode('span', { class: 'ob-meta', 'data-field': 'apply-note' }),
    htmlNode('span', { class: 'ob-meta ob-status', 'data-field': 'plan-status' })
  );

  box.append(bump, cors, actions);

  box.addEventListener('input', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    const element = target.getAttribute('data-target');
    if (!element || target.value.trim() === '') return;
    const value = Number(target.value);
    if (!Number.isFinite(value)) return;
    const { targets } = panelOf(root).selection;
    if (!(element in targets)) return;
    setSelection(root, { targets: { ...targets, [element]: value } });
  });
  box.addEventListener('click', (event) => {
    const hit = event.target instanceof Element ? event.target.closest('button') : null;
    if (!hit) return;
    const removed = hit.getAttribute('data-remove');
    if (removed) {
      const targets = { ...panelOf(root).selection.targets };
      delete targets[removed];
      setSelection(root, { targets });
    } else if (hit.getAttribute('data-field') === 'clear') {
      void clearBump(root);
    } else if (hit.getAttribute('data-field') === 'compute') {
      void compute(root);
    } else if (hit.getAttribute('data-field') === 'apply') {
      openConfirm(root);
    }
  });
  addSelect.addEventListener('change', () => {
    const select = /** @type {HTMLSelectElement} */ (addSelect);
    const element = select.value;
    select.value = '';
    if (!element) return;
    const { targets } = panelOf(root).selection;
    if (element in targets) return;
    setSelection(root, { targets: { ...targets, [element]: 0 } });
  });
  rule.addEventListener('change', () => {
    const value = /** @type {Rule} */ (/** @type {HTMLSelectElement} */ (rule).value);
    /** @type {Partial<Selection>} */
    const changes = { rule: value };
    if (value !== 'manual') changes.correctors = [];
    setSelection(root, changes);
  });

  body.append(box);
  return box;
}

/**
 * One table row for a bump BPM; its input is made once and kept for the row's life.
 * @param {string} element
 * @returns {HTMLTableRowElement}
 */
function bumpRow(element) {
  const row = /** @type {HTMLTableRowElement} */ (htmlNode('tr', { 'data-element': element }));
  const target = htmlNode('td');
  target.append(
    htmlNode('input', {
      type: 'number',
      step: 'any',
      'data-target': element,
      'aria-label': `target offset of ${element}`,
    })
  );
  const remove = htmlNode('td');
  remove.append(
    htmlNode(
      'button',
      {
        type: 'button',
        class: 'ob-link',
        'data-remove': element,
        'aria-label': `remove ${element}`,
      },
      '✕'
    )
  );
  row.append(
    htmlNode('td', { 'data-cell': 'bpm' }, element),
    htmlNode('td', { 'data-cell': 'now' }),
    target,
    htmlNode('td', { 'data-cell': 'predicted' }),
    remove
  );
  return row;
}

/**
 * Bring the bump table, add dropdown, rule and unsolved note in line with the
 * state and the selection. Rows are keyed by element: a row and its input
 * survive every poll, and an input is never written while it has focus.
 * @param {Element} root
 * @param {Panel} panel
 */
function renderTable(root, panel) {
  const box = controls(root);
  const { selection, state } = panel;
  const model = selectedModel(panel);
  const bpms = model ? model.members.bpms.filter((member) => member.plane === selection.plane) : [];
  const byElement = new Map(bpms.map((member) => [member.element, member]));
  const position = state?.orbit?.position ?? {};
  const golden = state?.orbit?.golden ?? {};
  const plan = state?.plan;
  const predicted =
    plan && model && plan.model === model.name && plan.plane === selection.plane
      ? (plan.predicted ?? null)
      : null;

  /** @param {Member | undefined} member */
  const goldOf = (member) => (member?.golden_addr ? golden[member.golden_addr] : null);
  /** @param {Member | undefined} member */
  const nowOf = (member) =>
    offset(member?.position_addr ? position[member.position_addr] : null, goldOf(member));

  const unit = bpms.find((member) => member.unit)?.unit ?? '';
  for (const name of ['now', 'target', 'predicted']) {
    const th = box.querySelector(`th[data-head="${name}"]`);
    if (th) th.textContent = unit ? `${name} ${unit}` : name;
  }

  const order = (/** @type {string} */ element) => byElement.get(element)?.s_m ?? Infinity;
  const elements = Object.keys(selection.targets).sort((a, b) => order(a) - order(b));
  const tbody = /** @type {HTMLTableSectionElement} */ (field(root, 'bump').querySelector('tbody'));
  /** @type {Map<string, HTMLTableRowElement>} */
  const existing = new Map();
  for (const row of [...tbody.children]) {
    const element = row.getAttribute('data-element') ?? '';
    if (row instanceof HTMLTableRowElement && element in selection.targets) {
      existing.set(element, row);
    } else {
      row.remove();
    }
  }
  elements.forEach((element, place) => {
    let row = existing.get(element);
    if (!row) row = bumpRow(element);
    const here = tbody.children[place] ?? null;
    if (here !== row) tbody.insertBefore(row, here);
    const member = byElement.get(element);
    const now = row.querySelector('[data-cell="now"]');
    if (now) now.textContent = cellNumber(nowOf(member));
    const pred = row.querySelector('[data-cell="predicted"]');
    if (pred)
      pred.textContent = predicted ? cellNumber(offset(predicted[element], goldOf(member))) : '—';
    const input = row.querySelector('input');
    const target = selection.targets[element];
    if (input && input !== document.activeElement) {
      if (input.value === '' || Number(input.value) !== target) input.value = String(target);
    }
  });
  field(root, 'bump-count').textContent = String(elements.length);
  field(root, 'bump-empty').hidden = elements.length > 0;

  const add = /** @type {HTMLSelectElement} */ (field(root, 'add-bpm'));
  const offered = bpms
    .filter((member) => !(member.element in selection.targets) && nowOf(member) != null)
    .map((member) => member.element);
  const listed = [...add.options].map((option) => option.value);
  if (listed.join('\n') !== ['', ...offered].join('\n')) {
    add.replaceChildren(
      htmlNode('option', { value: '' }, 'BPM…'),
      ...offered.map((element) => htmlNode('option', { value: element }, element))
    );
  }
  add.value = '';

  const rule = /** @type {HTMLSelectElement} */ (field(root, 'rule'));
  if (rule !== document.activeElement || rule.value !== selection.rule) rule.value = selection.rule;

  const note = field(root, 'unsolved');
  const planeCorrectors = new Set(
    (model?.members.correctors ?? [])
      .filter((member) => member.plane === selection.plane)
      .map((member) => member.element)
  );
  const skipped =
    plan && model && plan.model === model.name && plan.plane === selection.plane
      ? (plan.skipped_unsolved ?? [])
      : [];
  const unsolved = (model?.orm.unsolved ?? []).filter((element) => planeCorrectors.has(element));
  if (skipped.length > 0) {
    note.textContent = `skipped, the ORM could not solve them: ${skipped.join(', ')}`;
  } else if (unsolved.length > 0) {
    note.textContent = `not selectable, the ORM could not solve them: ${unsolved.join(', ')}`;
  } else {
    note.textContent = '';
  }
  note.hidden = note.textContent === '';
}

/**
 * Discard the backend's plan when state holds one, then apply the changes with
 * the local selection emptied. A failed discard leaves the plan and the
 * selection as they were and shows the backend's text.
 * @param {Element} root
 * @param {Partial<Selection>} changes
 * @param {typeof fetch} [fetcher]
 * @returns {Promise<boolean>} whether the selection was cleared
 */
async function discardThenClear(root, changes, fetcher) {
  const panel = panelOf(root);
  if (panel.state?.plan) {
    try {
      const response = await (fetcher ?? panel.fetcher)(DISCARD_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: '{}',
      });
      if (!response.ok) throw new Error(await refusalText(response));
    } catch (error) {
      showError(root, error instanceof Error ? error.message : String(error));
      render(root);
      return false;
    }
    // A poll sent before the discard still carries the plan: drop it.
    panel.shownGeneration = panel.generation;
    if (panel.state) panel.state = { ...panel.state, plan: null };
  }
  setSelection(root, { ...changes, targets: {}, rule: 'auto', correctors: [] });
  return true;
}

/**
 * Switch plane: discard any plan first, then clear the selection.
 * @param {Element} root
 * @param {Plane} plane
 * @param {typeof fetch} [fetcher]
 * @returns {Promise<boolean>}
 */
export async function choosePlane(root, plane, fetcher) {
  if (panelOf(root).selection.plane === plane) return true;
  return discardThenClear(root, { plane }, fetcher);
}

/**
 * Switch model: discard any plan first, then clear the selection.
 * @param {Element} root
 * @param {string} model
 * @param {typeof fetch} [fetcher]
 * @returns {Promise<boolean>}
 */
export async function chooseModel(root, model, fetcher) {
  if (panelOf(root).selection.model === model) return true;
  return discardThenClear(root, { model }, fetcher);
}

/**
 * Clear: discard any plan first, then empty the selection.
 * @param {Element} root
 * @param {typeof fetch} [fetcher]
 * @returns {Promise<boolean>}
 */
export async function clearBump(root, fetcher) {
  return discardThenClear(root, {}, fetcher);
}

/**
 * Wire the header controls (plane toggle, model dropdown, recompute) and set the
 * fetcher every control of this panel sends through.
 * @param {Element} root
 * @param {typeof fetch} [fetcher]
 */
export function bindControls(root, fetcher) {
  const panel = panelOf(root);
  if (fetcher) panel.fetcher = fetcher;
  field(root, 'plane').addEventListener('click', (event) => {
    const hit = event.target instanceof Element ? event.target.closest('[data-plane]') : null;
    const plane = hit?.getAttribute('data-plane');
    if (plane === 'x' || plane === 'y') void choosePlane(root, plane);
  });
  const select = /** @type {HTMLSelectElement} */ (field(root, 'model'));
  select.addEventListener('change', () => void chooseModel(root, select.value));
  field(root, 'recompute').addEventListener('click', () => void recompute(root, panel.fetcher));
  controls(root);
}

/**
 * Render everything from the state and the selection.
 * @param {Element} root
 */
export function render(root) {
  const panel = panelOf(root);
  renderHeader(root, panel);
  renderStrip(root, panel);
  renderTable(root, panel);
  renderCard(root, panel);
}

/**
 * Show one state: keep it, settle the model choice, take in a new plan, render.
 * @param {Element} root
 * @param {BumpState} state
 */
export function showState(root, state) {
  const panel = panelOf(root);
  panel.state = state;
  const names = (state.models ?? []).map((model) => model.name);
  if (!panel.selection.model || !names.includes(panel.selection.model)) {
    panel.selection = { ...panel.selection, model: names[0] ?? null };
  }
  field(root, 'error').hidden = true;
  hydrate(root, state.plan);
  render(root);
}

/**
 * Show why there is no state, leaving the last one on view.
 * @param {Element} root
 * @param {string} message
 */
export function showError(root, message) {
  const error = field(root, 'error');
  error.textContent = message;
  error.hidden = false;
}

/**
 * Read the state once and show it. Each request carries a generation; a
 * response older than one already shown is dropped.
 * @param {Element} root
 * @param {typeof fetch} [fetcher]
 */
export async function refresh(root, fetcher = fetch) {
  const panel = panelOf(root);
  const generation = ++panel.generation;
  /** @type {BumpState | null} */
  let state = null;
  /** @type {string | null} */
  let failure = null;
  try {
    const response = await fetcher(STATE_URL, { headers: { Accept: 'application/json' } });
    if (!response.ok) throw new Error(`the backend answered ${response.status}`);
    state = await response.json();
  } catch (error) {
    failure = error instanceof Error ? error.message : String(error);
  }
  if (generation <= panel.shownGeneration) return;
  panel.shownGeneration = generation;
  if (state) showState(root, state);
  else showError(root, failure ?? 'no state');
}

/**
 * Run the panel's own Compute. While it is in flight no poll hydrates; its
 * plan id becomes the last one seen, so the next poll does not re-hydrate it.
 * @param {Element} root
 * @param {() => Promise<{id: string} | null>} send
 * @returns {Promise<{id: string} | null>}
 */
export async function ownCompute(root, send) {
  const panel = panelOf(root);
  panel.computesInFlight += 1;
  try {
    const plan = await send();
    if (plan && plan.id != null) panel.lastSeenPlanId = plan.id;
    return plan;
  } finally {
    panel.computesInFlight -= 1;
  }
}

/**
 * Ask the backend to rebuild the selected model's ORM. A refusal shows the
 * answering side's own text (a 403 from the proxy says why).
 * @param {Element} root
 * @param {typeof fetch} [fetcher]
 */
export async function recompute(root, fetcher = fetch) {
  const model = panelOf(root).selection.model;
  if (!model) return;
  try {
    const response = await fetcher(RECOMPUTE_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ model }),
    });
    if (!response.ok) throw new Error(await refusalText(response));
  } catch (error) {
    showError(root, error instanceof Error ? error.message : String(error));
  }
}

/**
 * The backend's plan when it belongs to the selected model and plane, else null.
 * @param {Panel} panel
 * @returns {Plan | null}
 */
function shownPlan(panel) {
  const plan = panel.state?.plan ?? null;
  if (!plan || plan.model !== panel.selection.model || plan.plane !== panel.selection.plane) {
    return null;
  }
  return plan;
}

/**
 * The unit of the selected plane's BPMs, or an empty string.
 * @param {Panel} panel
 * @returns {string}
 */
function bpmUnit(panel) {
  const model = selectedModel(panel);
  const bpms = model ? model.members.bpms.filter((m) => m.plane === panel.selection.plane) : [];
  return bpms.find((member) => member.unit)?.unit ?? '';
}

/**
 * How far a refused new setpoint is outside its limits, for display only; null
 * when the limits do not say.
 * @param {PlanCorrector} row
 * @returns {number | null}
 */
function overshoot(row) {
  const limits = row.verdict?.limits;
  const value = row.new;
  if (!limits || value == null || !Number.isFinite(value)) return null;
  if (limits.max != null && value > limits.max) return value - limits.max;
  if (limits.min != null && value < limits.min) return limits.min - value;
  const step = row.delta == null ? null : Math.abs(row.delta);
  if (limits.max_step != null && step != null && step > limits.max_step) {
    return step - limits.max_step;
  }
  return null;
}

/**
 * A limit bound for a label; an open bound is an infinity.
 * @param {number | null} value
 * @param {string} open
 * @returns {string}
 */
function bound(value, open) {
  return value == null ? open : String(value);
}

/**
 * The verdict label of one row, and whether it is a refusal.
 * @param {PlanCorrector} row
 * @returns {{text: string, refused: boolean}}
 */
function verdictLabel(row) {
  const verdict = row.verdict;
  if (!verdict) return { text: '', refused: false };
  if (!verdict.allowed) {
    const over = overshoot(row);
    const unit = row.unit ? ` ${row.unit}` : '';
    const parts = [verdict.reason || 'refused'];
    if (over != null) parts.push(`over by ${over.toFixed(3)}${unit}`);
    return { text: parts.join(' · '), refused: true };
  }
  if (!verdict.enforced) return { text: 'limits checking off', refused: false };
  if (verdict.limits) {
    const { min, max } = verdict.limits;
    return { text: `limit ${bound(min, '−∞')}..${bound(max, '∞')}`, refused: false };
  }
  return { text: `no declared limit (mode: ${verdict.mode ?? 'none'})`, refused: false };
}

/**
 * The per-row note: the missing tolerance and an unsettled readback.
 * @param {PlanCorrector} row
 * @returns {string}
 */
function rowNote(row) {
  const notes = [];
  if (row.tolerance == null) notes.push('no tolerance: not staleness-checked');
  if (row.settled === false) notes.push(`not settled, readback ${cellNumber(row.readback)}`);
  return notes.join(' · ');
}

/**
 * The closure line: singular values kept, the largest predicted leakage, and
 * the BPMs and correctors the solve left out.
 * @param {Plan} plan
 * @param {string} unit
 * @returns {string}
 */
function closureText(plan, unit) {
  const parts = [];
  if (plan.sigma_kept != null) parts.push(`σ kept ${plan.sigma_kept}/${plan.sigma_total ?? '?'}`);
  if (plan.predicted_closure_max != null) {
    const suffix = unit ? ` ${unit}` : '';
    parts.push(`max predicted leakage ${cellNumber(plan.predicted_closure_max)}${suffix}`);
  }
  if (plan.dropped_bpms?.length) parts.push(`dropped BPMs: ${plan.dropped_bpms.join(', ')}`);
  if (plan.skipped_unsolved?.length) {
    parts.push(`skipped (ORM unsolved): ${plan.skipped_unsolved.join(', ')}`);
  }
  return parts.join(' · ');
}

/**
 * The plan's corrector rows as a table. In the Correctors box: now, Δ and new
 * as in the mockup, with outcomes; on the confirm card: the channel each row
 * writes, without outcomes.
 * @param {Plan} plan
 * @param {boolean} inBox
 * @returns {HTMLElement}
 */
function rowsTable(plan, inBox) {
  const table = htmlNode('table', { class: 'ob-table' });
  const unit = plan.correctors.find((row) => row.unit)?.unit ?? '';
  const withUnit = (/** @type {string} */ name) => (unit ? `${name} ${unit}` : name);
  /** @type {[string, string, (row: PlanCorrector) => HTMLElement][]} */
  const columns = [
    ['corrector', 'corrector', (row) => htmlNode('td', { 'data-cell': 'corrector' }, row.element)],
    [
      'channel',
      'channel',
      (row) => htmlNode('td', { 'data-cell': 'channel' }, row.setpoint_addr ?? ''),
    ],
    ['now', withUnit('now'), (row) => htmlNode('td', { 'data-cell': 'now' }, cellNumber(row.now))],
    [
      'delta',
      withUnit('Δ'),
      (row) => htmlNode('td', { 'data-cell': 'delta' }, cellNumber(row.delta)),
    ],
    ['new', withUnit('new'), (row) => htmlNode('td', { 'data-cell': 'new' }, cellNumber(row.new))],
    [
      'verdict',
      'verdict',
      (row) => {
        const verdict = verdictLabel(row);
        const node = htmlNode('td', { 'data-cell': 'verdict' }, verdict.text);
        if (verdict.refused) node.classList.add('ob-refused');
        return node;
      },
    ],
    ['note', '', (row) => htmlNode('td', { 'data-cell': 'note', class: 'ob-meta' }, rowNote(row))],
    [
      'outcome',
      'outcome',
      (row) =>
        htmlNode(
          'td',
          { 'data-cell': 'outcome' },
          row.outcome ? row.outcome.replace('_', ' ') : ''
        ),
    ],
  ];
  const shown = columns.filter(([name]) => (inBox ? name !== 'channel' : name !== 'outcome'));
  const headRow = htmlNode('tr');
  for (const [name, text] of shown) headRow.append(htmlNode('th', { 'data-head': name }, text));
  const head = htmlNode('thead');
  head.append(headRow);
  const body = htmlNode('tbody');
  for (const row of plan.correctors) {
    const tr = htmlNode('tr', { 'data-element': row.element });
    tr.append(...shown.map(([, , make]) => make(row)));
    body.append(tr);
  }
  table.append(head, body);
  return table;
}

/**
 * The plan inside the Correctors box, made with the controls.
 * @param {Element} root
 * @returns {HTMLElement}
 */
function card(root) {
  controls(root);
  return field(root, 'card');
}

/**
 * The plan status at the end of the actions row: what a ready plan holds, the
 * phase while applying, the status otherwise.
 * @param {Plan} plan
 * @returns {string}
 */
function statusText(plan) {
  if (plan.status === 'ready') {
    const targets = Object.keys(plan.targets ?? {}).length;
    const correctors = plan.correctors.length;
    return `plan #${plan.id} · ${targets} target${targets === 1 ? '' : 's'}, ${correctors} corrector${correctors === 1 ? '' : 's'}`;
  }
  if (plan.status === 'applying' && plan.phase) return `applying · ${plan.phase}`;
  return plan.status;
}

/**
 * @param {number} count
 * @returns {string}
 */
function setpoints(count) {
  return `${count} setpoint${count === 1 ? '' : 's'}`;
}

/**
 * Render the plan card and the Compute and Apply buttons from the state.
 * @param {Element} root
 * @param {Panel} panel
 */
function renderCard(root, panel) {
  const model = selectedModel(panel);
  const plan = shownPlan(panel);
  const allowed = panel.state?.posture?.allowed ?? false;

  const computeButton = /** @type {HTMLButtonElement} */ (field(root, 'compute'));
  computeButton.hidden = panel.readOnly || !model;
  computeButton.disabled =
    !model ||
    model.orm.status !== 'ready' ||
    Object.keys(panel.selection.targets).length === 0 ||
    panel.computesInFlight > 0 ||
    panel.applying;

  const applyButton = /** @type {HTMLButtonElement} */ (field(root, 'apply'));
  applyButton.hidden = panel.readOnly || !plan;
  applyButton.textContent = plan ? `Apply ${setpoints(plan.correctors.length)}…` : 'Apply…';
  applyButton.disabled = !plan || plan.status !== 'ready' || !allowed || panel.applying;
  const note = field(root, 'apply-note');
  note.textContent =
    plan && !panel.readOnly && !allowed ? (panel.state?.posture?.phrase ?? '') : '';
  note.hidden = note.textContent === '';
  if (note.textContent) applyButton.title = note.textContent;
  else applyButton.removeAttribute('title');

  const status = field(root, 'plan-status');
  status.textContent = plan ? statusText(plan) : '';
  status.hidden = !plan;

  const node = card(root);
  if (!plan) {
    node.hidden = true;
    node.replaceChildren();
    return;
  }
  node.hidden = false;
  const unit = bpmUnit(panel);
  const children = [];
  if (plan.status === 'not_applicable' && plan.reason) {
    children.push(htmlNode('p', { class: 'ob-refused', 'data-field': 'plan-reason' }, plan.reason));
  }
  children.push(rowsTable(plan, true));
  children.push(
    htmlNode('p', { class: 'ob-meta', 'data-field': 'closure' }, closureText(plan, unit))
  );
  const residual = Object.entries(plan.residual_after ?? {});
  if (residual.length > 0) {
    const label = unit ? `residual after (${unit})` : 'residual after';
    const text = residual.map(([element, value]) => `${element} ${cellNumber(value)}`).join(' · ');
    children.push(
      htmlNode('p', { class: 'ob-meta', 'data-field': 'residual' }, `${label}: ${text}`)
    );
  }
  node.replaceChildren(...children);
}

/**
 * @param {Element} root
 */
function closeConfirm(root) {
  const confirm = field(root, 'confirm');
  confirm.hidden = true;
  confirm.replaceChildren();
}

/**
 * The backend's refusal text: the `detail` of a JSON body, else the body
 * itself, else the status when the body is empty.
 * @param {Response} response
 * @returns {Promise<string>}
 */
export async function refusalText(response) {
  const text = (await response.text()).trim();
  try {
    const detail = JSON.parse(text)?.detail;
    if (typeof detail === 'string' && detail.trim()) return detail.trim();
  } catch {
    // not JSON: the text is the refusal
  }
  return text || `the backend answered ${response.status}`;
}

/**
 * Compute a plan from the selection through the panel's own Compute. A 403
 * makes the panel read-only; any other refusal shows the backend's text.
 * @param {Element} root
 */
export async function compute(root) {
  const panel = panelOf(root);
  const model = selectedModel(panel);
  const { selection } = panel;
  if (panel.readOnly || !model || model.orm.status !== 'ready') return;
  if (Object.keys(selection.targets).length === 0) return;
  /** @type {Record<string, unknown>} */
  const body = {
    model: model.name,
    plane: selection.plane,
    targets: { ...selection.targets },
    rule: selection.rule,
  };
  if (selection.rule === 'manual') body.correctors = [...selection.correctors];
  const send = async () => {
    render(root);
    try {
      const response = await panel.fetcher(COMPUTE_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(body),
      });
      if (response.status === 403) {
        panel.readOnly = true;
        closeConfirm(root);
        showNotice(root, `read-only: ${await refusalText(response)}`);
        return null;
      }
      if (!response.ok) throw new Error(await refusalText(response));
      /** @type {Plan} */
      const plan = await response.json();
      // A poll sent before the Compute still carries the old plan: drop it.
      panel.shownGeneration = panel.generation;
      if (panel.state) panel.state = { ...panel.state, plan };
      closeConfirm(root);
      field(root, 'error').hidden = true;
      return plan;
    } catch (error) {
      showError(root, error instanceof Error ? error.message : String(error));
      return null;
    }
  };
  await ownCompute(root, send);
  render(root);
}

/**
 * Open the in-panel confirm card for the shown plan: every setpoint, the
 * closure, and the Confirm and Cancel buttons. Only a ready plan opens it.
 * @param {Element} root
 */
export function openConfirm(root) {
  const panel = panelOf(root);
  const plan = shownPlan(panel);
  if (!plan || plan.status !== 'ready' || panel.readOnly || panel.applying) return;
  if (!(panel.state?.posture?.allowed ?? false)) return;
  const confirm = field(root, 'confirm');
  const box = htmlNode('div', {
    class: 'osprey-card ob-box',
    role: 'dialog',
    'aria-label': 'confirm apply',
  });
  const ok = htmlNode(
    'button',
    { type: 'button', class: 'osprey-btn ob-apply', 'data-field': 'confirm-apply' },
    `Apply ${setpoints(plan.correctors.length)}`
  );
  const cancel = htmlNode(
    'button',
    { type: 'button', class: 'osprey-btn', 'data-field': 'confirm-cancel' },
    'Cancel'
  );
  ok.addEventListener('click', () => void applyPlan(root));
  cancel.addEventListener('click', () => closeConfirm(root));
  const buttons = htmlNode('div', { class: 'ob-actions' });
  buttons.append(cancel, ok);
  box.append(
    htmlNode(
      'h3',
      { class: 'osprey-section-label' },
      `Apply plan ${plan.id} to ${plan.model} · ${plan.plane}`
    ),
    rowsTable(plan, false),
    htmlNode('p', { class: 'ob-meta' }, closureText(plan, bpmUnit(panel))),
    buttons
  );
  confirm.replaceChildren(box);
  confirm.hidden = false;
}

/**
 * Apply the shown plan: post its id and setpoint digest, then show the plan the
 * backend returns with its outcomes. A refusal shows the backend's text.
 * @param {Element} root
 */
export async function applyPlan(root) {
  const panel = panelOf(root);
  const plan = shownPlan(panel);
  closeConfirm(root);
  if (!plan || plan.status !== 'ready' || panel.applying) return;
  panel.applying = true;
  render(root);
  try {
    const response = await panel.fetcher(APPLY_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ plan_id: plan.id, setpoint_digest: plan.setpoint_digest }),
    });
    if (!response.ok) throw new Error(await refusalText(response));
    /** @type {Plan} */
    const done = await response.json();
    panel.shownGeneration = panel.generation;
    if (panel.state) panel.state = { ...panel.state, plan: done };
    field(root, 'error').hidden = true;
  } catch (error) {
    showError(root, error instanceof Error ? error.message : String(error));
  } finally {
    panel.applying = false;
    render(root);
  }
}

const root = document.getElementById('orbit-bump');
if (root) {
  bindControls(root);
  void refresh(root);
  setInterval(() => void refresh(root), REFRESH_MS);
}
