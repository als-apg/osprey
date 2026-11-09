// The orbit-bump panel: one selection object (what the operator is building) and
// a state loop that polls the backend and renders from state plus selection.
// Every address is relative: the page is served under the terminal's /panel/<id>/.

import {
  declinesToBrowser,
  isContextMenuKey,
  openContextMenu,
} from '/design-system/js/context-menu.js';
import { closePopover, togglePopover } from '/design-system/js/popover.js';
import { initTooltips } from '/design-system/js/tooltip.js';

const STATE_URL = 'api/state';
const RECOMPUTE_URL = 'api/recompute';
const DISCARD_URL = 'api/discard';
const COMPUTE_URL = 'api/compute';
const APPLY_URL = 'api/apply';
const DRAFT_URL = 'api/draft';
const REFRESH_MS = 2000;
// The pause after the last edit before the draft is published for the agent.
const DRAFT_DEBOUNCE_MS = 300;

/**
 * @typedef {'x' | 'y'} Plane
 * @typedef {'auto' | 'mml:[-1 1]' | 'mml:[-2 -1 1 2]' | 'mml:[-3 -1 1 3]' | 'manual'} Rule
 * @typedef {'idle' | 'building' | 'ready' | 'failed'} OrmStatus
 * @typedef {'flat' | 'peak' | 'cos'} Shape
 * @typedef {'closure' | 'targets'} Priority
 * @typedef {'free' | 'hold' | 'zero'} Others
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
 * @property {Rule} [rule] the corrector rule the plan was computed with
 * @property {PlanCorrector[]} correctors
 * @property {Record<string, number | null>} [predicted] position per BPM element after the bump
 * @property {{element: string, predicted_residual: number}[]} [leakage]
 * @property {number | null} [predicted_closure_max]
 * @property {number} [sigma_kept]
 * @property {number} [sigma_total]
 * @property {string[]} [skipped_unsolved] correctors the rule skipped because the ORM could not solve them
 * @property {string[]} [dropped_bpms] non-bump BPMs left out as unreadable
 * @property {Record<string, number>} [residual_after]
 * @property {Priority} [priority] close the bump first, or hit the targets first
 * @property {Others} [others] what every BPM of the plane not in the bump aims at
 * @property {boolean} [all_correctors] whether every solved corrector of the plane was used
 * @property {Record<string, number>} [miss] per bump BPM, predicted less its aim
 * @property {string[]} [missed] bump BPMs whose miss is past the threshold
 * @property {number} [target_miss_max] the largest |miss|
 * @property {string[]} [bounded] correctors the solve left at a limit
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
 * @property {Priority} priority
 * @property {Others} others what every BPM not in the bump aims at
 * @property {boolean} all_correctors every solved corrector of the plane, rule aside
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
 * @property {number} step the nudge step, in the BPM unit
 * @property {Shape} shape how the amplitude spreads over the bump
 * @property {number} amp the bump amplitude: the signed target of largest magnitude
 * @property {string | null} anchor the BPM picked last on the strip, where ⇧-click extends from
 * @property {{id: string, key: string} | null} computed the plan id this panel computed or took
 *   in last, with the selection it was computed for
 * @property {typeof fetch} fetcher what the panel's own controls send through
 * @property {boolean} bound whether bindControls wired this panel; only a wired one publishes its draft
 * @property {string | null} draftSent the draft last published, as sent
 * @property {string | null} draftPending the draft waiting out the debounce
 * @property {ReturnType<typeof setTimeout> | null} draftTimer
 */

/** @type {WeakMap<Element, Panel>} */
const panels = new WeakMap();

/** @returns {Selection} */
function emptySelection() {
  return {
    plane: 'x',
    model: null,
    targets: {},
    rule: 'auto',
    correctors: [],
    priority: 'closure',
    others: 'hold',
    all_correctors: false,
  };
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
      step: 0.01,
      shape: 'flat',
      amp: 0,
      anchor: null,
      computed: null,
      fetcher: (input, init) => fetch(input, init),
      bound: false,
      draftSent: null,
      draftPending: null,
      draftTimer: null,
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
  panel.amp = amplitudeOf(plan.targets);
  const rule =
    plan.rule && ALL_RULES.includes(/** @type {Rule} */ (plan.rule)) ? plan.rule : 'manual';
  panel.selection = {
    ...panel.selection,
    plane: plan.plane,
    model: plan.model,
    targets: { ...plan.targets },
    rule,
    correctors: rule === 'manual' ? plan.correctors.map((row) => row.element) : [],
    priority: plan.priority === 'targets' ? 'targets' : 'closure',
    others: plan.others && OTHERS_MODES.includes(plan.others) ? plan.others : 'hold',
    all_correctors: plan.all_correctors === true,
  };
  panel.computed = { id: plan.id, key: selectionKey(panel.selection) };
}

/**
 * What a plan is computed from, as one comparable string.
 * @param {Selection} selection
 * @returns {string}
 */
function selectionKey(selection) {
  const targets = Object.entries(selection.targets)
    .map(([element, value]) => [element, tidy(value)])
    .sort(([a], [b]) => String(a).localeCompare(String(b)));
  const correctors = selection.rule === 'manual' ? [...selection.correctors].sort() : [];
  return JSON.stringify([
    selection.plane,
    selection.model,
    selection.rule,
    targets,
    correctors,
    selection.priority,
    selection.others,
    selection.all_correctors,
  ]);
}

/**
 * Whether the selection was edited since the plan was computed: compared with
 * the selection the plan was computed for when this panel knows it, else with
 * the plan's own targets.
 * @param {Panel} panel
 * @param {Plan} plan
 * @returns {boolean}
 */
function edited(panel, plan) {
  if (panel.computed?.id === plan.id) return panel.computed.key !== selectionKey(panel.selection);
  const ours = Object.entries(panel.selection.targets);
  const theirs = plan.targets ?? {};
  if (ours.length !== Object.keys(theirs).length) return true;
  return ours.some(
    ([element, value]) => !(element in theirs) || tidy(theirs[element]) !== tidy(value)
  );
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

// Strip geometry in SVG user units: a plot band; under it the bump's grip; then
// a plan's Δ bars around their own zero axis; then the corrector ticks; the
// s-axis at the foot. A left margin holds the vertical labels.
const STRIP_W = 1000;
const STRIP_H = 242;
const STRIP_LEFT = 34;
const STRIP_RIGHT = 12;
const STRIP_TOP = 10;
const PLOT_BOTTOM = 148;
const LABEL_Y = 158;
const GRIP_Y = 163;
const GRIP_H = 14;
const DELTA_AXIS = 193;
const DELTA_HALF = 10;
const COR_Y = 207;
const COR_H = 8;
const STRIP_MID = (STRIP_TOP + PLOT_BOTTOM) / 2;
const HEADROOM = 1.15;
const S_TICKS = 4;
const HANDLE_HIT = 11;
const BAND_PAD = 10;

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

/**
 * A target value with the float noise of repeated steps taken off.
 * @param {number} value
 * @returns {number}
 */
function tidy(value) {
  const rounded = Math.round(value * 1e6) / 1e6;
  return Object.is(rounded, -0) ? 0 : rounded;
}

// Zoom, in metres: the narrowest window as a fraction of the ring, the wheel's
// zoom step, the shortest drag (user units) that counts as a range.
const MIN_SPAN_FRACTION = 0.01;
const WHEEL_STEP = 1.25;
const MIN_DRAG = 4;

/**
 * A drag in flight on one strip. On the empty plot: a marquee that picks the
 * BPMs it spans, a ⇧-drag that zooms to its range, or a pan. On a target
 * handle or the bump's grip: the targets move with the pointer, on the
 * vertical scale frozen when the drag started.
 * @typedef {object} Drag
 * @property {'marquee' | 'zoom' | 'pan' | 'handle' | 'grip'} mode
 * @property {number} x0 where the drag started, in user units
 * @property {number} x1 where the pointer is now, in user units
 * @property {number} y0 where the drag started, vertically, in user units
 * @property {{s0: number, s1: number}} view0 the window when the drag started
 * @property {number} yMax the vertical scale when the drag started
 * @property {string} [element] the handle's BPM
 * @property {Record<string, number>} [targets0] the targets when the drag started
 * @property {boolean} moved
 */

/** @type {WeakMap<Element, Drag>} */
const drags = new WeakMap();

/** @type {WeakMap<Element, number>} the vertical scale of each strip's last render */
const scales = new WeakMap();

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
 * The pointer's position in the strip's user units, or null when the strip has
 * no laid-out width (a hidden tile) and no position can be read.
 * @param {SVGElement} svg
 * @param {MouseEvent} event
 * @returns {{x: number, y: number} | null}
 */
function svgPoint(svg, event) {
  const box = svg.getBoundingClientRect();
  if (!(box.width > 0)) return null;
  const x = ((event.clientX - box.left) * STRIP_W) / box.width;
  const y = box.height > 0 ? ((event.clientY - box.top) * STRIP_H) / box.height : NaN;
  if (!Number.isFinite(x)) return null;
  // The wheel needs only x; a reading without a usable y counts as the top.
  return { x, y: Number.isFinite(y) ? y : 0 };
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
 * The strip's svg inside the body field, made (with its pointer, key and menu
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
    'aria-label': 'orbit strip',
  });
  svg.addEventListener('click', (event) => pickOnStrip(root, event));
  bindStrip(root, svg);
  body.prepend(svg);
  return svg;
}

/**
 * The plane's BPMs in ring order: by s, then by index for elements sharing one s.
 * @param {Panel} panel
 * @returns {Member[]}
 */
function planeBpms(panel) {
  const model = selectedModel(panel);
  const bpms = model ? model.members.bpms.filter((m) => m.plane === panel.selection.plane) : [];
  return [...bpms].sort((a, b) => a.s_m - b.s_m || a.index - b.index);
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
 * A BPM's reading relative to golden, or null when either is missing.
 * @param {Panel} panel
 * @param {Member | undefined} member
 * @returns {number | null}
 */
function orbitOf(panel, member) {
  if (!member) return null;
  const position = panel.state?.orbit?.position ?? {};
  const golden = panel.state?.orbit?.golden ?? {};
  return offset(
    member.position_addr ? position[member.position_addr] : null,
    member.golden_addr ? golden[member.golden_addr] : null
  );
}

/**
 * The BPMs that can join a bump, in ring order: those with a reading and a golden.
 * @param {Panel} panel
 * @returns {string[]}
 */
function pickable(panel) {
  return planeBpms(panel)
    .filter((member) => orbitOf(panel, member) != null)
    .map((member) => member.element);
}

/**
 * The bump BPMs in ring order.
 * @param {Panel} panel
 * @returns {string[]}
 */
function bumpOrder(panel) {
  const order = new Map(planeBpms(panel).map((member, at) => [member.element, at]));
  return Object.keys(panel.selection.targets).sort(
    (a, b) => (order.get(a) ?? Infinity) - (order.get(b) ?? Infinity)
  );
}

/**
 * Replace the targets, keeping the amplitude readout in step with them.
 * @param {Element} root
 * @param {Record<string, number>} targets
 */
function setTargets(root, targets) {
  const panel = panelOf(root);
  panel.amp = amplitudeOf(targets);
  setSelection(root, { targets });
}

/**
 * The signed target with the largest magnitude; zero for no targets.
 * @param {Record<string, number>} targets
 * @returns {number}
 */
function amplitudeOf(targets) {
  let amp = 0;
  for (const value of Object.values(targets)) if (Math.abs(value) > Math.abs(amp)) amp = value;
  return tidy(amp);
}

/**
 * Add BPMs to the bump at `value`; BPMs already in it keep their target, and
 * BPMs without a reading are left out.
 * @param {Element} root
 * @param {string[]} elements
 * @param {number} [value]
 */
export function addBpms(root, elements, value = 0) {
  const panel = panelOf(root);
  const allowed = new Set(pickable(panel));
  const targets = { ...panel.selection.targets };
  let changed = false;
  for (const element of elements) {
    if (element in targets || !allowed.has(element)) continue;
    targets[element] = value;
    changed = true;
  }
  if (changed) setTargets(root, targets);
}

/**
 * @param {Element} root
 * @param {string[]} elements
 */
export function removeBpms(root, elements) {
  const targets = { ...panelOf(root).selection.targets };
  for (const element of elements) delete targets[element];
  setTargets(root, targets);
}

/**
 * @param {Element} root
 * @param {string} element
 */
function toggleBpm(root, element) {
  if (element in panelOf(root).selection.targets) removeBpms(root, [element]);
  else addBpms(root, [element]);
}

/**
 * Add every pickable BPM from the last picked one to `element`, both included.
 * @param {Element} root
 * @param {string} element
 */
function extendTo(root, element) {
  const panel = panelOf(root);
  const order = planeBpms(panel).map((member) => member.element);
  const from = panel.anchor == null ? -1 : order.indexOf(panel.anchor);
  const to = order.indexOf(element);
  if (from < 0 || to < 0) {
    addBpms(root, [element]);
  } else {
    addBpms(root, order.slice(Math.min(from, to), Math.max(from, to) + 1));
  }
  panel.anchor = element;
}

/**
 * Move the targets of `elements` by `steps` nudge steps.
 * @param {Element} root
 * @param {string[]} elements
 * @param {number} steps
 */
export function nudge(root, elements, steps) {
  const panel = panelOf(root);
  const targets = { ...panel.selection.targets };
  for (const element of elements) {
    if (element in targets) targets[element] = tidy(targets[element] + steps * panel.step);
  }
  setTargets(root, targets);
}

/**
 * Set one BPM's target.
 * @param {Element} root
 * @param {string} element
 * @param {number} value
 */
function setTarget(root, element, value) {
  const { targets } = panelOf(root).selection;
  if (!(element in targets) || !Number.isFinite(value)) return;
  setTargets(root, { ...targets, [element]: tidy(value) });
}

/** @type {Shape[]} */
const SHAPES = ['flat', 'peak', 'cos'];

/**
 * The weight of each of `count` BPMs under a shape, in ring order: flat is 1
 * everywhere; peak a triangle and cos a raised cosine, both 1 at the centre and
 * 0 at the two ends. Under three BPMs every shape is flat.
 * @param {number} count
 * @param {Shape} shape
 * @returns {number[]}
 */
export function shapeWeights(count, shape) {
  return Array.from({ length: count }, (_, at) => {
    if (count < 3 || shape === 'flat') return 1;
    const t = at / (count - 1);
    return shape === 'peak' ? 1 - Math.abs(2 * t - 1) : 0.5 - 0.5 * Math.cos(2 * Math.PI * t);
  });
}

/**
 * Spread the amplitude over the bump BPMs in the chosen shape.
 * @param {Element} root
 */
function applyShape(root) {
  const panel = panelOf(root);
  const elements = bumpOrder(panel);
  if (elements.length === 0) {
    render(root);
    return;
  }
  const weights = shapeWeights(elements.length, panel.shape);
  /** @type {Record<string, number>} */
  const targets = {};
  elements.forEach((element, at) => {
    targets[element] = tidy(panel.amp * weights[at]);
  });
  setSelection(root, { targets });
}

/**
 * Set the amplitude and spread it over the bump.
 * @param {Element} root
 * @param {number} amp
 */
export function setAmplitude(root, amp) {
  if (!Number.isFinite(amp)) return;
  panelOf(root).amp = tidy(amp);
  applyShape(root);
}

/**
 * Choose the shape and spread the amplitude over the bump in it.
 * @param {Element} root
 * @param {Shape} shape
 */
export function setShape(root, shape) {
  panelOf(root).shape = shape;
  applyShape(root);
}

/**
 * Widen the bump by one pickable BPM on each side. Under flat the new BPMs take
 * the amplitude; under the other shapes the bump is spread again.
 * @param {Element} root
 */
export function widen(root) {
  const panel = panelOf(root);
  const order = pickable(panel);
  const bump = bumpOrder(panel).filter((element) => order.includes(element));
  if (bump.length === 0) return;
  const first = order.indexOf(bump[0]);
  const last = order.indexOf(bump[bump.length - 1]);
  const added = [order[first - 1], order[last + 1]].filter((element) => element != null);
  const targets = { ...panel.selection.targets };
  for (const element of added) targets[element] = panel.shape === 'flat' ? panel.amp : 0;
  panel.selection = { ...panel.selection, targets };
  applyShape(root);
}

/** The nudge steps offered, in the BPM unit. */
const STEPS = [0.001, 0.01, 0.1];

/**
 * A nudge step for a label: micrometres when the unit is mm.
 * @param {number} step
 * @param {string} unit
 * @returns {string}
 */
function stepLabel(step, unit) {
  if (unit === 'mm') return `${Math.round(step * 1000)} µm`;
  return `${step}${unit ? ` ${unit}` : ''}`;
}

/**
 * Choose the nudge step.
 * @param {Element} root
 * @param {number} step
 */
export function setStep(root, step) {
  if (!(step > 0)) return;
  panelOf(root).step = step;
  render(root);
}

/**
 * A click on the strip: a BPM dot toggles in the bump (⇧ extends from the last
 * one picked); a corrector tick toggles under the manual rule. Hollow dots and
 * unsolved ticks are never picked.
 * @param {Element} root
 * @param {MouseEvent} event
 */
function pickOnStrip(root, event) {
  const hit = event.target instanceof Element ? event.target.closest('[data-element]') : null;
  const element = hit?.getAttribute('data-element');
  if (!hit || !element || hit.classList.contains('hollow')) return;
  const panel = panelOf(root);
  const { selection } = panel;
  if (hit.classList.contains('ob-bpm')) {
    if (event.shiftKey && panel.anchor != null) {
      extendTo(root, element);
      return;
    }
    toggleBpm(root, element);
    panel.anchor = element;
  } else if (hit.classList.contains('ob-cor') && selection.rule === 'manual') {
    const correctors = selection.correctors.includes(element)
      ? selection.correctors.filter((name) => name !== element)
      : [...selection.correctors, element];
    setSelection(root, { correctors });
  }
}

/**
 * Wire the strip's gestures. On the empty plot a drag picks the BPMs it spans,
 * a ⇧-drag zooms to its range, and an Alt- or middle-button drag pans when
 * zoomed; the wheel zooms around the cursor and a double-click returns to the
 * whole ring. A target handle drags vertically and takes ↑/↓ (⇧ ×10),
 * ⌫/Delete and ←/→; the grip under the plot moves every target together.
 * A right-click opens the strip's context menu.
 * @param {Element} root
 * @param {SVGElement} svg
 */
function bindStrip(root, svg) {
  const current = () => {
    const panel = panelOf(root);
    const model = selectedModel(panel);
    return model ? shownView(panel, model) : null;
  };
  svg.addEventListener(
    'wheel',
    (event) => {
      const view = current();
      const point = svgPoint(svg, event);
      if (!view || point == null || event.deltaY === 0) return;
      event.preventDefault();
      const factor = event.deltaY > 0 ? WHEEL_STEP : 1 / WHEEL_STEP;
      const anchor = sAt(point.x, view);
      const span = (view.s1 - view.s0) * factor;
      const s0 = anchor - (anchor - view.s0) * factor;
      setView(root, { s0, s1: s0 + span });
    },
    { passive: false }
  );
  svg.addEventListener('pointerdown', (event) => {
    if (event.button === 2 || !(event.target instanceof Element)) return;
    const view = current();
    const point = svgPoint(svg, event);
    if (!view || point == null) return;
    const panel = panelOf(root);
    const handle = event.target.closest('.ob-target');
    const grip = event.target.closest('[data-grip]');
    /** @type {Drag['mode'] | null} */
    let mode = null;
    if (handle) mode = 'handle';
    else if (grip) mode = 'grip';
    else if (event.target.classList.contains('ob-bg')) {
      const panning = event.altKey || event.button === 1;
      if (event.shiftKey) mode = 'zoom';
      else if (panning) mode = panel.view != null ? 'pan' : null;
      else mode = 'marquee';
    }
    if (!mode) return;
    try {
      svg.setPointerCapture(event.pointerId);
    } catch {
      // No active pointer to capture (a synthetic event): the drag works without it.
    }
    drags.set(root, {
      mode,
      x0: point.x,
      x1: point.x,
      y0: point.y,
      view0: view,
      yMax: scales.get(root) ?? 1,
      element: handle?.getAttribute('data-element') ?? undefined,
      targets0: { ...panel.selection.targets },
      moved: false,
    });
    if (mode === 'handle' || mode === 'grip') {
      event.preventDefault();
      if (handle instanceof SVGElement) handle.focus({ preventScroll: true });
    }
  });
  svg.addEventListener('pointermove', (event) => {
    const drag = drags.get(root);
    const point = svgPoint(svg, event);
    if (!drag || point == null) return;
    drag.x1 = point.x;
    drag.moved = drag.moved || Math.abs(point.x - drag.x0) >= 1 || Math.abs(point.y - drag.y0) >= 1;
    if (drag.mode === 'marquee' || drag.mode === 'zoom') {
      renderStrip(root, panelOf(root));
    } else if (drag.mode === 'pan') {
      const shift = sAt(point.x, drag.view0) - sAt(drag.x0, drag.view0);
      setView(root, { s0: drag.view0.s0 - shift, s1: drag.view0.s1 - shift });
    } else {
      const panel = panelOf(root);
      const raw = ((drag.y0 - point.y) * drag.yMax) / (STRIP_MID - STRIP_TOP);
      const delta = Math.round(raw / panel.step) * panel.step;
      const targets0 = drag.targets0 ?? {};
      /** @type {Record<string, number>} */
      const targets = { ...panel.selection.targets };
      if (drag.mode === 'handle' && drag.element && drag.element in targets0) {
        targets[drag.element] = tidy(targets0[drag.element] + delta);
      } else if (drag.mode === 'grip') {
        for (const [element, value] of Object.entries(targets0)) {
          if (element in targets) targets[element] = tidy(value + delta);
        }
      }
      setTargets(root, targets);
    }
  });
  const finish = (/** @type {PointerEvent} */ event) => {
    const drag = drags.get(root);
    if (!drag) return;
    drags.delete(root);
    const point = event.type === 'pointerup' ? svgPoint(svg, event) : null;
    if (point != null) drag.x1 = point.x;
    const wide = event.type === 'pointerup' && Math.abs(drag.x1 - drag.x0) >= MIN_DRAG;
    if (wide && (drag.mode === 'marquee' || drag.mode === 'zoom')) {
      const a = sAt(Math.min(drag.x0, drag.x1), drag.view0);
      const b = sAt(Math.max(drag.x0, drag.x1), drag.view0);
      if (drag.mode === 'zoom') {
        setView(root, { s0: a, s1: b });
        return;
      }
      const panel = panelOf(root);
      const allowed = new Set(pickable(panel));
      const hits = planeBpms(panel)
        .filter((member) => member.s_m >= a && member.s_m <= b && allowed.has(member.element))
        .map((member) => member.element);
      if (hits.length > 0) {
        panel.anchor = hits[hits.length - 1];
        addBpms(root, hits);
        return;
      }
    }
    render(root);
  };
  svg.addEventListener('pointerup', finish);
  svg.addEventListener('pointercancel', finish);
  svg.addEventListener('dblclick', (event) => {
    if (event.target instanceof Element && event.target.closest('.ob-target, [data-grip]')) return;
    resetView(root);
  });
  svg.addEventListener('keydown', (event) => {
    const handle = event.target instanceof Element ? event.target.closest('.ob-target') : null;
    const element = handle?.getAttribute('data-element');
    if (!handle || !element) return;
    if (isContextMenuKey(event)) {
      event.preventDefault();
      openStripMenu(root, svg, {
        x: handle.getBoundingClientRect().left,
        y: handle.getBoundingClientRect().bottom,
        element,
        grip: false,
      });
      return;
    }
    const big = event.shiftKey ? 10 : 1;
    if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
      event.preventDefault();
      nudge(root, [element], event.key === 'ArrowUp' ? big : -big);
    } else if (event.key === 'Backspace' || event.key === 'Delete') {
      event.preventDefault();
      const order = bumpOrder(panelOf(root));
      const next = order[order.indexOf(element) + 1] ?? order[order.indexOf(element) - 1];
      removeBpms(root, [element]);
      if (next) focusHandle(root, next);
    } else if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault();
      const order = bumpOrder(panelOf(root));
      const next = order[order.indexOf(element) + (event.key === 'ArrowRight' ? 1 : -1)];
      if (next) focusHandle(root, next);
    }
  });
  svg.addEventListener('contextmenu', (event) => {
    if (declinesToBrowser(event)) return;
    event.preventDefault();
    const target = event.target instanceof Element ? event.target : null;
    const hit = target?.closest('.ob-bpm, .ob-target');
    const point = svgPoint(svg, event);
    const view = current();
    openStripMenu(root, svg, {
      x: event.clientX,
      y: event.clientY,
      element: hit?.getAttribute('data-element') ?? null,
      grip: !!target?.closest('[data-grip]'),
      s: point && view ? sAt(point.x, view) : undefined,
    });
  });
}

/**
 * Focus the target handle of a BPM, if it is drawn.
 * @param {Element} root
 * @param {string} element
 */
function focusHandle(root, element) {
  const handle = stripSvg(root).querySelector(`.ob-target[data-element="${element}"]`);
  if (handle instanceof SVGElement) handle.focus({ preventScroll: true });
}

/**
 * Focus and select a bump row's target input.
 * @param {Element} root
 * @param {string} element
 */
function focusTargetInput(root, element) {
  const input = field(root, 'bump').querySelector(`input[data-target="${element}"]`);
  if (input instanceof HTMLInputElement) {
    input.focus();
    input.select();
  }
}

/**
 * Open the strip's context menu: verbs for the BPM or grip under the pointer,
 * then the zoom verbs and the nudge step.
 * @param {Element} root
 * @param {SVGElement} svg
 * @param {{x: number, y: number, element: string | null, grip: boolean, s?: number}} at
 */
function openStripMenu(root, svg, at) {
  const panel = panelOf(root);
  const model = selectedModel(panel);
  if (!model) return;
  const { targets } = panel.selection;
  const bump = bumpOrder(panel);
  const allowed = new Set(pickable(panel));
  /** @type {import('/design-system/js/context-menu.js').MenuItem[]} */
  const items = [];
  const { element } = at;
  if (element) {
    const on = element in targets;
    items.push({ heading: element });
    items.push(
      on
        ? { label: 'Remove from bump', hint: '⌫', run: () => removeBpms(root, [element]) }
        : {
            label: 'Add to bump',
            hint: 'click',
            disabled: !allowed.has(element),
            run: () => {
              addBpms(root, [element]);
              panel.anchor = element;
            },
          }
    );
    if (on) {
      items.push(
        { label: 'Set target…', run: () => focusTargetInput(root, element) },
        { label: 'Zero target', run: () => setTarget(root, element, 0) }
      );
    }
    items.push(
      {
        label: 'Extend bump to here',
        hint: '⇧click',
        disabled: panel.anchor == null || on || !allowed.has(element),
        run: () => extendTo(root, element),
      },
      { divider: true }
    );
  } else if (at.grip) {
    items.push(
      {
        label: 'Zero the bump',
        run: () => setTargets(root, Object.fromEntries(bump.map((name) => [name, 0]))),
      },
      {
        label: 'Invert the bump',
        run: () =>
          setTargets(
            root,
            Object.fromEntries(bump.map((name) => [name, tidy(-(targets[name] ?? 0))]))
          ),
      },
      {
        label: 'Remove all',
        danger: true,
        disabled: bump.length === 0,
        run: () => setTargets(root, {}),
      },
      { divider: true }
    );
  }
  const byElement = new Map(planeBpms(panel).map((member) => [member.element, member]));
  const sOf = (/** @type {string | undefined} */ name) => byElement.get(name ?? '')?.s_m;
  const first = sOf(bump[0]);
  const last = sOf(bump[bump.length - 1]);
  const s = at.s;
  items.push(
    {
      label: 'Zoom to bump',
      disabled: first == null || last == null,
      run: () => {
        if (first == null || last == null) return;
        const pad = Math.max(model.length_m * 0.02, (last - first) * 0.4);
        setView(root, { s0: first - pad, s1: last + pad });
      },
    },
    {
      label: 'Zoom in here',
      hint: 'wheel',
      disabled: s == null,
      run: () => {
        if (s == null) return;
        const view = shownView(panel, model);
        const half = (view.s1 - view.s0) / 4;
        setView(root, { s0: s - half, s1: s + half });
      },
    },
    {
      label: 'Whole ring',
      hint: 'dbl-click',
      disabled: panel.view == null,
      run: () => resetView(root),
    },
    { divider: true },
    { heading: 'Nudge step' }
  );
  const unit = bpmUnit(panel);
  for (const step of STEPS) {
    items.push({
      label: stepLabel(step, unit),
      checked: panel.step === step,
      run: () => setStep(root, step),
    });
  }
  openContextMenu({
    x: at.x,
    y: at.y,
    anchorEl: /** @type {HTMLElement} */ (/** @type {unknown} */ (svg)),
    ariaLabel: 'orbit strip',
    items,
  });
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
 * bump's band, grip and target handles, the plan's predicted orbit and Δ bars,
 * BPM dots and corrector ticks at their s-positions. Values are in the BPM
 * members' own unit.
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
  const bpms = planeBpms(panel);
  const correctors = model.members.correctors.filter((member) => member.plane === plane);
  panel.view = clampView(panel.view, model.length_m);
  const zoomed = panel.view != null;
  const { s0, s1 } = shownView(panel, model);
  const span = s1 - s0 > 0 ? s1 - s0 : 1;
  const golden = state?.orbit?.golden ?? {};
  const planHere = shownPlan(panel);
  const predicted = planHere?.predicted ?? null;
  /** @type {Map<string, number>} */
  const steps = new Map();
  for (const row of planHere?.correctors ?? []) {
    if (row.delta != null && Number.isFinite(row.delta)) steps.set(row.element, row.delta);
  }
  const planned = new Set((planHere?.correctors ?? []).map((row) => row.element));
  const missed = new Set(planHere?.missed ?? []);
  const largestStep = Math.max(0, ...[...steps.values()].map(Math.abs));
  const drag = drags.get(root);

  const rows = bpms.map((member) => {
    const gold = member.golden_addr ? golden[member.golden_addr] : null;
    return {
      member,
      orbit: orbitOf(panel, member),
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
  const frozen = drag && (drag.mode === 'handle' || drag.mode === 'grip') ? drag.yMax : null;
  const yMax = frozen ?? (largest > 0 ? largest * HEADROOM : 1);
  scales.set(root, yMax);
  const px = (/** @type {number} */ s) =>
    STRIP_LEFT + ((s - s0) * (STRIP_W - STRIP_LEFT - STRIP_RIGHT)) / span;
  const py = (/** @type {number} */ v) => STRIP_MID - (v * (STRIP_MID - STRIP_TOP)) / yMax;
  const right = STRIP_W - STRIP_RIGHT;
  const clampX = (/** @type {number} */ x) => Math.min(right, Math.max(STRIP_LEFT, x));

  svg.setAttribute('data-s0', String(s0));
  svg.setAttribute('data-s1', String(s1));
  svg.classList.toggle('dragging', drag?.mode === 'handle' || drag?.mode === 'grip');
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
  // The empty plot, under everything: a drag that starts here picks, zooms or
  // pans, while dots, handles and ticks drawn on top keep their own gestures.
  nodes.push(
    svgNode('rect', {
      class: 'ob-bg',
      x: STRIP_LEFT,
      y: 0,
      width: STRIP_W - STRIP_LEFT - STRIP_RIGHT,
      height: DELTA_AXIS - DELTA_HALF,
      fill: 'transparent',
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

  // The bump's band, from its first to its last BPM, and the grip under the plot.
  const bump = bumpOrder(panel);
  const byElement = new Map(bpms.map((member) => [member.element, member]));
  const firstS = byElement.get(bump[0])?.s_m;
  const lastS = byElement.get(bump[bump.length - 1])?.s_m;
  if (firstS != null && lastS != null && lastS >= s0 && firstS <= s1) {
    const x0 = clampX(px(firstS) - BAND_PAD);
    const x1 = clampX(px(lastS) + BAND_PAD);
    plot.append(
      svgNode('rect', {
        class: 'ob-band',
        x: x0,
        y: STRIP_TOP - 4,
        width: Math.max(0, x1 - x0),
        height: PLOT_BOTTOM - STRIP_TOP + 8,
      }),
      svgNode('line', {
        class: 'ob-band-edge',
        x1: x0,
        x2: x0,
        y1: STRIP_TOP - 4,
        y2: PLOT_BOTTOM + 4,
      }),
      svgNode('line', {
        class: 'ob-band-edge',
        x1: x1,
        x2: x1,
        y1: STRIP_TOP - 4,
        y2: PLOT_BOTTOM + 4,
      })
    );
    const label =
      bump.length === 1
        ? `${bump[0]}  ⇕`
        : `${bump[0]}–${bump[bump.length - 1]} · ${bump.length} BPM  ⇕`;
    const width = Math.max(x1 - x0, label.length * 6.2 + 16);
    const centre = (x0 + x1) / 2;
    const gx = Math.min(right - width, Math.max(STRIP_LEFT, centre - width / 2));
    const grip = svgNode('rect', {
      class: 'ob-grip',
      'data-grip': '1',
      x: gx,
      y: GRIP_Y,
      width,
      height: GRIP_H,
      rx: 3,
    });
    grip.setAttribute('data-tip', 'drag to move every target');
    nodes.push(
      grip,
      svgNode(
        'text',
        { class: 'ob-grip-text', x: gx + width / 2, y: GRIP_Y + 10.5, 'text-anchor': 'middle' },
        label
      )
    );
  }

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

  // The aim of every held or zeroed BPM: its orbit now under hold, golden under zero.
  if (bump.length > 0 && selection.others !== 'free') {
    for (const row of rows) {
      if (row.target != null || row.orbit == null || !onView(row.member.s_m)) continue;
      const aim = selection.others === 'zero' ? 0 : row.orbit;
      plot.append(
        svgNode('circle', {
          class: 'ob-aim',
          'data-element': row.member.element,
          cx: px(row.member.s_m),
          cy: py(aim),
          r: 6,
          fill: 'none',
          stroke: 'var(--color-warning)',
        })
      );
    }
  }

  for (const row of rows) {
    const { member } = row;
    if (!onView(member.s_m)) continue;
    const hollow = row.orbit == null;
    const on = row.target != null;
    const classes = ['ob-bpm', on ? 'on' : '', hollow ? 'hollow' : ''].filter(Boolean);
    const colour = on ? 'var(--color-warning)' : 'var(--color-accent)';
    const reading = row.orbit == null ? '' : ` · ${cellNumber(row.orbit)}${unit ? ` ${unit}` : ''}`;
    plot.append(
      titled(
        svgNode('circle', {
          class: classes.join(' '),
          'data-element': member.element,
          cx: px(member.s_m),
          cy: py(row.orbit ?? 0),
          r: on ? 4.5 : 3.5,
          fill: hollow ? 'none' : colour,
          stroke: colour,
        }),
        hollow ? `${member.element}: no reading or no golden` : `${member.element}${reading}`
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
  // Handles last in the plot, so they sit over the dots and lines.
  for (const row of targets) {
    const x = px(row.member.s_m);
    const y = py(row.target ?? 0);
    const value = row.target ?? 0;
    const handle = svgNode('g', {
      class: 'ob-target',
      'data-element': row.member.element,
      'data-y': y,
      tabindex: 0,
      role: 'slider',
      'aria-label': `${row.member.element} target`,
      'aria-valuenow': value,
      'aria-valuetext': `${cellNumber(value)}${unit ? ` ${unit}` : ''}`,
    });
    if (missed.has(row.member.element)) {
      handle.classList.add('ob-missed');
      titled(handle, `${row.member.element}: target missed`);
    }
    handle.append(
      svgNode('line', { class: 'ob-tline', x1: x, y1: py(row.orbit ?? 0), x2: x, y2: y }),
      svgNode('circle', { class: 'ob-hit', cx: x, cy: y, r: HANDLE_HIT }),
      svgNode('circle', { class: 'ob-miss-ring', cx: x, cy: y, r: 9 }),
      svgNode('path', { class: 'ob-diamond', d: `M${x} ${y - 6}l6 6-6 6-6-6z` }),
      svgNode(
        'text',
        { class: 'ob-handle-label', x, y: y - 9, 'text-anchor': 'middle' },
        cellNumber(value)
      )
    );
    plot.append(handle);
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
  const corUnit = correctors.find((member) => member.unit)?.unit ?? '';
  if (largestStep > 0) {
    nodes.push(
      svgNode('line', {
        class: 'ob-grid',
        'data-field': 'delta-axis',
        x1: STRIP_LEFT,
        y1: DELTA_AXIS,
        x2: right,
        y2: DELTA_AXIS,
      }),
      svgNode('text', { x: 2, y: DELTA_AXIS + 3 }, `Δ${corUnit}`)
    );
  }
  for (const member of visibleCorrectors) {
    const on = selection.correctors.includes(member.element);
    const blocked = unsolved.has(member.element);
    const step = steps.get(member.element);
    const classes = [
      'ob-cor',
      on ? 'on' : '',
      planned.has(member.element) ? 'picked' : '',
      blocked ? 'unsolved hollow' : '',
      manual && !blocked ? 'pickable' : '',
    ].filter(Boolean);
    const tick = svgNode('rect', {
      class: classes.join(' '),
      'data-element': member.element,
      x: px(member.s_m) - width / 2,
      y: COR_Y,
      width,
      height: COR_H,
    });
    tick.setAttribute(
      'data-tip',
      blocked
        ? `${member.element}: the ORM could not solve it`
        : step != null
          ? `${member.element} · Δ ${cellNumber(step)}${corUnit ? ` ${corUnit}` : ''}`
          : member.element
    );
    plot.append(tick);
    if (step && largestStep > 0) {
      const height = (Math.abs(step) * DELTA_HALF) / largestStep;
      plot.append(
        svgNode('rect', {
          class: 'ob-delta',
          'data-element': member.element,
          x: px(member.s_m) - width / 2,
          y: step > 0 ? DELTA_AXIS - height : DELTA_AXIS,
          width,
          height,
        })
      );
    }
  }
  // Zoomed in, every BPM and corrector in view is named: BPMs under the plot,
  // correctors under their ticks.
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
        },
        member.element
      );
    for (const row of rows) {
      if (onView(row.member.s_m)) plot.append(label(row.member, LABEL_Y));
    }
    for (const member of visibleCorrectors) plot.append(label(member, COR_Y + COR_H + 10));
  }
  nodes.push(plot);

  if (drag?.mode === 'marquee' || drag?.mode === 'zoom') {
    const a = clampX(Math.min(drag.x0, drag.x1));
    const b = clampX(Math.max(drag.x0, drag.x1));
    nodes.push(
      svgNode('rect', {
        class: drag.mode === 'zoom' ? 'ob-zoom-select' : 'ob-marquee',
        x: a,
        y: STRIP_TOP - 4,
        width: Math.max(0, b - a),
        height: PLOT_BOTTOM - STRIP_TOP + 8,
      })
    );
  }

  // Redrawing replaces the handles: keep the keyboard on the one that had it.
  const active = svg.ownerDocument.activeElement;
  const focused =
    active && svg.contains(active)
      ? active.closest('.ob-target')?.getAttribute('data-element')
      : null;
  svg.replaceChildren(...nodes);
  if (focused) {
    const again = svg.querySelector(`.ob-target[data-element="${focused}"]`);
    if (again instanceof SVGElement) again.focus({ preventScroll: true });
  }

  const inView = rows
    .map((row) => row.member)
    .filter((member) => onView(member.s_m))
    .map((member) => member.element);
  renderRangeChip(root, panel.view, inView);
}

/**
 * The rules the Correctors box offers: auto (one upstream, every corrector
 * between, two downstream), tight (MML [-1 1]), manual (the clicked ticks).
 * @type {[Rule, string, string][]}
 */
const RULES = [
  ['auto', 'auto', 'one corrector upstream, every corrector between, two downstream'],
  ['mml:[-1 1]', 'tight', 'MML [-1 1]: the nearest corrector on each side of the bump'],
  ['manual', 'manual', 'the corrector ticks you click on the strip'],
];

/**
 * The wider MML corrector lists, in the overflow menu after the rule toggle.
 * @type {[Rule, string][]}
 */
const MORE_RULES = [
  ['mml:[-2 -1 1 2]', '±2 (−2 −1 1 2)'],
  ['mml:[-3 -1 1 3]', '±3 (−3 −1 1 3)'],
];

/**
 * The solve priorities, in the overflow menu.
 * @type {[Priority, string, string][]}
 */
const PRIORITIES = [
  ['closure', 'close bump', 'hold the other BPMs first; targets may be missed'],
  [
    'targets',
    'hit targets',
    'meet the targets exactly where the correctors can; the other BPMs may leak',
  ],
];

/**
 * What every BPM of the plane not in the bump aims at, on the Bump card.
 * @type {[Others, string, string][]}
 */
const OTHERS = [
  ['free', 'free', 'other BPMs unconstrained: only the targets are solved for'],
  ['hold', 'hold', 'other BPMs stay at their orbit at Compute'],
  ['zero', 'zero', 'other BPMs go to golden'],
];

/** @type {Others[]} */
const OTHERS_MODES = OTHERS.map(([value]) => value);

/** @type {Rule[]} */
const ALL_RULES = [...RULES.map(([value]) => value), ...MORE_RULES.map(([value]) => value)];

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
 * A Lucide-style stroke icon.
 * @param {string} d the path data
 * @returns {SVGElement}
 */
function icon(d) {
  const svg = svgNode('svg', { class: 'osprey-ic', viewBox: '0 0 24 24', 'aria-hidden': 'true' });
  svg.append(svgNode('path', { d }));
  return svg;
}

/**
 * A −/input/+ stepper.
 * @param {Record<string, string>} input the input's attributes
 * @param {string} name what the buttons step, for their labels
 * @param {string} [unitField] the data-field of a unit label after the input
 * @returns {HTMLElement}
 */
function stepper(input, name, unitField) {
  const box = htmlNode('span', { class: 'ob-stepper' });
  const less = htmlNode(
    'button',
    {
      type: 'button',
      class: 'osprey-icon-btn',
      'data-step-dir': '-1',
      'aria-label': `lower ${name}`,
    },
    '−'
  );
  const more = htmlNode(
    'button',
    {
      type: 'button',
      class: 'osprey-icon-btn',
      'data-step-dir': '1',
      'aria-label': `raise ${name}`,
    },
    '+'
  );
  box.append(less, htmlNode('input', { type: 'number', ...input }));
  if (unitField) box.append(htmlNode('span', { class: 'ob-unit', 'data-field': unitField }));
  box.append(more);
  return box;
}

/** Ids for the picker's listbox and options, unique per page. */
let pickerSeq = 0;

/**
 * The BPMs a picker query names, in ring order: `22` the 22nd BPM, `21-23`
 * the 21st to the 23rd, `2*` the BPMs whose name (with any leading "BPM" taken
 * off) starts with 2, anything else the BPMs whose name contains it. A leading
 * "BPM" in the query is ignored; an empty query names every BPM.
 * @param {string} query
 * @param {string[]} names the BPMs in ring order
 * @returns {string[]}
 */
export function matchBpms(query, names) {
  const raw = query.trim().toLowerCase();
  const q = raw.replace(/^bpm\s*/, '');
  if (!q) return [...names];
  const bare = (/** @type {string} */ name) => name.toLowerCase().replace(/^bpm[\s_:-]*/, '');
  const range = q.match(/^(\d+)\s*[-–]\s*(\d+)$/);
  if (range) {
    const [a, b] = [Number(range[1]), Number(range[2])].sort((x, y) => x - y);
    return names.slice(Math.max(0, a - 1), b);
  }
  if (q.endsWith('*')) {
    const prefix = q.slice(0, -1);
    return names.filter((n) => bare(n).startsWith(prefix) || n.toLowerCase().startsWith(prefix));
  }
  if (/^\d+$/.test(q)) {
    const exact = names.filter((n) => bare(n) === q || bare(n) === String(Number(q)));
    if (exact.length > 0) return exact;
    const nth = names[Number(q) - 1];
    return nth ? [nth] : [];
  }
  return names.filter((n) => n.toLowerCase().includes(q) || n.toLowerCase().includes(raw));
}

/**
 * The BPM picker in the Bump card's head: a combobox input over a listbox of
 * the BPMs its query names. A row toggles its BPM without closing; ↑/↓ arm a
 * row and Enter picks it, or with none armed toggles a single match or adds
 * every match; Escape and a press outside close the list.
 * @param {Element} root
 * @returns {HTMLElement}
 */
function picker(root) {
  const seq = ++pickerSeq;
  const listId = `ob-pick-list-${seq}`;
  const wrap = htmlNode('span', { class: 'ob-picker' });
  const label = htmlNode('label', { class: 'osprey-search-field' });
  const input = /** @type {HTMLInputElement} */ (
    htmlNode('input', {
      'data-field': 'pick',
      placeholder: 'add BPM… 22, 21-23, 2*',
      autocomplete: 'off',
      spellcheck: 'false',
      role: 'combobox',
      'aria-label': 'add BPMs',
      'aria-autocomplete': 'list',
      'aria-expanded': 'false',
      'aria-controls': listId,
      title: 'BPM by number, a range 21-23, or a prefix 2*',
    })
  );
  label.append(icon('M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14zM20 20l-3.5-3.5'), input);
  label.append(htmlNode('span', { class: 'osprey-kbd' }, '↵'));
  const list = htmlNode('div', {
    class: 'osprey-popover',
    id: listId,
    role: 'listbox',
    'aria-label': 'BPMs',
    'aria-multiselectable': 'true',
    'data-field': 'pick-list',
  });
  list.hidden = true;
  wrap.append(label, list);

  let armed = -1;
  /** @type {string[]} */
  let matches = [];

  const rows = () => /** @type {HTMLElement[]} */ ([...list.querySelectorAll('[role="option"]')]);
  const arm = (/** @type {number} */ next) => {
    const all = rows();
    armed = all.length === 0 ? -1 : next;
    all.forEach((row, at) => row.classList.toggle('armed', at === armed));
    const row = all[armed];
    if (row) {
      input.setAttribute('aria-activedescendant', row.id);
      if (typeof row.scrollIntoView === 'function') row.scrollIntoView({ block: 'nearest' });
    } else {
      input.removeAttribute('aria-activedescendant');
    }
  };
  const onOutside = (/** @type {Event} */ event) => {
    if (!(event.target instanceof Node) || !wrap.contains(event.target)) close();
  };
  const close = () => {
    list.hidden = true;
    input.setAttribute('aria-expanded', 'false');
    input.removeAttribute('aria-activedescendant');
    armed = -1;
    input.ownerDocument.removeEventListener('pointerdown', onOutside, true);
  };
  const fill = () => {
    const panel = panelOf(root);
    const keep = rows()[armed]?.getAttribute('data-name') ?? null;
    const names = pickable(panel);
    matches = matchBpms(input.value, names);
    const byElement = new Map(planeBpms(panel).map((member) => [member.element, member]));
    const unit = bpmUnit(panel);
    /** @type {HTMLElement[]} */
    const children = [];
    const query = input.value.trim();
    if (query && matches.length > 1) {
      const all = htmlNode('button', {
        type: 'button',
        class: 'osprey-popover-item',
        role: 'option',
        id: `${listId}-all`,
        'aria-selected': 'false',
        'data-all': '1',
      });
      all.append(
        htmlNode('span', { class: 'ob-check' }, '+'),
        htmlNode('span', {}, `add all ${matches.length} shown`)
      );
      children.push(all);
    }
    if (matches.length === 0) {
      children.push(htmlNode('div', { class: 'osprey-list-empty' }, `no BPM matches "${query}"`));
    }
    const { targets } = panel.selection;
    matches.slice(0, 60).forEach((name, at) => {
      const on = name in targets;
      const member = byElement.get(name);
      const row = htmlNode('button', {
        type: 'button',
        class: 'osprey-popover-item',
        role: 'option',
        id: `${listId}-${at}`,
        'aria-selected': String(on),
        'data-name': name,
      });
      const value = on ? targets[name] : orbitOf(panel, member);
      const where = member ? `${member.s_m.toFixed(1)} m · ` : '';
      row.append(
        htmlNode('span', { class: 'ob-check' }, on ? '✓' : ''),
        htmlNode('span', {}, name),
        htmlNode(
          'span',
          { class: 'osprey-popover-aside' },
          `${where}${cellNumber(value)}${unit ? ` ${unit}` : ''}`
        )
      );
      children.push(row);
    });
    list.replaceChildren(...children);
    const again = keep ? rows().findIndex((row) => row.getAttribute('data-name') === keep) : -1;
    arm(again);
  };
  const open = () => {
    fill();
    if (list.hidden) {
      list.hidden = false;
      input.setAttribute('aria-expanded', 'true');
      input.ownerDocument.addEventListener('pointerdown', onOutside, true);
    }
  };
  const activate = (/** @type {HTMLElement} */ row) => {
    if (row.hasAttribute('data-all')) {
      addBpms(root, matches);
      input.value = '';
    } else {
      const name = row.getAttribute('data-name');
      if (name) toggleBpm(root, name);
    }
    open();
  };

  input.addEventListener('focus', open);
  input.addEventListener('input', () => {
    armed = -1;
    open();
  });
  input.addEventListener('keydown', (event) => {
    const all = rows();
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      if (list.hidden) open();
      if (all.length === 0) return;
      const step = event.key === 'ArrowDown' ? 1 : -1;
      arm(armed < 0 ? (step > 0 ? 0 : all.length - 1) : (armed + step + all.length) % all.length);
    } else if (event.key === 'Enter') {
      event.preventDefault();
      const row = all[armed];
      if (row) {
        activate(row);
        return;
      }
      if (matches.length === 1) toggleBpm(root, matches[0]);
      else if (matches.length > 1 && input.value.trim()) addBpms(root, matches);
      input.value = '';
      open();
    } else if (event.key === 'Escape') {
      if (!list.hidden) {
        event.preventDefault();
        event.stopPropagation();
        close();
      }
    }
  });
  // Rows keep the input's focus: the press would otherwise blur it.
  list.addEventListener('mousedown', (event) => event.preventDefault());
  list.addEventListener('pointerdown', (event) => event.preventDefault());
  list.addEventListener('click', (event) => {
    const row = event.target instanceof Element ? event.target.closest('[role="option"]') : null;
    if (row instanceof HTMLElement) activate(row);
  });
  wrap.addEventListener('ob-refresh', () => {
    if (!list.hidden) fill();
  });
  return wrap;
}

/**
 * A segmented control.
 * @param {string} name its data-field
 * @param {string} attr the data attribute each button carries its value in
 * @param {[string, string, string?][]} options value, label and tooltip
 * @param {string} label its accessible name
 * @returns {HTMLElement}
 */
function segmented(name, attr, options, label) {
  const seg = htmlNode('span', {
    class: 'osprey-seg',
    'data-field': name,
    role: 'group',
    'aria-label': label,
  });
  for (const [value, text, tip] of options) {
    const button = htmlNode(
      'button',
      { type: 'button', [attr]: value, 'aria-pressed': 'false' },
      text
    );
    if (tip) button.title = tip;
    seg.append(button);
  }
  return seg;
}

/**
 * Press the button of a segmented control whose `attr` equals `value`.
 * @param {HTMLElement} seg
 * @param {string} attr
 * @param {string} value
 */
function pressSeg(seg, attr, value) {
  for (const button of seg.querySelectorAll(`[${attr}]`)) {
    button.setAttribute('aria-pressed', String(button.getAttribute(attr) === value));
  }
}

/**
 * The overflow after the rule toggle: a ⋯ button opening a popover with the
 * wider MML corrector lists, a check on the active one.
 * @param {Element} root
 * @returns {HTMLElement}
 */
function ruleMenu(root) {
  const holder = htmlNode('span', { class: 'ob-menu' });
  const id = `ob-rules-${++pickerSeq}`;
  const button = htmlNode('button', {
    type: 'button',
    class: 'osprey-icon-btn',
    'data-field': 'rule-more',
    'aria-controls': id,
    'aria-haspopup': 'menu',
    'aria-expanded': 'false',
    'aria-pressed': 'false',
    'aria-label': 'more corrector rules',
    title: 'more corrector rules',
  });
  button.append(icon('M5 12h.01M12 12h.01M19 12h.01'));
  const menu = htmlNode('div', {
    class: 'osprey-popover',
    id,
    role: 'menu',
    'aria-label': 'more corrector rules',
    'data-field': 'rule-menu',
  });
  menu.hidden = true;
  /**
   * @param {string} attr
   * @param {[string, string, string?][]} options value, label and tooltip
   * @param {string} label the group's accessible name
   */
  const group = (attr, options, label) => {
    const box = htmlNode('div', { role: 'group', 'aria-label': label });
    for (const [value, text, tip] of options) {
      const item = htmlNode(
        'button',
        {
          type: 'button',
          class: 'osprey-popover-item',
          role: 'menuitemradio',
          'aria-checked': 'false',
          [attr]: value,
        },
        text
      );
      if (tip) item.title = tip;
      const check = icon('M5 12.5l4.5 4.5L19 7.5');
      check.classList.add('osprey-popover-check');
      item.append(check);
      box.append(item);
    }
    return box;
  };
  const allCorrectors = htmlNode(
    'button',
    {
      type: 'button',
      class: 'osprey-popover-item',
      role: 'menuitemcheckbox',
      'aria-checked': 'false',
      'data-all-correctors': '',
      title: 'every solved corrector of the plane, whatever the rule',
    },
    'Use all correctors of the plane'
  );
  const allCheck = icon('M5 12.5l4.5 4.5L19 7.5');
  allCheck.classList.add('osprey-popover-check');
  allCorrectors.append(allCheck);
  menu.append(
    group('data-rule', MORE_RULES, 'wider corrector lists'),
    htmlNode('div', { class: 'osprey-popover-sep' }),
    htmlNode('div', { class: 'osprey-popover-title' }, 'Priority'),
    group('data-priority', PRIORITIES, 'priority'),
    htmlNode('div', { class: 'osprey-popover-sep' }),
    allCorrectors
  );
  holder.append(button, menu);
  button.addEventListener('click', () => togglePopover(button, menu));
  menu.addEventListener('click', (event) => {
    const hit =
      event.target instanceof Element
        ? event.target.closest('[data-rule], [data-priority], [data-all-correctors]')
        : null;
    if (!hit) return;
    const rule = /** @type {Rule | null} */ (hit.getAttribute('data-rule'));
    const priority = hit.getAttribute('data-priority');
    if (rule && ALL_RULES.includes(rule)) setSelection(root, { rule, correctors: [] });
    else if (priority === 'closure' || priority === 'targets') setSelection(root, { priority });
    else if (hit.hasAttribute('data-all-correctors')) {
      setSelection(root, { all_correctors: !panelOf(root).selection.all_correctors });
    } else return;
    closePopover(menu);
  });
  return holder;
}

/**
 * Mark the active rule (a toggle button, or the ⋯ button and its checked item),
 * the checked priority and all-correctors items, and the chips naming a non-default one.
 * @param {Element} root
 * @param {Selection} selection
 */
function renderRule(root, selection) {
  const { rule, priority, all_correctors: every } = selection;
  pressSeg(field(root, 'rule'), 'data-rule', rule);
  const wide = MORE_RULES.some(([value]) => value === rule);
  field(root, 'rule-more').setAttribute('aria-pressed', String(wide));
  // The popover sits at <body> level while open, so it is found by id.
  const id = field(root, 'rule-more').getAttribute('aria-controls') ?? '';
  const menu = root.querySelector(`[id="${id}"]`) ?? root.ownerDocument.getElementById(id);
  /** @type {[string, string][]} */
  const marks = [
    ['data-rule', rule],
    ['data-priority', priority],
  ];
  for (const [attr, value] of marks) {
    for (const item of menu?.querySelectorAll(`[${attr}]`) ?? []) {
      item.setAttribute('aria-checked', String(item.getAttribute(attr) === value));
    }
  }
  menu?.querySelector('[data-all-correctors]')?.setAttribute('aria-checked', String(every));
  field(root, 'priority-chip').hidden = priority === 'closure';
  field(root, 'all-chip').hidden = !every;
}

/**
 * The header's options menu: a hamburger that opens a popover with the nudge
 * step, the legend and the gestures. Made on first use at the end of the bar.
 * @param {Element} root
 * @returns {HTMLElement}
 */
function optionsMenu(root) {
  const found = root.querySelector('[data-field="menu"]');
  if (found instanceof HTMLElement) return found;
  const holder = htmlNode('span', { class: 'ob-menu' });
  const button = htmlNode('button', {
    type: 'button',
    class: 'osprey-icon-btn',
    'data-field': 'menu-button',
    'aria-haspopup': 'true',
    'aria-expanded': 'false',
    'aria-label': 'options',
    title: 'Options',
  });
  button.append(icon('M4 7h16M4 12h16M4 17h16'));
  const menu = htmlNode('div', {
    class: 'osprey-popover ob-options',
    id: `ob-options-${++pickerSeq}`,
    role: 'dialog',
    'aria-label': 'options',
    'data-field': 'menu',
  });
  menu.hidden = true;
  const step = segmented(
    'step',
    'data-step',
    STEPS.map((value) => [String(value), String(value)]),
    'nudge step'
  );
  step.classList.add('ob-menu-seg');
  const legend = htmlNode('div', { class: 'ob-legend' });
  for (const [key, text] of [
    ['ob-key-orbit', 'orbit'],
    ['ob-key-golden', 'golden'],
    ['ob-key-target', 'target'],
    ['ob-key-aim', 'aim'],
    ['ob-key-predicted', 'predicted'],
  ]) {
    const entry = htmlNode('span');
    entry.append(htmlNode('i', { class: key }), text);
    legend.append(entry);
  }
  const gestures = htmlNode('div', { class: 'ob-gestures' });
  /** @type {[string[], string][]} */
  const rows = [
    [['drag across dots'], 'add BPMs'],
    [['click', ' · ', '⇧', 'click'], 'toggle · extend'],
    [['drag ◆', ' · ', 'drag grip'], 'set target · move bump'],
    [['↑↓', ' · ', '⇧', ' · ', '⌫'], 'nudge · ×10 · remove'],
    [['⇧', 'drag', ' · wheel'], 'zoom'],
    [['⌥', 'drag'], 'pan'],
    [['dbl-click'], 'whole ring'],
    [['right-click'], 'menu'],
  ];
  const keys = new Set(['⇧', '↑↓', '⌫', '⌥']);
  for (const [parts, what] of rows) {
    const gesture = htmlNode('span', { class: 'ob-gesture' });
    for (const part of parts) {
      gesture.append(keys.has(part) ? htmlNode('span', { class: 'osprey-kbd' }, part) : part);
    }
    gestures.append(gesture, htmlNode('span', {}, what));
  }
  menu.append(
    htmlNode('div', { class: 'osprey-popover-title' }, 'Nudge step'),
    step,
    htmlNode('div', { class: 'osprey-popover-sep' }),
    htmlNode('div', { class: 'osprey-popover-title' }, 'Legend'),
    legend,
    htmlNode('div', { class: 'osprey-popover-sep' }),
    htmlNode('div', { class: 'osprey-popover-title' }, 'Gestures'),
    gestures
  );
  holder.append(button, menu);
  button.addEventListener('click', () => togglePopover(button, menu));
  step.addEventListener('click', (event) => {
    const hit = event.target instanceof Element ? event.target.closest('[data-step]') : null;
    const value = Number(hit?.getAttribute('data-step'));
    if (value > 0) setStep(root, value);
  });
  const bar = root.querySelector('.ob-bar') ?? root;
  bar.append(holder);
  return menu;
}

/**
 * The controls under the strip (Bump card, Correctors card, actions), made
 * with their handlers on first use inside the body field.
 * @param {Element} root
 * @returns {HTMLElement}
 */
function controls(root) {
  const body = field(root, 'body');
  const found = body.querySelector('[data-field="controls"]');
  if (found instanceof HTMLElement) return found;

  const box = htmlNode('div', { class: 'ob-cols', 'data-field': 'controls' });

  const bump = htmlNode('section', { class: 'osprey-card ob-box', 'data-field': 'bump-card' });
  const bumpHead = htmlNode('h3', { class: 'osprey-section-label' });
  const title = htmlNode('span', {}, 'Bump · ');
  title.append(htmlNode('span', { 'data-field': 'bump-count' }, '0'), ' BPM');
  bumpHead.append(title, htmlNode('span', { class: 'ob-spacer' }), picker(root));

  const shape = htmlNode('div', { class: 'ob-shape', 'data-field': 'shape-row' });
  const amp = stepper(
    { 'data-field': 'amp', 'aria-label': 'bump amplitude', step: '0.01', value: '0.000' },
    'amplitude',
    'amp-unit'
  );
  amp.title = 'bump amplitude, applied over the selected BPMs';
  const shapes = segmented(
    'shape',
    'data-shape',
    [
      ['flat', 'flat', 'every bump BPM at the amplitude'],
      ['peak', 'peak', 'a triangle: the amplitude at the centre, zero at the ends'],
      ['cos', 'cosine', 'a raised cosine: the amplitude at the centre, zero at the ends'],
    ],
    'bump shape'
  );
  shapes.title = 'bump shape over the selected BPMs';
  const others = segmented('others', 'data-others', OTHERS, 'other BPMs');
  others.title = 'other BPMs: every BPM not in the bump, inside and outside the corrector span';
  const widenButton = htmlNode(
    'button',
    {
      type: 'button',
      class: 'osprey-btn',
      'data-field': 'widen',
      title: 'widen the bump by one BPM on each side',
    },
    '± 1 BPM'
  );
  shape.append(amp, shapes, htmlNode('span', { class: 'ob-spacer' }), others, widenButton);

  const table = htmlNode('table', { class: 'ob-table', 'data-field': 'bump' });
  const headRow = htmlNode('tr');
  for (const [name, text] of [
    ['bpm', 'BPM'],
    ['s', 's'],
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
    'No BPM in the bump yet — drag across the dots on the strip, or type in the picker.'
  );
  bump.append(bumpHead, shape, table, empty);

  const cors = htmlNode('section', { class: 'osprey-card ob-box', 'data-field': 'cor-card' });
  const corsHead = htmlNode('h3', { class: 'osprey-section-label' });
  const rule = segmented(
    'rule',
    'data-rule',
    RULES.map(([value, text, tip]) => [value, text, tip]),
    'corrector rule'
  );
  const priorityChip = htmlNode('span', {
    class: 'osprey-chip',
    'data-field': 'priority-chip',
    title: 'priority: hit the targets first; the other BPMs may leak',
  });
  priorityChip.append(htmlNode('span', { class: 'osprey-chip-label' }, 'hit targets'));
  priorityChip.hidden = true;
  const allChip = htmlNode('span', {
    class: 'osprey-chip',
    'data-field': 'all-chip',
    title: 'every solved corrector of the plane; the rule is ignored',
  });
  allChip.append(htmlNode('span', { class: 'osprey-chip-label' }, 'all correctors'));
  allChip.hidden = true;
  corsHead.append(
    htmlNode('span', { 'data-field': 'cor-title' }, 'Correctors'),
    htmlNode('span', { class: 'ob-spacer' }),
    priorityChip,
    allChip,
    rule,
    ruleMenu(root)
  );
  const plan = htmlNode('div', { 'data-field': 'card' });
  plan.hidden = true;
  const corEmpty = htmlNode(
    'p',
    { class: 'osprey-list-empty', 'data-field': 'cor-empty' },
    'Compute to see the corrector plan.'
  );
  cors.append(corsHead, plan, corEmpty);

  const actions = htmlNode('div', { class: 'ob-actions', 'data-field': 'actions' });
  const status = htmlNode('span', { class: 'osprey-chip', 'data-field': 'plan-status' });
  status.append(htmlNode('span', { class: 'osprey-chip-label', 'data-field': 'plan-status-text' }));
  status.hidden = true;
  const miss = htmlNode('span', { class: 'osprey-chip osprey-chip--warn', 'data-field': 'miss' });
  miss.append(htmlNode('span', { class: 'osprey-chip-label', 'data-field': 'miss-text' }));
  miss.hidden = true;
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
    htmlNode('span', { class: 'ob-spacer' }),
    miss,
    status
  );

  box.append(bump, cors, actions);

  // Bump table: typed targets, ↑/↓ nudges, −/+ and × buttons, the focused row.
  table.addEventListener('change', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    const element = target.getAttribute('data-target');
    if (!element || target.value.trim() === '') return;
    setTarget(root, element, Number(target.value));
  });
  table.addEventListener('input', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    const element = target.getAttribute('data-target');
    if (!element || target.value.trim() === '') return;
    setTarget(root, element, Number(target.value));
  });
  table.addEventListener('keydown', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    const element = target.getAttribute('data-target');
    if (!element || (event.key !== 'ArrowUp' && event.key !== 'ArrowDown')) return;
    event.preventDefault();
    const big = event.shiftKey ? 10 : 1;
    nudge(root, [element], event.key === 'ArrowUp' ? big : -big);
    target.value = String(panelOf(root).selection.targets[element] ?? target.value);
  });
  table.addEventListener('focusin', (event) => {
    const row = event.target instanceof Element ? event.target.closest('tr') : null;
    for (const each of table.querySelectorAll('tbody tr'))
      each.classList.toggle('ob-focus', each === row);
  });
  table.addEventListener('focusout', (event) => {
    const next = /** @type {FocusEvent} */ (event).relatedTarget;
    if (next instanceof Node && table.contains(next)) return;
    for (const each of table.querySelectorAll('tbody tr')) each.classList.remove('ob-focus');
  });
  table.addEventListener('click', (event) => {
    const hit = event.target instanceof Element ? event.target.closest('button') : null;
    const row = hit?.closest('tr');
    const element = row?.getAttribute('data-element');
    if (!hit || !element) return;
    if (hit.hasAttribute('data-remove')) {
      removeBpms(root, [element]);
      return;
    }
    const dir = Number(hit.getAttribute('data-step-dir'));
    if (dir) {
      const big = /** @type {MouseEvent} */ (event).shiftKey ? 10 : 1;
      nudge(root, [element], dir * big);
      const input = row?.querySelector('input');
      if (input) input.value = String(panelOf(root).selection.targets[element]);
    }
  });

  // Shape row: amplitude stepper, shape, widen.
  const ampInput = /** @type {HTMLInputElement} */ (amp.querySelector('input'));
  ampInput.addEventListener('change', () => {
    if (ampInput.value.trim() !== '') setAmplitude(root, Number(ampInput.value));
  });
  ampInput.addEventListener('keydown', (event) => {
    if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return;
    event.preventDefault();
    const panel = panelOf(root);
    const steps = (event.key === 'ArrowUp' ? 1 : -1) * (event.shiftKey ? 10 : 1);
    setAmplitude(root, panel.amp + steps * panel.step);
    ampInput.value = panel.amp.toFixed(3);
  });
  amp.addEventListener('click', (event) => {
    const hit = event.target instanceof Element ? event.target.closest('[data-step-dir]') : null;
    if (!hit) return;
    const panel = panelOf(root);
    const steps = Number(hit.getAttribute('data-step-dir')) * (event.shiftKey ? 10 : 1);
    setAmplitude(root, panel.amp + steps * panel.step);
  });
  shapes.addEventListener('click', (event) => {
    const hit = event.target instanceof Element ? event.target.closest('[data-shape]') : null;
    const value = hit?.getAttribute('data-shape');
    if (value && SHAPES.includes(/** @type {Shape} */ (value)))
      setShape(root, /** @type {Shape} */ (value));
  });
  widenButton.addEventListener('click', () => widen(root));
  others.addEventListener('click', (event) => {
    const hit = event.target instanceof Element ? event.target.closest('[data-others]') : null;
    const value = /** @type {Others | null} */ (hit?.getAttribute('data-others') ?? null);
    if (value && OTHERS_MODES.includes(value)) setSelection(root, { others: value });
  });

  rule.addEventListener('click', (event) => {
    const hit = event.target instanceof Element ? event.target.closest('[data-rule]') : null;
    const value = /** @type {Rule | null} */ (hit?.getAttribute('data-rule') ?? null);
    if (!value) return;
    /** @type {Partial<Selection>} */
    const changes = { rule: value };
    if (value !== 'manual') changes.correctors = [];
    setSelection(root, changes);
  });

  actions.addEventListener('click', (event) => {
    const hit = event.target instanceof Element ? event.target.closest('button') : null;
    const name = hit?.getAttribute('data-field');
    if (name === 'clear') {
      panelOf(root).amp = 0;
      void clearBump(root);
    } else if (name === 'compute') {
      void compute(root);
    } else if (name === 'apply') {
      openConfirm(root);
    }
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
  const target = htmlNode('td', { 'data-cell': 'target' });
  const box = stepper(
    { step: 'any', 'data-target': element, 'aria-label': `target offset of ${element}` },
    `the target of ${element}`
  );
  box.classList.add('ob-stepper-sm');
  target.append(box);
  const remove = htmlNode('td');
  remove.append(
    htmlNode(
      'button',
      {
        type: 'button',
        class: 'osprey-icon-btn',
        'data-remove': element,
        'aria-label': `remove ${element}`,
        title: 'remove from the bump',
      },
      '×'
    )
  );
  row.append(
    htmlNode('td', { 'data-cell': 'bpm' }, element),
    htmlNode('td', { 'data-cell': 's', class: 'ob-num' }),
    htmlNode('td', { 'data-cell': 'now', class: 'ob-num' }),
    target,
    htmlNode('td', { 'data-cell': 'predicted', class: 'ob-num' }),
    remove
  );
  return row;
}

/**
 * Bring the Bump card (table, picker, shape row) and the corrector rule in
 * line with the state and the selection. Rows are keyed by element: a row and
 * its input survive every poll, and an input is never written while it has focus.
 * @param {Element} root
 * @param {Panel} panel
 */
function renderTable(root, panel) {
  const box = controls(root);
  const { selection, state } = panel;
  const model = selectedModel(panel);
  const bpms = planeBpms(panel);
  const byElement = new Map(bpms.map((member) => [member.element, member]));
  const golden = state?.orbit?.golden ?? {};
  const plan = shownPlan(panel);
  const predicted = plan?.predicted ?? null;
  const missed = new Set(plan?.missed ?? []);

  /** @param {Member | undefined} member */
  const goldOf = (member) => (member?.golden_addr ? golden[member.golden_addr] : null);

  const unit = bpmUnit(panel);
  for (const name of ['now', 'target', 'predicted']) {
    const th = box.querySelector(`th[data-head="${name}"]`);
    if (th) th.textContent = unit ? `${name} ${unit}` : name;
  }

  const elements = bumpOrder(panel);
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
    const s = row.querySelector('[data-cell="s"]');
    if (s) s.textContent = member ? `${member.s_m.toFixed(1)} m` : '—';
    const now = row.querySelector('[data-cell="now"]');
    if (now) now.textContent = cellNumber(orbitOf(panel, member));
    const pred = row.querySelector('[data-cell="predicted"]');
    if (pred) {
      pred.textContent = predicted ? cellNumber(offset(predicted[element], goldOf(member))) : '—';
      pred.classList.toggle('ob-missed', missed.has(element));
    }
    const input = row.querySelector('input');
    const target = selection.targets[element];
    if (input) {
      input.step = String(panel.step);
      if (input !== input.ownerDocument.activeElement) {
        if (input.value === '' || Number(input.value) !== target) input.value = String(target);
      }
    }
  });
  field(root, 'bump-count').textContent = String(elements.length);
  field(root, 'bump-empty').hidden = elements.length > 0;
  field(root, 'bump').hidden = elements.length === 0;

  const amp = /** @type {HTMLInputElement} */ (field(root, 'amp'));
  amp.step = String(panel.step);
  if (amp !== amp.ownerDocument.activeElement) amp.value = panel.amp.toFixed(3);
  field(root, 'amp-unit').textContent = unit;
  pressSeg(field(root, 'shape'), 'data-shape', panel.shape);
  pressSeg(field(root, 'others'), 'data-others', selection.others);
  /** @type {HTMLButtonElement} */ (field(root, 'widen')).disabled = elements.length === 0;
  field(root, 'pick-list').parentElement?.dispatchEvent(new Event('ob-refresh'));

  const step = field(root, 'step');
  for (const button of step.querySelectorAll('[data-step]')) {
    button.textContent = stepLabel(Number(button.getAttribute('data-step')), unit);
  }
  pressSeg(step, 'data-step', String(panel.step));

  renderRule(root, selection);

  const title = field(root, 'cor-title');
  const planeCorrectors = new Set(
    (model?.members.correctors ?? [])
      .filter((member) => member.plane === selection.plane)
      .map((member) => member.element)
  );
  const unsolved = (model?.orm.unsolved ?? []).filter((element) => planeCorrectors.has(element));
  const tip = plan
    ? planSummary(plan, unit)
    : unsolved.length > 0
      ? `not selectable, the ORM could not solve them: ${unsolved.join(', ')}`
      : '';
  if (tip) title.title = tip;
  else title.removeAttribute('title');
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
  panel.anchor = null;
  panel.amp = 0;
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
  panel.bound = true;
  field(root, 'plane').addEventListener('click', (event) => {
    const hit = event.target instanceof Element ? event.target.closest('[data-plane]') : null;
    const plane = hit?.getAttribute('data-plane');
    if (plane === 'x' || plane === 'y') void choosePlane(root, plane);
  });
  const select = /** @type {HTMLSelectElement} */ (field(root, 'model'));
  select.addEventListener('change', () => void chooseModel(root, select.value));
  field(root, 'recompute').addEventListener('click', () => void recompute(root, panel.fetcher));
  optionsMenu(root);
  controls(root);
}

/**
 * Render everything from the state and the selection.
 * @param {Element} root
 */
export function render(root) {
  const panel = panelOf(root);
  optionsMenu(root);
  renderHeader(root, panel);
  renderStrip(root, panel);
  renderTable(root, panel);
  renderCard(root, panel);
  publishDraft(root);
}

/**
 * What the operator has set, computed or not, as the backend keeps it for the agent.
 * @param {Panel} panel
 */
export function draftOf(panel) {
  const { selection } = panel;
  return {
    plane: selection.plane,
    model: selection.model,
    targets: { ...selection.targets },
    rule: selection.rule,
    correctors: selection.rule === 'manual' ? [...selection.correctors] : [],
    priority: selection.priority,
    others: selection.others,
    all_correctors: selection.all_correctors,
    shape: panel.shape,
    amplitude: panel.amp,
  };
}

/**
 * Publish the draft once the operator pauses, so the agent sees what the operator
 * sees. Only a change is sent. A failure is dropped and retried on the next change
 * or poll, except a 403: a login that may not post keeps its draft to itself.
 * @param {Element} root
 */
function publishDraft(root) {
  const panel = panelOf(root);
  if (!panel.bound || panel.readOnly || !panel.state) return;
  const body = JSON.stringify(draftOf(panel));
  if (body === panel.draftSent || body === panel.draftPending) return;
  if (panel.draftTimer != null) clearTimeout(panel.draftTimer);
  panel.draftPending = body;
  panel.draftTimer = setTimeout(async () => {
    panel.draftTimer = null;
    panel.draftPending = null;
    panel.draftSent = body;
    try {
      const response = await panel.fetcher(DRAFT_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
      });
      if (!response.ok && response.status !== 403) panel.draftSent = null;
    } catch {
      panel.draftSent = null;
    }
  }, DRAFT_DEBOUNCE_MS);
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
 * What a plan misses: "targets missed · up to 0.110 mm", or '' when it meets them.
 * @param {Plan} plan
 * @param {string} unit
 * @returns {string}
 */
function missText(plan, unit) {
  if (!plan.missed?.length) return '';
  const suffix = unit ? ` ${unit}` : '';
  return `targets missed · up to ${(plan.target_miss_max ?? 0).toFixed(3)}${suffix}`;
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
  if (plan.bounded?.length) parts.push(`at limit: ${plan.bounded.join(', ')}`);
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
  const shown = columns.filter(([name]) =>
    inBox ? name !== 'channel' && name !== 'note' : name !== 'outcome'
  );
  const headRow = htmlNode('tr');
  for (const [name, text] of shown) headRow.append(htmlNode('th', { 'data-head': name }, text));
  const head = htmlNode('thead');
  head.append(headRow);
  const body = htmlNode('tbody');
  for (const row of plan.correctors) {
    const tr = htmlNode('tr', { 'data-element': row.element });
    tr.append(...shown.map(([, , make]) => make(row)));
    const note = rowNote(row);
    if (inBox && note) tr.title = note;
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
  if (plan.status === 'ready') return `plan #${plan.id}`;
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

  const stale = plan != null && plan.status === 'ready' && edited(panel, plan);
  const phrase = plan && !panel.readOnly && !allowed ? (panel.state?.posture?.phrase ?? '') : '';
  const applyButton = /** @type {HTMLButtonElement} */ (field(root, 'apply'));
  applyButton.hidden = panel.readOnly || !plan;
  applyButton.textContent = plan ? `Apply ${setpoints(plan.correctors.length)}…` : 'Apply…';
  applyButton.disabled = !plan || plan.status !== 'ready' || !allowed || stale || panel.applying;
  const why = phrase || (stale && plan ? `edited since plan #${plan.id}: Compute again` : '');
  if (why) applyButton.title = why;
  else applyButton.removeAttribute('title');

  // The one status indicator: hidden with nothing to say.
  const status = field(root, 'plan-status');
  const hasTargets = Object.keys(panel.selection.targets).length > 0;
  let text = '';
  if (panel.computesInFlight > 0) text = 'computing…';
  else if (panel.applying) text = 'applying…';
  else if (plan && plan.status !== 'ready') text = statusText(plan);
  else if (stale || (!plan && hasTargets)) text = 'edited · not computed';
  else if (plan) text = statusText(plan);
  field(root, 'plan-status-text').textContent = text;
  status.hidden = text === '';
  status.classList.toggle('osprey-chip--connecting', text === 'edited · not computed');

  const miss = field(root, 'miss');
  const missing = plan ? missText(plan, bpmUnit(panel)) : '';
  field(root, 'miss-text').textContent = missing;
  miss.hidden = missing === '';
  if (missing && plan) miss.title = `missed: ${(plan.missed ?? []).join(', ')}`;
  else miss.removeAttribute('title');

  field(root, 'cor-empty').hidden = plan != null;
  const node = card(root);
  if (!plan) {
    node.hidden = true;
    node.replaceChildren();
    return;
  }
  node.hidden = false;
  const children = [];
  if (plan.status === 'not_applicable' && plan.reason) {
    children.push(htmlNode('p', { class: 'ob-refused', 'data-field': 'plan-reason' }, plan.reason));
  }
  children.push(rowsTable(plan, true));
  node.replaceChildren(...children);
}

/**
 * The Correctors title's tooltip for a plan: its id, the closure line and the
 * residual after an apply.
 * @param {Plan} plan
 * @param {string} unit
 * @returns {string}
 */
function planSummary(plan, unit) {
  const parts = [`plan #${plan.id}`];
  const closure = closureText(plan, unit);
  if (closure) parts.push(closure);
  const residual = Object.entries(plan.residual_after ?? {});
  if (residual.length > 0) {
    const label = unit ? `residual after (${unit})` : 'residual after';
    parts.push(`${label}: ${residual.map(([e, v]) => `${e} ${cellNumber(v)}`).join(', ')}`);
  }
  return parts.join(' · ');
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
    priority: selection.priority,
    others: selection.others,
    all_correctors: selection.all_correctors,
  };
  if (selection.rule === 'manual') body.correctors = [...selection.correctors];
  const key = selectionKey(selection);
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
      if (plan.id != null) panel.computed = { id: plan.id, key };
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
  if (edited(panel, plan)) return;
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
    rowsTable(plan, false)
  );
  const missing = missText(plan, bpmUnit(panel));
  if (missing) {
    box.append(
      htmlNode(
        'p',
        { class: 'ob-refused', 'data-field': 'confirm-miss' },
        `${missing} at ${(plan.missed ?? []).join(', ')}`
      )
    );
  }
  box.append(htmlNode('p', { class: 'ob-meta' }, closureText(plan, bpmUnit(panel))), buttons);
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
  initTooltips(document.body);
  bindControls(root);
  void refresh(root);
  setInterval(() => void refresh(root), REFRESH_MS);
}
