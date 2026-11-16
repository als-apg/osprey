// @ts-check
/**
 * The facts strip: a band of short facts, each a caption and a value (the
 * design system's status strip). The run page shows the run's figures in it;
 * any view may use it for a few figures side by side.
 *
 * A fact whose value is null or undefined is left out: the strip never shows
 * a placeholder for a value the service did not supply. {@link updateFacts}
 * writes new values in place, so a strip that changes every evaluation is not
 * rebuilt under the cursor.
 *
 * @module facts-strip
 */

import { h } from './hyperscript.js';

/**
 * @typedef {{
 *   key: string,
 *   label: string,
 *   value: string|number|Node|null|undefined,
 *   accent?: boolean,
 *   tip?: string|null,
 * }} Fact
 */

/**
 * The facts that have a value.
 *
 * @param {Fact[]} facts
 * @returns {Fact[]}
 */
export function shownFacts(facts) {
  return facts.filter((f) => f.value !== null && f.value !== undefined && f.value !== '');
}

/** @param {Fact} fact */
function factNode(fact) {
  const value = fact.value instanceof Node ? fact.value : String(fact.value);
  return h(
    'span',
    {
      class: `osprey-fact${fact.accent ? ' accent' : ''}`,
      'data-fact': fact.key,
      title: fact.tip || null,
    },
    h('span', { class: 'osprey-kicker', text: fact.label }),
    h('span', { class: 'osprey-fact-value' }, value)
  );
}

/**
 * A facts strip.
 *
 * @param {Fact[]} facts
 * @param {{ label?: string }} [options]  `label` names the strip for assistive tech
 * @returns {HTMLElement}
 */
export function factsStrip(facts, { label = '' } = {}) {
  return h(
    'div',
    { class: 'osprey-status-strip opt-facts', role: 'group', 'aria-label': label || null },
    ...shownFacts(facts).map(factNode)
  );
}

/**
 * Show `facts` in a drawn strip. Values change in place; a fact that comes or
 * goes redraws the strip's children.
 *
 * @param {HTMLElement} strip  a {@link factsStrip}
 * @param {Fact[]} facts
 */
export function updateFacts(strip, facts) {
  const shown = shownFacts(facts);
  const nodes = /** @type {HTMLElement[]} */ (Array.from(strip.children));
  const same =
    nodes.length === shown.length && nodes.every((n, i) => n.dataset.fact === shown[i].key);
  if (!same) {
    strip.replaceChildren(...shown.map(factNode));
    return;
  }
  shown.forEach((fact, i) => {
    const value = nodes[i].querySelector('.osprey-fact-value');
    if (!value) return;
    if (fact.value instanceof Node) value.replaceChildren(fact.value);
    else if (value.textContent !== String(fact.value)) value.textContent = String(fact.value);
    nodes[i].classList.toggle('accent', !!fact.accent);
  });
}
