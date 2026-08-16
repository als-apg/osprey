// @ts-check
/*
 * OKF Knowledge Panel — search results: render and clear.
 *
 * Boot wiring only: `initSearchResults` stores the two injected handles once,
 * at boot; the render exports read those module-private handles rather than
 * taking them as parameters, matching `tree.js`.
 *
 * RANKED VS UNRANKED
 * ------------------
 * `/api/search` returns a `score` per hit: a number when the qmd sidecar
 * ranked the results (they arrive best-first) and `null` when the substring
 * fallback produced them, which does not rank. Both are normal states — a
 * deployment with no sidecar is a supported configuration, not a degradation —
 * so the unranked list renders exactly as it always has, with no apology and
 * no empty slot where a rank would go.
 *
 * The score itself is deliberately NOT displayed. It is rank-derived (qmd
 * hands back 1, 0.5, 0.33 … for the first three hits) and carries nothing the
 * ordering does not already say; rendered as "33%" beside a document an
 * operator is deciding whether to trust, it would read as a calibrated
 * confidence the number does not have. What the score genuinely licenses is
 * the claim that position means something — so ranked results get their rank
 * shown and a caption saying so, and that is all.
 */

import { el } from "./helpers.js";

/** @typedef {{id: string, title?: string, snippet?: string, score?: number|null}} SearchResultItem */

/** @type {HTMLElement} */
let containerEl;
/** @type {(id: string) => void} */
let onSelect;

/**
 * Store the injected DOM handle and selection callback. Called once at boot,
 * before any other export in this module is used.
 *
 * @param {{containerEl: HTMLElement, onSelect: (id: string) => void}} handles
 */
export function initSearchResults(handles) {
  containerEl = handles.containerEl;
  onSelect = handles.onSelect;
}

/**
 * Hide the results panel and drop its contents.
 */
export function clearSearchResults() {
  containerEl.hidden = true;
  containerEl.innerHTML = "";
}

/**
 * Replace the results panel with `message` as a muted line.
 *
 * @param {string} message
 */
export function renderSearchMessage(message) {
  containerEl.hidden = false;
  containerEl.innerHTML = "";
  containerEl.appendChild(el("p", { class: "muted", text: message }));
}

/**
 * True when the backend ranked these results, i.e. at least one hit carries a
 * numeric score. A mixed list cannot occur — one backend answers a query — so
 * one scored hit settles it for the whole list.
 *
 * @param {SearchResultItem[]} results
 * @returns {boolean}
 */
export function isRanked(results) {
  return results.some((r) => typeof r.score === "number");
}

/**
 * Render `/api/search` hits into the results panel.
 *
 * Ranked hits go into an `<ol>` carrying an explicit rank per item and a
 * caption naming the ordering; unranked hits keep the plain `<ul>` of
 * link-plus-snippet cards. Both share the same item markup, so the sidebar
 * looks like one component in either mode.
 *
 * @param {SearchResultItem[]} results
 */
export function renderSearchResults(results) {
  containerEl.hidden = false;
  containerEl.innerHTML = "";

  if (results.length === 0) {
    containerEl.appendChild(el("p", { class: "muted", text: "No matches." }));
    return;
  }

  const ranked = isRanked(results);
  if (ranked) {
    containerEl.appendChild(
      el("p", { class: "muted result-mode", text: "Ranked by relevance" })
    );
  }

  const list = el(ranked ? "ol" : "ul", {
    class: ranked ? "result-list result-list-ranked" : "result-list",
  });

  results.forEach(function (r, index) {
    const item = el("li", { class: "result-item" });
    if (ranked) {
      // aria-hidden: the <ol> already gives assistive tech the position, so an
      // exposed rank would be announced twice ("item 1 of 3", then "1"). The
      // span is a visual repeat of that position, nothing more.
      item.appendChild(
        el("span", {
          class: "result-rank",
          text: String(index + 1),
          "aria-hidden": "true",
        })
      );
    }
    const link = el("a", {
      class: "result-link",
      href: "#",
      text: r.title || r.id,
    });
    link.dataset.conceptId = r.id;
    link.addEventListener("click", function (ev) {
      ev.preventDefault();
      onSelect(r.id);
    });
    item.appendChild(link);
    if (r.snippet) {
      item.appendChild(el("p", { class: "result-snippet", text: String(r.snippet) }));
    }
    list.appendChild(item);
  });

  containerEl.appendChild(list);
}
