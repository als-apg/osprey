// @ts-check
/*
 * OKF Knowledge Panel — sidebar tree: render, highlight, and highlight state.
 *
 * Boot wiring only: `initTree` stores the three injected handles once, at
 * boot. Every other export reads those module-private handles rather than
 * taking them as parameters, matching the closure shape this module was
 * extracted from in app.js.
 */

import { el } from "./helpers.js";

/** @typedef {{id: string, title?: string, description?: string}} Concept */
/** @typedef {{id?: string, label?: string, concepts?: Concept[]}} Group */

/** @type {HTMLElement} */
let treeEl;
/** @type {HTMLElement | null} */
let structureLink = null;
/** @type {(id: string) => void} */
let onSelect;
/** @type {string | null} */
let activeConceptId = null;

/**
 * Store the injected DOM handles and selection callback. Called once at
 * boot, before any other export in this module is used.
 *
 * @param {{treeEl: HTMLElement, structureLink: HTMLElement | null, onSelect: (id: string) => void}} handles
 */
export function initTree(handles) {
  treeEl = handles.treeEl;
  structureLink = handles.structureLink;
  onSelect = handles.onSelect;
}

/**
 * @param {Group[]} groups
 */
export function renderTree(groups) {
  treeEl.innerHTML = "";
  for (const group of groups) {
    const section = el("section", { class: "group" });
    section.appendChild(
      el("h2", { class: "group-label", text: group.label || group.id || "" })
    );

    const concepts = group.concepts || [];
    if (concepts.length === 0) {
      section.appendChild(
        el("p", { class: "muted empty-group", text: "(no concepts)" })
      );
    } else {
      const list = el("ul", { class: "concept-list" });
      for (const concept of concepts) {
        const link = el("a", {
          class: "concept-link",
          href: "#",
          text: concept.title || concept.id,
          title: concept.description || "",
        });
        link.dataset.conceptId = concept.id;
        link.addEventListener("click", function (ev) {
          ev.preventDefault();
          selectConcept(concept.id);
        });
        const li = el("li", null, [link]);
        li.dataset.conceptId = concept.id;
        list.appendChild(li);
      }
      section.appendChild(list);
    }
    treeEl.appendChild(section);
  }
  if (activeConceptId != null) applyConceptHighlight(activeConceptId);
}

// Shared body for the concept-link `.active` walk + scroll-into-view.
/**
 * @param {string} id
 */
function applyConceptHighlight(id) {
  const links = /** @type {NodeListOf<HTMLElement>} */ (
    treeEl.querySelectorAll(".concept-link")
  );
  let activeLink = /** @type {HTMLElement | null} */ (null);
  links.forEach(function (link) {
    if (link.dataset.conceptId === id) {
      link.classList.add("active");
      activeLink = link;
    } else {
      link.classList.remove("active");
    }
  });
  // Scroll the active entry into view so the user can always see where they
  // are in the tree — important when jumping via an in-body cross-link to a
  // concept far down the (scrolled) sidebar.
  if (activeLink) activeLink.scrollIntoView({ block: "nearest", inline: "nearest" });
}

/**
 * @param {string} id
 */
export function highlightActive(id) {
  activeConceptId = id;
  if (structureLink) structureLink.classList.remove("active");
  applyConceptHighlight(id);
}

// Mark the structure-overview entry active and clear any concept highlight.
export function highlightStructure() {
  activeConceptId = null;
  const links = treeEl.querySelectorAll(".concept-link");
  links.forEach(function (link) {
    link.classList.remove("active");
  });
  if (structureLink) structureLink.classList.add("active");
}

/**
 * @param {string} id
 */
export function selectConcept(id) {
  // The sidebar highlight is applied authoritatively in renderConcept (which
  // every navigation path funnels through), so just trigger the load here.
  onSelect(id);
}
