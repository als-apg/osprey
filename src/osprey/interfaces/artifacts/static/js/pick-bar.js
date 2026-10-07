// @ts-check
/**
 * OSPREY Artifact Gallery — the pick bar.
 *
 * A slim bar above the sidebar list, shown while two or more rows are picked:
 * "<n> selected · Delete · Clear". Delete removes the picked artifacts in one
 * request after one confirm. Pinned artifacts in a pick are kept, and the
 * confirm says how many stay. The preview pane is not this bar's concern: it
 * keeps showing the last clicked artifact.
 *
 * `createPickBar` takes every effect injected — fetch, confirm, the state.js
 * accessors and the re-render to run after a delete — so the bar holds no
 * state of its own; `mountPickBar` binds it to the page's elements and to
 * state.js.
 *
 * @module pick-bar
 */

import {
  getArtifacts,
  removeArtifact,
  getSelectedArtifact,
  setSelectedArtifact,
  getFocusedArtifact,
  setFocusedArtifact,
  getPickedIds,
  clearPicked,
} from "./state.js";

/**
 * @typedef {object} PickBarElements
 * @property {HTMLElement} bar
 * @property {HTMLElement} count
 * @property {HTMLButtonElement} deleteBtn
 * @property {HTMLButtonElement} clearBtn
 */

/**
 * @typedef {object} PickBarDeps
 * @property {(url: string, init: RequestInit) => Promise<any>} fetch
 * @property {(message: string) => boolean} confirm
 * @property {() => Set<string>} getPickedIds
 * @property {() => void} clearPicked
 * @property {() => any[]} getArtifacts
 * @property {(id: string) => void} removeArtifact
 * @property {() => any|null} getSelectedArtifact
 * @property {(a: any|null) => void} setSelectedArtifact
 * @property {() => any|null} getFocusedArtifact
 * @property {(a: any|null) => void} setFocusedArtifact
 * @property {() => void} onDeleted - runs once after a successful delete, to re-render the views
 * @property {() => void} [onCleared] - runs after Clear, to re-render the sidebar rows
 * @property {(...args: any[]) => void} [log] - where a failed delete is reported
 */

/**
 * Split the picks into the ids a delete sends and the count of pinned rows it keeps.
 * @param {PickBarDeps} deps
 * @returns {{picked: number, doomed: string[], pinned: number}}
 */
function splitPicks(deps) {
  const picked = deps.getPickedIds();
  /** @type {Map<string, any>} */
  const byId = new Map(deps.getArtifacts().map((a) => [a.id, a]));
  /** @type {string[]} */
  const doomed = [];
  let pinned = 0;
  picked.forEach((id) => {
    const a = byId.get(id);
    if (!a) return;
    if (a.pinned) pinned += 1;
    else doomed.push(id);
  });
  return { picked: picked.size, doomed, pinned };
}

/**
 * @param {number} n
 * @param {number} pinned
 * @returns {string}
 */
function confirmText(n, pinned) {
  const what = `Delete ${n} artifact${n === 1 ? "" : "s"}?`;
  const kept = pinned ? ` ${pinned} pinned ${pinned === 1 ? "stays" : "stay"}.` : "";
  return `${what}${kept} This cannot be undone.`;
}

/**
 * @param {PickBarElements} els
 * @param {PickBarDeps} deps
 * @returns {{render: () => void, clear: () => boolean}}
 */
export function createPickBar(els, deps) {
  const log = deps.log ?? ((...args) => console.error(...args));

  function render() {
    const { picked, doomed } = splitPicks(deps);
    els.bar.hidden = picked < 2;
    els.count.textContent = `${picked} selected`;
    const allPinned = picked > 0 && doomed.length === 0;
    els.deleteBtn.disabled = allPinned;
    els.deleteBtn.title = allPinned ? "Pinned artifacts are kept" : "";
  }

  function deletePicked() {
    const { doomed, pinned } = splitPicks(deps);
    if (doomed.length === 0) return;
    if (!deps.confirm(confirmText(doomed.length, pinned))) return;
    deps
      .fetch("/api/artifacts/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ids: doomed }),
      })
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then((/** @type {{deleted: string[]}} */ data) => {
        for (const id of data.deleted || []) {
          deps.removeArtifact(id);
          if (deps.getSelectedArtifact()?.id === id) deps.setSelectedArtifact(null);
          if (deps.getFocusedArtifact()?.id === id) deps.setFocusedArtifact(null);
        }
        deps.clearPicked();
        render();
        deps.onDeleted();
      })
      .catch((err) => log("Delete failed:", err));
  }

  /**
   * Empty the picks and re-render.
   * @returns {boolean} whether any row was picked
   */
  function clear() {
    if (deps.getPickedIds().size === 0) return false;
    deps.clearPicked();
    render();
    deps.onCleared?.();
    return true;
  }

  els.deleteBtn.addEventListener("click", deletePicked);
  els.clearBtn.addEventListener("click", clear);

  return { render, clear };
}

/**
 * Bind the pick bar to the gallery page's `#pick-bar` elements and to state.js.
 * @param {{onDeleted: () => void, onCleared: () => void}} views - the re-renders after a delete and after Clear
 * @returns {{render: () => void, clear: () => boolean}}
 */
export function mountPickBar({ onDeleted, onCleared }) {
  return createPickBar(
    {
      bar: /** @type {HTMLElement} */ (document.getElementById("pick-bar")),
      count: /** @type {HTMLElement} */ (document.getElementById("pick-count")),
      deleteBtn: /** @type {HTMLButtonElement} */ (document.getElementById("pick-delete")),
      clearBtn: /** @type {HTMLButtonElement} */ (document.getElementById("pick-clear")),
    },
    {
      fetch: (url, init) => fetch(url, init),
      confirm: (message) => window.confirm(message),
      getPickedIds,
      clearPicked,
      getArtifacts,
      removeArtifact,
      getSelectedArtifact,
      setSelectedArtifact,
      getFocusedArtifact,
      setFocusedArtifact,
      onDeleted,
      onCleared,
    },
  );
}
