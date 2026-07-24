// @ts-check
/* OSPREY Web Terminal — Empty Workspace Placeholder
 *
 * The "no active panel" card panel-manager renders into the workspace content
 * host when nothing can hold the slot (every panel hidden or unhealthy). Pure
 * DOM-in, DOM-out — no panel state — so it lives outside panel-manager.
 */

/**
 * Render the placeholder card into the workspace content host, replacing the
 * host's children. Also clears the container's `data-active-panel` stamp so
 * the default padded card chrome returns (a previous canvas-painting panel
 * may have opted out of it — see panel-manager's activateTab).
 * @param {HTMLElement | null} containerEl
 * @param {HTMLElement | null} contentEl
 * @param {string} message
 */
export function renderEmptyState(containerEl, contentEl, message) {
  if (!contentEl) return;
  if (containerEl) delete containerEl.dataset.activePanel;
  contentEl.innerHTML = `
    <div class="artifacts-empty-state">
      <div class="artifacts-empty-icon">
        <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="1">
          <path d="M12 2L2 7l10 5 10-5-10-5z" transform="translate(12 8) scale(1.2)"/>
          <path d="M2 17l10 5 10-5" transform="translate(12 8) scale(1.2)"/>
          <path d="M2 12l10 5 10-5" transform="translate(12 8) scale(1.2)"/>
        </svg>
      </div>
      <div class="artifacts-empty-title">Services</div>
      <div class="artifacts-empty-text"></div>
    </div>
  `;
  // message goes in via textContent — the one dynamic value never rides the
  // innerHTML sink (current callers pass literals; this keeps it safe if one
  // ever doesn't).
  const text = contentEl.querySelector('.artifacts-empty-text');
  if (text) text.textContent = message;
}
