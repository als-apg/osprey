/* OSPREY Logbook Entry Composer
 *
 * Creates a modal UI for composing and submitting ARIEL logbook entries
 * from artifacts and data context items in the gallery.
 *
 * Depends on: gallery.js (for window._galleryState)
 */

(function () {
  "use strict";

  // Pencil icon SVG for the logbook button
  const PENCIL_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>';

  let modal = null;

  // ---- Modal DOM ----

  function createLogbookModal() {
    if (modal) return modal;

    const overlay = document.createElement("div");
    overlay.className = "logbook-overlay";
    overlay.style.display = "none";
    overlay.innerHTML = `
      <div class="logbook-modal">
        <div class="logbook-modal-header">
          <h3>Compose Logbook Entry</h3>
          <button class="logbook-modal-close" title="Close">&times;</button>
        </div>
        <div class="logbook-modal-body" id="logbook-body">
          <div class="logbook-spinner" id="logbook-spinner">Composing entry&hellip;</div>
          <div id="logbook-form" style="display:none">
            <div class="logbook-field">
              <label for="logbook-subject">Subject</label>
              <input type="text" class="logbook-input" id="logbook-subject" maxlength="120">
            </div>
            <div class="logbook-field">
              <label for="logbook-details">Details</label>
              <textarea class="logbook-textarea" id="logbook-details" rows="6"></textarea>
            </div>
            <div class="logbook-field">
              <label for="logbook-tags">Tags <span style="color:var(--text-muted)">(comma-separated)</span></label>
              <input type="text" class="logbook-tags-input" id="logbook-tags">
            </div>
          </div>
          <div class="logbook-error" id="logbook-error" style="display:none"></div>
        </div>
        <div class="logbook-modal-footer">
          <button class="logbook-btn logbook-btn-cancel" id="logbook-cancel">Cancel</button>
          <button class="logbook-btn logbook-btn-submit" id="logbook-submit" disabled>Submit as Draft</button>
        </div>
      </div>
    `;

    document.body.appendChild(overlay);
    modal = overlay;

    // Event listeners
    overlay.querySelector(".logbook-modal-close").addEventListener("click", hideModal);
    overlay.querySelector("#logbook-cancel").addEventListener("click", hideModal);
    overlay.querySelector("#logbook-submit").addEventListener("click", submitLogbook);
    overlay.addEventListener("click", function (e) {
      if (e.target === overlay) hideModal();
    });

    return modal;
  }

  function showModal() {
    const m = createLogbookModal();
    m.style.display = "";
  }

  function hideModal() {
    if (modal) modal.style.display = "none";
  }

  function showSpinner() {
    const spinner = document.getElementById("logbook-spinner");
    const form = document.getElementById("logbook-form");
    const error = document.getElementById("logbook-error");
    const submit = document.getElementById("logbook-submit");
    if (spinner) spinner.style.display = "";
    if (form) form.style.display = "none";
    if (error) error.style.display = "none";
    if (submit) submit.disabled = true;
  }

  function showForm(data) {
    const spinner = document.getElementById("logbook-spinner");
    const form = document.getElementById("logbook-form");
    const submit = document.getElementById("logbook-submit");
    if (spinner) spinner.style.display = "none";
    if (form) form.style.display = "";
    if (submit) submit.disabled = false;

    document.getElementById("logbook-subject").value = data.subject || "";
    document.getElementById("logbook-details").value = data.details || "";
    document.getElementById("logbook-tags").value = (data.tags || []).join(", ");

    // Stash artifact_ids for submit
    form.dataset.artifactIds = JSON.stringify(data.artifact_ids || []);
  }

  function showError(msg) {
    const spinner = document.getElementById("logbook-spinner");
    const error = document.getElementById("logbook-error");
    if (spinner) spinner.style.display = "none";
    if (error) {
      error.textContent = msg;
      error.style.display = "";
    }
  }

  // ---- API calls ----

  async function openComposeModal(opts) {
    showModal();
    showSpinner();

    const body = {};
    if (opts.artifact_id) body.artifact_id = opts.artifact_id;
    if (opts.context_id != null) body.context_id = opts.context_id;

    try {
      const resp = await fetch("/api/logbook/compose", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        showError(err.detail || `Compose failed (${resp.status})`);
        return;
      }
      const data = await resp.json();
      showForm(data);
    } catch (e) {
      showError("Network error: " + e.message);
    }
  }

  async function submitLogbook() {
    const subject = document.getElementById("logbook-subject").value.trim();
    const details = document.getElementById("logbook-details").value.trim();
    const tagsStr = document.getElementById("logbook-tags").value;
    const tags = tagsStr ? tagsStr.split(",").map(t => t.trim()).filter(Boolean) : [];
    const form = document.getElementById("logbook-form");
    const artifactIds = form ? JSON.parse(form.dataset.artifactIds || "[]") : [];

    if (!subject || !details) {
      showError("Subject and details are required.");
      return;
    }

    const submitBtn = document.getElementById("logbook-submit");
    if (submitBtn) submitBtn.disabled = true;

    try {
      const resp = await fetch("/api/logbook/submit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          subject: subject,
          details: details,
          tags: tags,
          artifact_ids: artifactIds,
        }),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        showError(err.detail || `Submit failed (${resp.status})`);
        if (submitBtn) submitBtn.disabled = false;
        return;
      }
      const data = await resp.json();
      // Success — show brief confirmation then close
      const body = document.getElementById("logbook-body");
      if (body) {
        body.innerHTML = `
          <div style="text-align:center; padding:var(--space-6); color:var(--color-success);">
            <div style="font-size:var(--text-xl); margin-bottom:var(--space-2);">Draft created</div>
            <div style="font-size:var(--text-sm); color:var(--text-secondary);">
              ${data.draft_id}<br>
              <a href="${data.url}" target="_blank" rel="noopener"
                 style="color:var(--color-accent-light);">Open in ARIEL</a>
            </div>
          </div>
        `;
      }
      if (submitBtn) submitBtn.style.display = "none";
      // Auto-close after 3s
      setTimeout(hideModal, 3000);
    } catch (e) {
      showError("Network error: " + e.message);
      if (submitBtn) submitBtn.disabled = false;
    }
  }

  // ---- Button injection ----

  function createLogbookBtn(opts) {
    const btn = document.createElement("button");
    btn.className = "logbook-action-btn";
    btn.title = "Compose logbook entry";
    btn.innerHTML = PENCIL_ICON + " Logbook";
    btn.addEventListener("click", function (e) {
      e.stopPropagation();
      openComposeModal(opts);
    });
    return btn;
  }

  /**
   * Inject logbook buttons into focus and preview action bars.
   * Called after each render pass by gallery.js.
   */
  function injectLogbookButtons() {
    const gs = window._galleryState;
    if (!gs) return;

    // Remove any previously injected buttons to avoid duplicates
    document.querySelectorAll(".logbook-action-btn").forEach(b => b.remove());

    // Artifact Focus — append to the nav bar in focus-footer
    const focusedArtifact = gs.getFocusedArtifact && gs.getFocusedArtifact();
    if (focusedArtifact) {
      const bar = document.querySelector("#focus-container .focus-nav");
      if (bar) {
        bar.appendChild(createLogbookBtn({ artifact_id: focusedArtifact.id }));
      }
    }

    // Artifact Preview — append to the header actions bar
    const selectedArtifact = gs.getSelectedArtifact && gs.getSelectedArtifact();
    if (selectedArtifact) {
      const bar = document.querySelector("#preview-content .preview-header-actions");
      if (bar) {
        bar.appendChild(createLogbookBtn({ artifact_id: selectedArtifact.id }));
      }
    }

    // Context Focus — append to the nav bar in focus-footer
    const focusedContext = gs.getFocusedContext && gs.getFocusedContext();
    if (focusedContext) {
      const bar = document.querySelector("#ctx-focus-container .focus-nav");
      if (bar) {
        bar.appendChild(createLogbookBtn({ context_id: focusedContext.id }));
      }
    }

    // Context Preview — append to the header actions bar
    const selectedContext = gs.getSelectedContext && gs.getSelectedContext();
    if (selectedContext) {
      const bar = document.querySelector("#ctx-preview-content .preview-header-actions");
      if (bar) {
        bar.appendChild(createLogbookBtn({ context_id: selectedContext.id }));
      }
    }
  }

  // Expose for gallery.js to call
  window.injectLogbookButtons = injectLogbookButtons;
})();
