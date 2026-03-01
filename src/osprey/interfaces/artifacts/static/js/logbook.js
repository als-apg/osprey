/* OSPREY Logbook Entry Composer
 *
 * Creates a modal UI for composing and submitting ARIEL logbook entries
 * from artifacts in the gallery.
 *
 * Modal phases:
 *   P2a — Steering Panel (purpose, detail, nudge, context, model)
 *   P2b — Prompt Preview (read-only)
 *   P2c — Prompt Editor (editable textarea)
 *   P2d — Composing (spinner)
 *   P3  — Review Form (subject, details, tags → submit)
 *
 * Depends on: gallery.js (for window._galleryState)
 */

(function () {
  "use strict";

  // Pencil icon SVG for the logbook button
  const PENCIL_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>';

  let modal = null;
  let currentPhase = "steering";
  let currentOpts = {};          // {artifact_id}
  let allArtifacts = [];         // cached from /api/artifacts for "Choose…" picker

  // ---- Modal DOM ----

  function createLogbookModal() {
    if (modal) return modal;

    const overlay = document.createElement("div");
    overlay.className = "logbook-overlay";
    overlay.style.display = "none";
    overlay.innerHTML = `
      <div class="logbook-modal">
        <div class="logbook-modal-header">
          <h3 id="logbook-header-title">Compose Logbook Entry</h3>
          <button class="logbook-modal-close" title="Close">&times;</button>
        </div>
        <div class="logbook-modal-body" id="logbook-body">

          <!-- P2a: Steering Panel -->
          <div id="logbook-phase-steering">
            <div class="logbook-steering-panel">
              <div class="logbook-field">
                <label for="logbook-purpose">Purpose</label>
                <select class="logbook-purpose-select" id="logbook-purpose">
                  <option value="observation" selected>Observation</option>
                  <option value="action_taken">Action Taken</option>
                  <option value="anomaly">Anomaly / Issue</option>
                  <option value="investigation">Investigation</option>
                  <option value="routine_check">Routine Check</option>
                  <option value="general">General</option>
                </select>
              </div>
              <div class="logbook-field">
                <label>Detail Level</label>
                <div class="logbook-detail-toggle" id="logbook-detail-toggle">
                  <button type="button" class="logbook-detail-btn" data-level="brief">Brief</button>
                  <button type="button" class="logbook-detail-btn active" data-level="standard">Standard</button>
                  <button type="button" class="logbook-detail-btn" data-level="detailed">Detailed</button>
                </div>
              </div>
              <div class="logbook-field">
                <label for="logbook-nudge">Additional guidance <span style="color:var(--text-muted)">(optional)</span></label>
                <input type="text" class="logbook-nudge-input" id="logbook-nudge" placeholder="e.g. Focus on SR current readings&hellip;">
              </div>

              <!-- Context selection -->
              <div class="logbook-field">
                <label>Give Access To</label>
                <div class="logbook-context-controls">
                  <label class="logbook-checkbox-label">
                    <input type="checkbox" id="logbook-ctx-session" checked>
                    <span>Session Log</span>
                  </label>
                  <div class="logbook-artifact-scope">
                    <label class="logbook-radio-label">
                      <input type="radio" name="logbook-artifact-scope" value="this" checked>
                      <span>This Artifact</span>
                    </label>
                    <label class="logbook-radio-label">
                      <input type="radio" name="logbook-artifact-scope" value="all">
                      <span>All Artifacts</span>
                    </label>
                    <label class="logbook-radio-label">
                      <input type="radio" name="logbook-artifact-scope" value="choose">
                      <span>Choose&hellip;</span>
                    </label>
                  </div>
                  <div class="logbook-artifact-picker" id="logbook-artifact-picker" style="display:none">
                    <div class="logbook-artifact-picker-list" id="logbook-artifact-picker-list">
                      <span style="color:var(--text-muted); font-size:var(--text-xs);">Loading artifacts&hellip;</span>
                    </div>
                  </div>
                </div>
              </div>

              <!-- Model selector -->
              <div class="logbook-field">
                <label for="logbook-model">Model</label>
                <select class="logbook-purpose-select" id="logbook-model">
                  <option value="haiku" selected>Haiku (fast)</option>
                  <option value="sonnet">Sonnet (balanced)</option>
                  <option value="opus">Opus (thorough)</option>
                </select>
              </div>
            </div>
          </div>

          <!-- P2b: Prompt Preview -->
          <div id="logbook-phase-preview" style="display:none">
            <pre class="logbook-prompt-preview" id="logbook-prompt-text"></pre>
          </div>

          <!-- P2c: Prompt Editor -->
          <div id="logbook-phase-editor" style="display:none">
            <textarea class="logbook-prompt-editor" id="logbook-prompt-edit" spellcheck="false"></textarea>
          </div>

          <!-- P2d: Composing spinner -->
          <div id="logbook-phase-composing" style="display:none">
            <div class="logbook-spinner">Composing entry&hellip;</div>
          </div>

          <!-- P3: Review form -->
          <div id="logbook-phase-review" style="display:none">
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

        <!-- Footer: buttons change per phase -->
        <div class="logbook-modal-footer" id="logbook-footer">
          <div class="logbook-actions" id="logbook-actions"></div>
        </div>
      </div>
    `;

    document.body.appendChild(overlay);
    modal = overlay;

    // Close handlers
    overlay.querySelector(".logbook-modal-close").addEventListener("click", hideModal);
    overlay.addEventListener("click", function (e) {
      if (e.target === overlay) hideModal();
    });

    // Detail toggle buttons
    overlay.querySelectorAll(".logbook-detail-btn").forEach(function (btn) {
      btn.addEventListener("click", function () {
        overlay.querySelectorAll(".logbook-detail-btn").forEach(function (b) {
          b.classList.remove("active");
        });
        btn.classList.add("active");
      });
    });

    // Artifact scope radio → show/hide picker
    overlay.querySelectorAll('input[name="logbook-artifact-scope"]').forEach(function (radio) {
      radio.addEventListener("change", function () {
        var picker = document.getElementById("logbook-artifact-picker");
        if (!picker) return;
        if (radio.value === "choose" && radio.checked) {
          picker.style.display = "";
          loadArtifactPicker();
        } else {
          picker.style.display = "none";
        }
      });
    });

    return modal;
  }

  // ---- Artifact picker ----

  function loadArtifactPicker() {
    var list = document.getElementById("logbook-artifact-picker-list");
    if (!list) return;

    // If already loaded, don't reload
    if (allArtifacts.length > 0) {
      renderArtifactPicker(list);
      return;
    }

    fetch("/api/artifacts")
      .then(function (resp) { return resp.json(); })
      .then(function (data) {
        allArtifacts = data.artifacts || [];
        renderArtifactPicker(list);
      })
      .catch(function () {
        list.innerHTML = '<span style="color:var(--color-error); font-size:var(--text-xs);">Failed to load artifacts</span>';
      });
  }

  function renderArtifactPicker(list) {
    if (allArtifacts.length === 0) {
      list.innerHTML = '<span style="color:var(--text-muted); font-size:var(--text-xs);">No artifacts available</span>';
      return;
    }

    list.innerHTML = "";
    allArtifacts.forEach(function (art) {
      var isCurrentArtifact = (art.id === currentOpts.artifact_id);
      var label = document.createElement("label");
      label.className = "logbook-checkbox-label logbook-artifact-pick-item";
      label.innerHTML =
        '<input type="checkbox" value="' + art.id + '"' + (isCurrentArtifact ? " checked" : "") + '>' +
        '<span class="logbook-artifact-pick-title">' + escapeHtml(art.title || art.id) + '</span>' +
        '<span class="logbook-artifact-pick-type">' + escapeHtml(art.artifact_type || "") + '</span>';
      list.appendChild(label);
    });
  }

  function escapeHtml(str) {
    var div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // ---- Phase management ----

  const PHASES = ["steering", "preview", "editor", "composing", "review"];

  function showPhase(phase) {
    currentPhase = phase;
    PHASES.forEach(function (p) {
      var el = document.getElementById("logbook-phase-" + p);
      if (el) el.style.display = (p === phase) ? "" : "none";
    });
    // Do NOT hide errors here — errors must persist across phase transitions
    // so the user sees them after returning from composing on failure.
    // Errors are cleared explicitly at the start of each action handler.
    renderFooterButtons(phase);
    updateHeaderTitle(phase);
  }

  function clearError() {
    var err = document.getElementById("logbook-error");
    if (err) err.style.display = "none";
  }

  function updateHeaderTitle(phase) {
    var title = document.getElementById("logbook-header-title");
    if (!title) return;
    var titles = {
      steering: "Compose Logbook Entry",
      preview: "Prompt Preview",
      editor: "Edit Prompt",
      composing: "Compose Logbook Entry",
      review: "Review Draft",
    };
    title.textContent = titles[phase] || "Compose Logbook Entry";
  }

  function renderFooterButtons(phase) {
    var container = document.getElementById("logbook-actions");
    if (!container) return;
    container.innerHTML = "";

    if (phase === "steering") {
      container.appendChild(makeBtn("Cancel", "cancel", hideModal));
      container.appendChild(makeBtn("Show Prompt", "secondary", onShowPrompt));
      container.appendChild(makeBtn("Create Logbook Draft", "primary", onCreateDraft));
    } else if (phase === "preview") {
      container.appendChild(makeBtn("Go Back", "cancel", function () { clearError(); showPhase("steering"); }));
      container.appendChild(makeBtn("Manually Edit", "secondary", onManualEdit));
      container.appendChild(makeBtn("Create Logbook Draft", "primary", onCreateDraft));
    } else if (phase === "editor") {
      container.appendChild(makeBtn("Go Back", "cancel", function () { clearError(); showPhase("preview"); }));
      container.appendChild(makeBtn("Create Logbook Draft", "primary", onCreateDraft));
    } else if (phase === "composing") {
      // No buttons during composing
    } else if (phase === "review") {
      container.appendChild(makeBtn("Cancel", "cancel", hideModal));
      container.appendChild(makeBtn("Submit as Draft", "primary", submitLogbook));
    }
  }

  function makeBtn(label, style, handler) {
    var btn = document.createElement("button");
    btn.className = "logbook-btn logbook-btn-" + style;
    btn.textContent = label;
    btn.addEventListener("click", handler);
    return btn;
  }

  // ---- Steering getters ----

  function getSteeringValues() {
    var purpose = document.getElementById("logbook-purpose");
    var activeDetail = document.querySelector(".logbook-detail-btn.active");
    var nudge = document.getElementById("logbook-nudge");
    var model = document.getElementById("logbook-model");
    return {
      purpose: purpose ? purpose.value : "general",
      detail_level: activeDetail ? activeDetail.dataset.level : "standard",
      nudge: nudge ? nudge.value.trim() : "",
      model: model ? model.value : "haiku",
    };
  }

  function getContextValues() {
    var sessionLog = document.getElementById("logbook-ctx-session");
    var include_session_log = sessionLog ? sessionLog.checked : true;

    var scopeRadio = document.querySelector('input[name="logbook-artifact-scope"]:checked');
    var scope = scopeRadio ? scopeRadio.value : "this";

    var artifact_ids = null;
    if (scope === "all") {
      artifact_ids = ["all"];
    } else if (scope === "choose") {
      artifact_ids = [];
      document.querySelectorAll("#logbook-artifact-picker-list input[type=checkbox]:checked").forEach(function (cb) {
        artifact_ids.push(cb.value);
      });
      if (artifact_ids.length === 0) artifact_ids = null; // fall back to single artifact_id
    }
    // scope === "this" → artifact_ids stays null, uses currentOpts.artifact_id

    return {
      include_session_log: include_session_log,
      artifact_ids: artifact_ids,
    };
  }

  // ---- Phase handlers ----

  async function onShowPrompt() {
    clearError();
    var vals = getSteeringValues();
    var body = { purpose: vals.purpose, detail_level: vals.detail_level };
    if (vals.nudge) body.nudge = vals.nudge;

    try {
      var resp = await fetch("/api/logbook/assemble-prompt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!resp.ok) {
        var err = await resp.json().catch(function () { return {}; });
        showError(err.detail || "Failed to assemble prompt (" + resp.status + ")");
        return;
      }
      var data = await resp.json();
      document.getElementById("logbook-prompt-text").textContent = data.prompt;
      showPhase("preview");
    } catch (e) {
      showError("Network error: " + e.message);
    }
  }

  function onManualEdit() {
    var previewText = document.getElementById("logbook-prompt-text").textContent;
    document.getElementById("logbook-prompt-edit").value = previewText;
    showPhase("editor");
  }

  async function onCreateDraft() {
    clearError();

    var sourcePhase = currentPhase;
    var fromEditor = (sourcePhase === "editor");

    // Gather all values BEFORE switching to composing phase (while controls are live)
    var ctxVals = getContextValues();
    var steering = getSteeringValues();

    // Build request body
    var body = {};

    // Artifact identity
    if (ctxVals.artifact_ids) {
      body.artifact_ids = ctxVals.artifact_ids;
    } else if (currentOpts.artifact_id) {
      body.artifact_id = currentOpts.artifact_id;
    }
    // Context controls
    body.include_session_log = ctxVals.include_session_log;

    // Prompt source
    if (fromEditor) {
      var editorEl = document.getElementById("logbook-prompt-edit");
      if (editorEl && editorEl.value.trim()) {
        body.custom_prompt = editorEl.value.trim();
      }
    } else {
      body.purpose = steering.purpose;
      body.detail_level = steering.detail_level;
      if (steering.nudge) body.nudge = steering.nudge;
    }

    // Model selection
    body.model = steering.model;

    // NOW switch to composing spinner
    showPhase("composing");

    try {
      var resp = await fetch("/api/logbook/compose", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!resp.ok) {
        var err = await resp.json().catch(function () { return {}; });
        // Return to source phase FIRST, then show error (so error persists)
        showPhase(fromEditor ? "editor" : "steering");
        showError(err.detail || "Compose failed (" + resp.status + ")");
        return;
      }
      var data = await resp.json();
      showReviewForm(data);
    } catch (e) {
      showPhase(fromEditor ? "editor" : "steering");
      showError("Network error: " + e.message);
    }
  }

  function showReviewForm(data) {
    document.getElementById("logbook-subject").value = data.subject || "";
    document.getElementById("logbook-details").value = data.details || "";
    document.getElementById("logbook-tags").value = (data.tags || []).join(", ");

    var reviewEl = document.getElementById("logbook-phase-review");
    if (reviewEl) reviewEl.dataset.artifactIds = JSON.stringify(data.artifact_ids || []);

    showPhase("review");
  }

  // ---- Shared helpers ----

  function showModal() {
    var m = createLogbookModal();
    m.style.display = "";
  }

  function hideModal() {
    if (modal) modal.style.display = "none";
    resetModal();
  }

  function resetModal() {
    var purpose = document.getElementById("logbook-purpose");
    if (purpose) purpose.value = "observation";
    var nudge = document.getElementById("logbook-nudge");
    if (nudge) nudge.value = "";
    if (modal) {
      modal.querySelectorAll(".logbook-detail-btn").forEach(function (btn) {
        btn.classList.toggle("active", btn.dataset.level === "standard");
      });
    }
    // Reset context controls
    var sessionCb = document.getElementById("logbook-ctx-session");
    if (sessionCb) sessionCb.checked = true;
    var thisRadio = document.querySelector('input[name="logbook-artifact-scope"][value="this"]');
    if (thisRadio) thisRadio.checked = true;
    var picker = document.getElementById("logbook-artifact-picker");
    if (picker) picker.style.display = "none";
    // Reset model
    var model = document.getElementById("logbook-model");
    if (model) model.value = "haiku";
    // Clear editor/preview/review
    var promptText = document.getElementById("logbook-prompt-text");
    if (promptText) promptText.textContent = "";
    var promptEdit = document.getElementById("logbook-prompt-edit");
    if (promptEdit) promptEdit.value = "";
    var subject = document.getElementById("logbook-subject");
    if (subject) subject.value = "";
    var details = document.getElementById("logbook-details");
    if (details) details.value = "";
    var tags = document.getElementById("logbook-tags");
    if (tags) tags.value = "";
    // Clear cached artifacts so picker refreshes next open
    allArtifacts = [];
  }

  function showError(msg) {
    var error = document.getElementById("logbook-error");
    if (error) {
      error.textContent = msg;
      error.style.display = "";
    }
  }

  // ---- Submit ----

  async function submitLogbook() {
    clearError();

    var subject = document.getElementById("logbook-subject").value.trim();
    var details = document.getElementById("logbook-details").value.trim();
    var tagsStr = document.getElementById("logbook-tags").value;
    var tags = tagsStr ? tagsStr.split(",").map(function (t) { return t.trim(); }).filter(Boolean) : [];
    var reviewEl = document.getElementById("logbook-phase-review");
    var artifactIds = reviewEl ? JSON.parse(reviewEl.dataset.artifactIds || "[]") : [];

    if (!subject || !details) {
      showError("Subject and details are required.");
      return;
    }

    var btns = document.querySelectorAll("#logbook-actions .logbook-btn-primary");
    btns.forEach(function (b) { b.disabled = true; });

    try {
      var resp = await fetch("/api/logbook/submit", {
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
        var err = await resp.json().catch(function () { return {}; });
        showError(err.detail || "Submit failed (" + resp.status + ")");
        btns.forEach(function (b) { b.disabled = false; });
        return;
      }
      var data = await resp.json();
      var body = document.getElementById("logbook-body");
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
      var actions = document.getElementById("logbook-actions");
      if (actions) actions.innerHTML = "";
      modal = null;
      setTimeout(hideModal, 3000);
    } catch (e) {
      showError("Network error: " + e.message);
      btns.forEach(function (b) { b.disabled = false; });
    }
  }

  // ---- API: open compose modal ----

  async function openComposeModal(opts) {
    currentOpts = opts;
    showModal();
    clearError();
    showPhase("steering");
  }

  // ---- Button injection ----

  function createLogbookBtn(opts) {
    var btn = document.createElement("button");
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
    var gs = window._galleryState;
    if (!gs) return;

    document.querySelectorAll(".logbook-action-btn").forEach(function (b) { b.remove(); });

    var focusedArtifact = gs.getFocusedArtifact && gs.getFocusedArtifact();
    if (focusedArtifact) {
      var bar = document.querySelector("#focus-container .focus-nav");
      if (bar) {
        bar.appendChild(createLogbookBtn({ artifact_id: focusedArtifact.id }));
      }
    }

    var selectedArtifact = gs.getSelectedArtifact && gs.getSelectedArtifact();
    if (selectedArtifact) {
      var bar = document.querySelector("#preview-content .preview-header-actions");
      if (bar) {
        bar.appendChild(createLogbookBtn({ artifact_id: selectedArtifact.id }));
      }
    }
  }

  window.injectLogbookButtons = injectLogbookButtons;
})();
