// @ts-check
/* OSPREY Logbook — modal HTML template.
 *
 * Stateless markup for the logbook entry composer modal, extracted from
 * logbook.js to keep that module under the max-lines cap. Imported and
 * assigned to overlay.innerHTML by createLogbookModal(). No interpolation:
 * this is a static constant.
 */

export const LOGBOOK_MODAL_HTML = `
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
                    <span style="color:var(--text-muted); font-size:var(--art-text-xs);">Loading artifacts&hellip;</span>
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
