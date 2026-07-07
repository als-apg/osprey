/**
 * Unit tests for the Artifact Gallery logbook entry composer (logbook.js):
 * the client-side DOM gap left by the backend `test_logbook.py` suite
 * (which remains the API authority for `/api/logbook/*`).
 *
 * Pure-logic/DOM guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/artifacts/logbook.test.mjs
 *
 * Covers the module's behavior-neutral named exports added for Task 2.4:
 *   - makeBtn: footer button factory (class, text, click wiring).
 *   - updateHeaderTitle: phase -> header-title text lookup, including the
 *     unknown-phase fallback.
 *   - getSteeringValues / getContextValues: branch logic over the steering
 *     panel and context controls, including the artifact-picker checkbox
 *     `.value` readback path (the Task 1.4 attribute-sink fix — ids are
 *     assigned as a DOM property, never interpolated into innerHTML).
 *
 * `escapeHtml` is the canonical design-system helper (tested in
 * tests/interfaces/design_system/dom.test.mjs and exercised end-to-end
 * against hostile artifact metadata in security_render.test.mjs) — it is
 * deliberately NOT re-tested here.
 */

import { test, expect, describe, afterEach } from 'vitest';

import {
  makeBtn,
  updateHeaderTitle,
  getSteeringValues,
  getContextValues,
} from '../../../src/osprey/interfaces/artifacts/static/js/logbook.js';

afterEach(() => {
  document.body.innerHTML = '';
});

// =========================================================================
// makeBtn
// =========================================================================

describe('makeBtn', () => {
  test('builds a button with the expected class and label text', () => {
    const btn = makeBtn('Cancel', 'cancel', () => {});
    expect(btn.tagName).toBe('BUTTON');
    expect(btn.className).toBe('logbook-btn logbook-btn-cancel');
    expect(btn.textContent).toBe('Cancel');
  });

  test('wires the click handler so a click invokes it exactly once', () => {
    let calls = 0;
    const btn = makeBtn('Submit', 'primary', () => { calls += 1; });
    document.body.appendChild(btn);

    btn.click();
    btn.click();

    expect(calls).toBe(2);
  });

  test('a different style produces a different modifier class', () => {
    const btn = makeBtn('Show Prompt', 'secondary', () => {});
    expect(btn.className).toBe('logbook-btn logbook-btn-secondary');
  });
});

// =========================================================================
// updateHeaderTitle
// =========================================================================

describe('updateHeaderTitle', () => {
  function mountTitleFixture() {
    document.body.innerHTML = '<h3 id="logbook-header-title">placeholder</h3>';
  }

  test.each([
    ['steering', 'Compose Logbook Entry'],
    ['preview', 'Prompt Preview'],
    ['editor', 'Edit Prompt'],
    ['composing', 'Compose Logbook Entry'],
    ['review', 'Review Draft'],
  ])('phase "%s" sets the header title to "%s"', (phase, expected) => {
    mountTitleFixture();
    updateHeaderTitle(phase);
    expect(document.getElementById('logbook-header-title').textContent).toBe(expected);
  });

  test('an unknown phase falls back to the default composer title', () => {
    mountTitleFixture();
    updateHeaderTitle('not-a-real-phase');
    expect(document.getElementById('logbook-header-title').textContent).toBe('Compose Logbook Entry');
  });

  test('is a no-op (does not throw) when the header title element is absent', () => {
    document.body.innerHTML = '';
    expect(() => updateHeaderTitle('review')).not.toThrow();
  });
});

// =========================================================================
// getSteeringValues
// =========================================================================

describe('getSteeringValues', () => {
  function mountSteeringFixture() {
    document.body.innerHTML = `
      <select id="logbook-purpose">
        <option value="observation">Observation</option>
        <option value="anomaly" selected>Anomaly</option>
      </select>
      <div class="logbook-detail-toggle">
        <button type="button" class="logbook-detail-btn" data-level="brief">Brief</button>
        <button type="button" class="logbook-detail-btn active" data-level="detailed">Detailed</button>
      </div>
      <input type="text" id="logbook-nudge" value="  focus on SR current  ">
      <select id="logbook-model">
        <option value="haiku">Haiku</option>
        <option value="opus" selected>Opus</option>
      </select>
    `;
  }

  test('reads the live control values, trimming the nudge text', () => {
    mountSteeringFixture();
    expect(getSteeringValues()).toEqual({
      purpose: 'anomaly',
      detail_level: 'detailed',
      nudge: 'focus on SR current',
      model: 'opus',
    });
  });

  test('falls back to defaults for every field when no controls are present', () => {
    document.body.innerHTML = '';
    expect(getSteeringValues()).toEqual({
      purpose: 'general',
      detail_level: 'standard',
      nudge: '',
      model: 'haiku',
    });
  });

  test('detail_level falls back to "standard" when no detail button carries .active', () => {
    mountSteeringFixture();
    document.querySelector('.logbook-detail-btn.active').classList.remove('active');
    expect(getSteeringValues().detail_level).toBe('standard');
  });

  test('nudge is an empty string (not whitespace) when the input is blank', () => {
    mountSteeringFixture();
    /** @type {HTMLInputElement} */ (document.getElementById('logbook-nudge')).value = '   ';
    expect(getSteeringValues().nudge).toBe('');
  });
});

// =========================================================================
// getContextValues
// =========================================================================

describe('getContextValues', () => {
  /**
   * Mirrors the real steering-panel markup for the context controls plus
   * the artifact picker list. `checkedIds`, when given, are appended as
   * checkboxes with ids assigned via the `.value` DOM PROPERTY — exactly
   * how the real `renderArtifactPicker` (logbook.js) does it for the
   * Task 1.4 attribute-sink fix — never interpolated into a `value="..."`
   * HTML string.
   *
   * @param {{sessionChecked?: boolean, scope?: 'this'|'all'|'choose'|null, pickerIds?: string[], checkedPickerIds?: string[]}} opts
   */
  function mountContextFixture({
    sessionChecked = true,
    scope = 'this',
    pickerIds = [],
    checkedPickerIds = [],
  } = {}) {
    document.body.innerHTML = `
      <input type="checkbox" id="logbook-ctx-session" ${sessionChecked ? 'checked' : ''}>
      <div class="logbook-artifact-scope">
        <label><input type="radio" name="logbook-artifact-scope" value="this" ${scope === 'this' ? 'checked' : ''}></label>
        <label><input type="radio" name="logbook-artifact-scope" value="all" ${scope === 'all' ? 'checked' : ''}></label>
        <label><input type="radio" name="logbook-artifact-scope" value="choose" ${scope === 'choose' ? 'checked' : ''}></label>
      </div>
      <div id="logbook-artifact-picker-list"></div>
    `;

    const list = document.getElementById('logbook-artifact-picker-list');
    pickerIds.forEach((id) => {
      const label = document.createElement('label');
      label.innerHTML = '<input type="checkbox">';
      const cb = /** @type {HTMLInputElement} */ (label.querySelector('input[type=checkbox]'));
      cb.value = id; // property assignment, matching renderArtifactPicker
      cb.checked = checkedPickerIds.includes(id);
      list.appendChild(label);
    });
  }

  test('scope "this": session checked, artifact_ids stays null (uses currentOpts.artifact_id)', () => {
    mountContextFixture({ sessionChecked: true, scope: 'this' });
    expect(getContextValues()).toEqual({ include_session_log: true, artifact_ids: null });
  });

  test('session log unchecked is reflected in include_session_log', () => {
    mountContextFixture({ sessionChecked: false, scope: 'this' });
    expect(getContextValues().include_session_log).toBe(false);
  });

  test('scope "all" sets artifact_ids to ["all"]', () => {
    mountContextFixture({ scope: 'all' });
    expect(getContextValues().artifact_ids).toEqual(['all']);
  });

  test('scope "choose" collects the ids of checked picker checkboxes via .value', () => {
    mountContextFixture({
      scope: 'choose',
      pickerIds: ['art-1', 'art-2', 'art-3'],
      checkedPickerIds: ['art-1', 'art-3'],
    });
    expect(getContextValues().artifact_ids).toEqual(['art-1', 'art-3']);
  });

  test('scope "choose" with nothing checked falls back to null (single artifact_id path)', () => {
    mountContextFixture({
      scope: 'choose',
      pickerIds: ['art-1', 'art-2'],
      checkedPickerIds: [],
    });
    expect(getContextValues().artifact_ids).toBeNull();
  });

  test('no scope radio checked defaults to "this" (artifact_ids null) and session defaults true when the checkbox is absent', () => {
    document.body.innerHTML = '<div id="logbook-artifact-picker-list"></div>';
    expect(getContextValues()).toEqual({ include_session_log: true, artifact_ids: null });
  });
});
