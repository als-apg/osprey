// @ts-nocheck
// TODO(frontend-hardening): type-clean this test; tracked in eslint.config.js local/no-ts-nocheck allowlist, which may only shrink.
/**
 * Unit tests for the Lattice Dashboard computation-settings layer
 * (settings.js: the declarative SETTINGS_FIELDS schema plus the settings
 * form built from it).
 *
 * happy-dom environment (configured globally), fetch mocked (net.js's
 * apiFetch is the only external dependency):
 *   npx vitest run tests/interfaces/lattice_dashboard/settings.test.mjs
 */

import { test, expect, vi, describe, afterEach, beforeEach } from 'vitest';

import {
  SETTINGS_FIELDS,
  renderSettingsForm,
  collectSettingsFromForm,
  loadSettings,
} from '../../../src/osprey/interfaces/lattice_dashboard/static/js/settings.js';

/** Minimal DOM fixture matching lattice_dashboard/static/index.html's structure. */
function mountFixture() {
  document.body.innerHTML = `<div id="settings-container"></div>`;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('renderSettingsForm', () => {
  beforeEach(mountFixture);

  test('renders exactly one input per schema field, across every group', () => {
    renderSettingsForm({});
    const totalFields = Object.values(SETTINGS_FIELDS).reduce((n, g) => n + g.fields.length, 0);
    const inputs = document.querySelectorAll('.settings-field-input');
    expect(inputs.length).toBe(totalFields);
  });

  test('renders one group header per SETTINGS_FIELDS entry, in schema order', () => {
    renderSettingsForm({});
    const headers = [...document.querySelectorAll('.settings-group-header')].map(h => h.textContent);
    expect(headers).toEqual(Object.values(SETTINGS_FIELDS).map(g => g.label));
  });

  test('each input carries the correct type/min/max/step/dataset from its schema field', () => {
    renderSettingsForm({});
    for (const [group, meta] of Object.entries(SETTINGS_FIELDS)) {
      for (const field of meta.fields) {
        const input = document.getElementById(`setting-${group}-${field.key}`);
        expect(input, `${group}.${field.key} should exist`).not.toBeNull();
        expect(input.type).toBe('number');
        expect(input.min).toBe(String(field.min));
        expect(input.max).toBe(String(field.max));
        expect(input.step).toBe(String(field.step));
        expect(input.dataset.group).toBe(group);
        expect(input.dataset.key).toBe(field.key);
        expect(input.dataset.fieldType).toBe(field.type);
      }
    }
  });

  test('a known settings value populates the input, int fields rounded', () => {
    renderSettingsForm({ da: { nturns: 512.7, n_bisect: 10 } });
    expect(document.getElementById('setting-da-nturns').value).toBe('513');
  });

  test('an unset int_or_null field shows the "auto" placeholder, not a value', () => {
    renderSettingsForm({ lma: {} });
    const input = document.getElementById('setting-lma-n_sectors');
    expect(input.value).toBe('');
    expect(input.placeholder).toBe('auto');
  });

  test('an explicit int_or_null value overrides the placeholder', () => {
    renderSettingsForm({ lma: { n_sectors: 42 } });
    const input = document.getElementById('setting-lma-n_sectors');
    expect(input.value).toBe('42');
  });

  test('a float field keeps its fractional value unrounded', () => {
    renderSettingsForm({ chromaticity: { dp_min_pct: -12.5 } });
    expect(document.getElementById('setting-chromaticity-dp_min_pct').value).toBe('-12.5');
  });

  test('re-rendering replaces the previous form rather than appending to it', () => {
    renderSettingsForm({});
    renderSettingsForm({});
    const totalFields = Object.values(SETTINGS_FIELDS).reduce((n, g) => n + g.fields.length, 0);
    expect(document.querySelectorAll('.settings-field-input').length).toBe(totalFields);
  });

  test('renders RESET and APPLY action buttons', () => {
    renderSettingsForm({});
    expect(document.querySelector('.settings-btn:not(.settings-btn--apply)').textContent).toBe('RESET');
    expect(document.querySelector('.settings-btn--apply').textContent).toBe('APPLY');
  });

  test('is a no-op when #settings-container is absent from the page', () => {
    document.body.innerHTML = '';
    expect(() => renderSettingsForm({})).not.toThrow();
  });
});

describe('collectSettingsFromForm', () => {
  beforeEach(mountFixture);

  test('round-trips every declared field, honoring each type', () => {
    // Seed with a value for every field (including int_or_null) so the
    // round trip covers all three type branches.
    const seeded = {};
    for (const [group, meta] of Object.entries(SETTINGS_FIELDS)) {
      seeded[group] = {};
      for (const field of meta.fields) {
        seeded[group][field.key] = field.type === 'float' ? field.min + field.step / 2 : field.min;
      }
    }
    renderSettingsForm(seeded);

    const collected = collectSettingsFromForm();

    for (const [group, meta] of Object.entries(SETTINGS_FIELDS)) {
      for (const field of meta.fields) {
        const expected = seeded[group][field.key];
        const actual = collected[group][field.key];
        if (field.type === 'float') {
          expect(actual, `${group}.${field.key}`).toBeCloseTo(expected, 5);
        } else {
          expect(actual, `${group}.${field.key}`).toBe(Math.round(expected));
        }
      }
    }
  });

  test('an int_or_null field left blank round-trips to null', () => {
    renderSettingsForm({});
    const input = document.getElementById('setting-lma-n_sectors');
    input.value = '';
    const collected = collectSettingsFromForm();
    expect(collected.lma.n_sectors).toBeNull();
  });

  test('an int_or_null field with a value round-trips to an integer', () => {
    renderSettingsForm({});
    const input = document.getElementById('setting-lma-n_sectors');
    input.value = '7';
    const collected = collectSettingsFromForm();
    expect(collected.lma.n_sectors).toBe(7);
  });

  test('an int field truncates a fractional string via parseInt', () => {
    renderSettingsForm({});
    const input = document.getElementById('setting-da-nturns');
    input.value = '100.9';
    const collected = collectSettingsFromForm();
    expect(collected.da.nturns).toBe(100);
  });

  test('a float field preserves fractional precision', () => {
    renderSettingsForm({});
    const input = document.getElementById('setting-chromaticity-dp_min_pct');
    input.value = '-3.25';
    const collected = collectSettingsFromForm();
    expect(collected.chromaticity.dp_min_pct).toBeCloseTo(-3.25, 5);
  });
});

describe('loadSettings', () => {
  beforeEach(mountFixture);

  test('fetches /api/settings and renders the returned form', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ da: { nturns: 256 } }),
    }));

    await loadSettings();

    expect(fetch).toHaveBeenCalledWith('/api/settings', expect.objectContaining({
      headers: { 'Content-Type': 'application/json' },
    }));
    expect(document.getElementById('setting-da-nturns').value).toBe('256');
  });

  test('a fetch failure is caught and logged, leaving the form unrendered', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('network down')));
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

    await expect(loadSettings()).resolves.toBeUndefined();
    expect(warnSpy).toHaveBeenCalled();
    expect(document.querySelectorAll('.settings-field-input').length).toBe(0);

    warnSpy.mockRestore();
  });
});
