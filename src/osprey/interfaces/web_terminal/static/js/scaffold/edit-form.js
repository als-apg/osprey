// @ts-check
/**
 * OSPREY Web Terminal — Scaffold Gallery: edit-view forms
 *
 * The Edit mode's content renderers: the front-matter form
 * (name/description/model/maxTurns/disallowedTools fields plus an
 * instructions textarea, for agent-shaped artifacts), and the plain-text
 * fallback textarea for everything else.
 * The "edit forms" half of the edit workflow; the write-side actions
 * (ownership take/release, discard/save, reset-to-framework,
 * reload+reopen, close-detail) live in scaffold/edit.js. Mirrors the
 * shell-vs-content seam of scaffold/detail.js + scaffold/detail-content.js.
 *
 * Every field/textarea input here just flips `gallery.editDirty` and
 * re-renders the mode tabs (so Discard/Save appear) -- the actual save/
 * discard/ownership workflow that reads this dirty flag back out lives in
 * scaffold/edit.js.
 *
 * @module scaffold/edit-form
 */

import { fetchJSON } from '../api.js';
import { parseFrontMatter, lockEditor } from './utils.js';

/**
 * `detailContentEl` grows `_frontMatterFields`/`_bodyTextarea` when the
 * front-matter form is mounted below -- both read back by
 * ArtifactGallery.saveOverride() (scaffold/edit.js).
 * @typedef {HTMLElement & {
 *   _frontMatterFields?: Record<string, HTMLInputElement>,
 *   _bodyTextarea?: HTMLTextAreaElement,
 * }} EditContentElement
 */

/**
 * The subset of an ArtifactGallery instance these edit-form renderers read,
 * write, or call into.
 * @typedef {object} ScaffoldGalleryEditFormHost
 * @property {any} selectedArtifact
 * @property {boolean} editDirty
 * @property {EditContentElement|null} detailContentEl
 * @property {number} detailRenderSeq
 * @property {() => void} renderDetailModes
 * @property {{id: string, name: string}[]} [servedModels] Ids the provider
 *   serves, offered as suggestions in an agent's model field.
 */

/**
 * Create the scaffold gallery's edit-form renderers, bound to a fixed
 * gallery host.
 *
 * @param {ScaffoldGalleryEditFormHost} gallery
 */
export function createScaffoldGalleryEditForm(gallery) {
  /** @returns {Promise<void>} */
  async function renderEdit() {
    // Same claim on the content pane the read-side renderers take: a mode
    // click during a slow fetch puts a second render in flight, and the later
    // one has to win regardless of which fetch lands first (see
    // scaffold/detail.js's renderDetailContent).
    const render = gallery.detailRenderSeq;

    const data = await fetchJSON(`/api/scaffold/${encodeURIComponent(gallery.selectedArtifact.name)}`); // fetchJSON prefixes internally
    const content = data.content || '';

    if (!gallery.detailContentEl || gallery.detailRenderSeq !== render) return;

    // Clear content area
    while (gallery.detailContentEl.firstChild) {
      gallery.detailContentEl.removeChild(gallery.detailContentEl.firstChild);
    }

    const { frontMatter, body } = parseFrontMatter(content);
    if (frontMatter && frontMatter.model) {
      renderFrontMatterForm(content, frontMatter, body);
    } else {
      renderPlainTextEditor(content);
    }
  }

  /**
   * @param {string} content
   * @returns {void}
   */
  function renderPlainTextEditor(content) {
    if (!gallery.detailContentEl) return;

    const textarea = document.createElement('textarea');
    textarea.className = 'prompts-edit-textarea';
    textarea.spellcheck = false;
    textarea.value = content;
    if (gallery.selectedArtifact?.read_only) {
      lockEditor(textarea);
    }

    textarea.addEventListener('input', () => {
      gallery.editDirty = true;
      gallery.renderDetailModes();
    });

    gallery.detailContentEl.appendChild(textarea);
  }

  /**
   * @param {string} fullContent
   * @param {Record<string, string>} frontMatter
   * @param {string} body
   * @returns {void}
   */
  function renderFrontMatterForm(fullContent, frontMatter, body) {
    const container = gallery.detailContentEl;
    if (!container) return;

    const form = document.createElement('div');
    form.className = 'prompts-fm-form';

    const formTitle = document.createElement('div');
    formTitle.className = 'prompts-fm-form-title';
    formTitle.textContent = 'AGENT CONFIGURATION';
    form.appendChild(formTitle);

    const fieldDefs = [
      { key: 'name', label: 'name', type: 'text' },
      { key: 'description', label: 'description', type: 'text' },
      { key: 'model', label: 'model', type: 'model' },
      { key: 'maxTurns', label: 'maxTurns', type: 'number' },
      { key: 'disallowedTools', label: 'disallowedTools', type: 'text' },
    ];

    /** @type {Record<string, HTMLInputElement>} */
    const fieldRefs = {};

    for (const def of fieldDefs) {
      const value = frontMatter[def.key] || '';
      const { wrapper, input } = _createFormField(def.key, def.label, def.type, value);
      form.appendChild(wrapper);
      fieldRefs[def.key] = input;
    }

    container.appendChild(form);

    const instrTitle = document.createElement('div');
    instrTitle.className = 'prompts-fm-form-title';
    instrTitle.style.padding = '10px 12px 0';
    instrTitle.textContent = 'AGENT INSTRUCTIONS';
    container.appendChild(instrTitle);

    const bodyTextarea = document.createElement('textarea');
    bodyTextarea.className = 'prompts-edit-textarea';
    bodyTextarea.spellcheck = false;
    bodyTextarea.value = body;
    if (gallery.selectedArtifact?.read_only) {
      lockEditor(bodyTextarea);
      for (const input of Object.values(fieldRefs)) input.disabled = true;
    }

    bodyTextarea.addEventListener('input', () => {
      gallery.editDirty = true;
      gallery.renderDetailModes();
    });

    container.appendChild(bodyTextarea);

    // Store references for saveOverride()
    container._frontMatterFields = fieldRefs;
    container._bodyTextarea = bodyTextarea;
  }

  /**
   * @param {string} key
   * @param {string} label
   * @param {string} type
   * @param {string} value
   * @returns {{wrapper: HTMLDivElement, input: HTMLInputElement}}
   */
  function _createFormField(key, label, type, value) {
    const wrapper = document.createElement('div');
    wrapper.className = 'prompts-fm-field';

    const labelEl = document.createElement('span');
    labelEl.className = 'prompts-fm-field-label';
    labelEl.textContent = label;
    wrapper.appendChild(labelEl);

    /** @type {HTMLInputElement} */
    let input;

    if (type === 'model') {
      // A model id the provider serves. Free text: the served ids, when the
      // gallery carries them, are offered as suggestions, never as the only
      // choices.
      input = document.createElement('input');
      input.type = 'text';
      input.className = 'settings-input';
      input.value = value;
      const served = gallery.servedModels || [];
      if (served.length) {
        const list = document.createElement('datalist');
        list.id = `prompts-fm-${key}-options`;
        for (const model of served) {
          const optEl = document.createElement('option');
          optEl.value = model.id;
          optEl.label = model.name;
          list.appendChild(optEl);
        }
        input.setAttribute('list', list.id);
        wrapper.appendChild(list);
      }
    } else if (type === 'number') {
      input = document.createElement('input');
      input.type = 'number';
      input.className = 'settings-input';
      input.min = '1';
      input.max = '100';
      input.value = value;
    } else {
      input = document.createElement('input');
      input.type = 'text';
      input.className = 'settings-input';
      input.value = value;
    }

    input.addEventListener('input', () => {
      gallery.editDirty = true;
      gallery.renderDetailModes();
    });
    input.addEventListener('change', () => {
      gallery.editDirty = true;
      gallery.renderDetailModes();
    });

    wrapper.appendChild(input);
    return { wrapper, input };
  }

  return { renderEdit, renderPlainTextEditor, renderFrontMatterForm };
}
