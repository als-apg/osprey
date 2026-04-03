/**
 * OSPREY Channel Finder — Modal System
 *
 * Form dialogs, confirm dialogs, and a low-level showModal primitive.
 */

import { esc } from './utils.js';

let _activeModal = null;

/**
 * Show a form modal that collects named fields.
 * @param {object} opts
 * @param {string} opts.title - Dialog title.
 * @param {Array<{name:string, label:string, type?:string, required?:boolean, value?:string}>} opts.fields
 * @param {string} [opts.submitLabel='Save'] - Submit button label.
 * @returns {Promise<object|null>} Field values keyed by name, or null on cancel.
 */
export function formModal({ title, fields, submitLabel = 'Save' }) {
  return new Promise(resolve => {
    const fieldHTML = fields.map(f => {
      const id = `modal-field-${f.name}`;
      const req = f.required ? 'required' : '';
      const val = f.value != null ? esc(f.value) : '';
      if (f.type === 'textarea') {
        return `<div class="form-group">
          <label class="form-label" for="${id}">${esc(f.label)}</label>
          <textarea class="form-input modal-field" id="${id}" data-name="${esc(f.name)}"
                    ${req} rows="3">${val}</textarea>
        </div>`;
      }
      if (f.type === 'select') {
        const options = (f.options || []).map(o =>
          `<option value="${esc(o.value)}"${o.value === f.value ? ' selected' : ''}>${esc(o.label)}</option>`
        ).join('');
        return `<div class="form-group">
          <label class="form-label" for="${id}">${esc(f.label)}</label>
          <select class="form-input modal-field" id="${id}" data-name="${esc(f.name)}" ${req}>${options}</select>
        </div>`;
      }
      return `<div class="form-group">
        <label class="form-label" for="${id}">${esc(f.label)}</label>
        <input class="form-input modal-field" id="${id}" data-name="${esc(f.name)}"
               type="${f.type || 'text'}" ${req} value="${val}"
               ${f.placeholder ? `placeholder="${esc(f.placeholder)}"` : ''}>
      </div>`;
    }).join('');

    const actions = `
      <button class="btn btn-secondary modal-cancel">Cancel</button>
      <button class="btn btn-primary modal-submit">${esc(submitLabel)}</button>
    `;

    const { close, el } = showModal({ title, body: fieldHTML, actions });

    const submitBtn = el.querySelector('.modal-submit');
    const cancelBtn = el.querySelector('.modal-cancel');
    const inputs = el.querySelectorAll('.modal-field');

    // Focus first input
    inputs[0]?.focus();

    // Required-field validation
    const checkValidity = () => {
      const allValid = [...inputs].every(inp =>
        !inp.hasAttribute('required') || inp.value.trim()
      );
      submitBtn.disabled = !allValid;
    };
    inputs.forEach(inp => inp.addEventListener('input', checkValidity));
    checkValidity();

    const submit = () => {
      const result = {};
      inputs.forEach(inp => { result[inp.dataset.name] = inp.value.trim(); });
      close();
      resolve(result);
    };

    const cancel = () => { close(); resolve(null); };

    submitBtn.addEventListener('click', submit);
    cancelBtn.addEventListener('click', cancel);

    // Enter key submits (on non-textarea inputs)
    inputs.forEach(inp => {
      if (inp.tagName !== 'TEXTAREA') {
        inp.addEventListener('keydown', e => {
          if (e.key === 'Enter' && !submitBtn.disabled) submit();
        });
      }
    });
  });
}

/**
 * Show a confirmation dialog with optional impact callout.
 * @param {object} opts
 * @param {string} opts.title - Dialog title.
 * @param {string} opts.message - Confirmation message.
 * @param {string} [opts.impact] - Amber impact callout text.
 * @param {string} [opts.confirmLabel='Delete'] - Confirm button label.
 * @param {boolean} [opts.danger=false] - Style confirm as danger button.
 * @returns {Promise<boolean>} True if confirmed.
 */
export function confirmModal({ title, message, impact, confirmLabel = 'Delete', danger = false }) {
  return new Promise(resolve => {
    const body = `
      <p style="color: var(--text-secondary); margin: 0 0 var(--space-3);">${esc(message)}</p>
      ${impact ? `<div class="modal-impact">${esc(impact)}</div>` : ''}
    `;
    const actions = `
      <button class="btn btn-secondary modal-cancel">Cancel</button>
      <button class="btn ${danger ? 'btn-danger' : 'btn-primary'} modal-confirm">${esc(confirmLabel)}</button>
    `;

    const { close, el } = showModal({ title, body, actions, size: 'sm' });

    el.querySelector('.modal-confirm').addEventListener('click', () => { close(); resolve(true); });
    el.querySelector('.modal-cancel').addEventListener('click', () => { close(); resolve(false); });
    el.querySelector('.modal-confirm').focus();
  });
}

/**
 * Low-level modal primitive.
 * @param {object} opts
 * @param {string} opts.title - Modal title.
 * @param {string} opts.body - Inner HTML for modal body.
 * @param {string} opts.actions - Inner HTML for footer buttons.
 * @param {string} [opts.size='md'] - Size class: 'sm' | 'md'.
 * @returns {{ close: Function, el: HTMLElement }} Close function and panel element.
 */
export function showModal({ title, body, actions, size = 'md' }) {
  // Close any existing modal
  if (_activeModal) _activeModal.close();

  const backdrop = document.createElement('div');
  backdrop.className = 'modal-backdrop';
  backdrop.innerHTML = `
    <div class="modal-panel modal-${size}">
      <div class="modal-header">
        <span class="modal-title">${esc(title)}</span>
        <button class="modal-close" aria-label="Close">&times;</button>
      </div>
      <div class="modal-body">${body}</div>
      ${actions ? `<div class="modal-footer">${actions}</div>` : ''}
    </div>
  `;

  document.body.appendChild(backdrop);

  const panel = backdrop.querySelector('.modal-panel');

  const close = () => {
    backdrop.remove();
    if (_activeModal?.close === close) _activeModal = null;
  };

  // Close on backdrop click (not panel)
  backdrop.addEventListener('click', e => { if (e.target === backdrop) close(); });
  // Close on ESC
  const onKey = e => { if (e.key === 'Escape') { close(); document.removeEventListener('keydown', onKey); } };
  document.addEventListener('keydown', onKey);
  // Close button
  backdrop.querySelector('.modal-close').addEventListener('click', close);

  _activeModal = { close };
  return { close, el: panel };
}
