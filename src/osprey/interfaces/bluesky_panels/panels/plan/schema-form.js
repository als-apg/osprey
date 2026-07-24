// @ts-check
/**
 * schema-form — build an interactive, two-dimensional parameter GUI from a
 * plan's JSON Schema and read a nested `plan_args` object back out of it.
 *
 * Plan schemas are Pydantic ``model_json_schema()`` documents, so this renderer
 * handles the full shape those emit — not just flat scalars. Each schema shape
 * gets a purpose-built control:
 *
 * - ``string`` / ``integer`` / ``number`` scalars → typed inputs (numeric
 *   bounds surfaced as native ``min``/``max``),
 * - ``boolean`` → a toggle switch,
 * - ``enum`` → a ``<select>``,
 * - ``array`` of scalars → a **chip editor** (type a value, Enter/comma/blur
 *   commits it as a removable chip — built for device-name lists),
 * - ``array`` of flat objects → an **editable table** (one column per field,
 *   one row per item, add/remove rows — built for ``grid_scan``'s
 *   ``axes: list[GridAxis]``),
 * - anything deeper (arrays of non-flat objects, nested objects) → stacked
 *   row/fieldset editors as a generic fallback.
 *
 * Nested models arrive as ``$ref`` pointers into a top-level ``$defs`` map, and
 * optional/defaulted fields arrive wrapped in ``anyOf``/``allOf`` — both are
 * resolved here so builders only ever see a concrete node.
 *
 * Layout is two-dimensional: ``renderSchemaForm``'s ``opts.layout`` accepts
 * rows of field names (``[['correctors','detectors'], ['span_a','num']]``) so
 * a plan can place fields side by side; unlisted fields are auto-flowed
 * (scalars packed into shared rows, wide editors on their own row).
 *
 * Everything is built with ``createElement`` + ``textContent``/``.value`` — no
 * ``innerHTML`` — so schema-derived strings (titles, descriptions, enum
 * values) are never interpreted as markup. There is no HTML-injection surface
 * in this module by construction.
 *
 * Structural edits (chip added/removed, table row added/removed) dispatch a
 * bubbling ``form-change`` CustomEvent from the edited element, so a host can
 * listen on the form container (alongside native ``input``/``change``) to
 * recompute live summaries.
 *
 * Each builder returns a ``{ el, collect, setValue }`` triple: ``el`` is the
 * DOM node to mount, ``collect()`` returns that field's current value — or
 * the ``OMIT`` sentinel when the field is blank, so plan-side defaults apply
 * for anything the operator left untouched — and ``setValue(value)`` applies
 * a value programmatically (no native ``input``/``change`` events), for a
 * server-held draft to be reflected into a rendered form.
 *
 * ``renderSchemaForm`` keys every top-level field's ``{el, setValue}`` into a
 * field registry and attaches it, plus an ``applyValues(values)`` method, to
 * the returned collector function (the callable return contract is
 * unchanged — it is still ``collectPlanArgs()`` to a caller that only calls
 * it). ``applyValues`` composes recursively the same way ``collect()``
 * does — a nested object's ``setValue`` fans out to its children's
 * ``setValue`` — but containers with a dynamic row count (the chip/
 * channel-list editors, the axes table, array-of-object rows) treat their
 * value as a whole-value replacement: they rebuild their internal state from
 * the incoming array rather than patch existing rows, since a static
 * per-row registry would go stale across add/remove-row edits. Only one
 * ``form-change`` event fires per ``applyValues`` call (documented at the
 * call site as programmatic, so a host can tell it apart from a user edit).
 *
 * @module schema-form
 */

/**
 * Sentinel returned by a field's ``collect()`` when it has no value to
 * contribute (blank input, empty chip list, …). Parent collectors drop
 * ``OMIT`` children rather than emitting ``null``/``""`` so the plan's own
 * defaults apply.
 *
 * @type {unique symbol}
 */
export const OMIT = Symbol('omit');

/**
 * @typedef {object} JsonSchemaNode
 * @property {string} [type]
 * @property {string} [title]
 * @property {string} [description]
 * @property {unknown} [default]
 * @property {unknown[]} [enum]
 * @property {JsonSchemaNode} [items]
 * @property {Record<string, JsonSchemaNode>} [properties]
 * @property {string[]} [required]
 * @property {string} [$ref]
 * @property {JsonSchemaNode[]} [anyOf]
 * @property {JsonSchemaNode[]} [allOf]
 * @property {number} [minimum]
 * @property {number} [maximum]
 * @property {number} [exclusiveMinimum]
 * @property {number} [exclusiveMaximum]
 * @property {number} [minItems]
 * @property {string} [x-widget]
 */

/**
 * @typedef {object} Field
 * @property {HTMLElement} el       The DOM node to mount for this field.
 * @property {() => unknown} collect Current value, or ``OMIT`` if blank.
 * @property {(value: unknown) => void} setValue Apply ``value`` programmatically:
 *   replaces the control's displayed/internal state without firing native
 *   ``input``/``change`` events. Containers (chips, channel-list, the axes
 *   table, array-of-object rows, nested objects) treat this as whole-value
 *   replacement — they rebuild internal rows/children from ``value`` rather
 *   than patch individual cells.
 */

/**
 * One entry in the top-level field registry that ``renderSchemaForm``
 * exposes via the returned collector's ``.fields`` property — the DOM node
 * plus the programmatic setter, keyed by top-level schema property name.
 *
 * @typedef {object} RegisteredField
 * @property {HTMLElement} el
 * @property {(value: unknown) => void} setValue
 */

/**
 * The callable ``renderSchemaForm`` returns: a zero-arg function that reads
 * the whole form into a ``plan_args`` object (unchanged contract — this is
 * what ``panel.js`` calls as ``collectPlanArgs()``), plus two properties
 * layered on top for programmatic draft application:
 * ``applyValues(values)`` sets every field present in ``values`` (keys
 * absent from ``values`` are left untouched) and dispatches one
 * ``form-change`` event; ``fields`` is the top-level field registry.
 * ``applyValues`` accepts ``unknown`` (not just ``Record<string, unknown>``)
 * because a missing/malformed server-held draft (``undefined``/``null``/a
 * non-object) is a safe no-op rather than a caller error — see its
 * implementation in ``withFieldRegistry``.
 *
 * @typedef {(() => Record<string, unknown>) & {
 *   applyValues: (values: unknown) => void,
 *   fields: Record<string, RegisteredField>
 * }} PlanArgsCollector
 */

/**
 * Tiny hyperscript helper: create an element, apply props, append children.
 * Strings/numbers become text nodes (never parsed as HTML).
 *
 * @param {string} tag
 * @param {Record<string, unknown>} [props]
 * @param {...(Node|string|number|null|undefined)} children
 * @returns {HTMLElement}
 */
function h(tag, props, ...children) {
  const node = document.createElement(tag);
  if (props) {
    for (const [key, value] of Object.entries(props)) {
      if (value === null || value === undefined || value === false) continue;
      if (key === 'class') {
        node.className = String(value);
      } else if (key === 'text') {
        node.textContent = String(value);
      } else if (key in node) {
        // Property assignment (value, checked, disabled, min, max, step, …).
        /** @type {any} */ (node)[key] = value;
      } else {
        node.setAttribute(key, String(value));
      }
    }
  }
  for (const child of children) {
    if (child === null || child === undefined) continue;
    node.appendChild(typeof child === 'object' ? child : document.createTextNode(String(child)));
  }
  return node;
}

/**
 * Announce a structural form edit (chip/row added or removed) to listeners on
 * an ancestor — native ``input`` events cover typed edits, this covers the
 * rest.
 *
 * @param {HTMLElement} fromEl
 */
function emitChange(fromEl) {
  fromEl.dispatchEvent(new CustomEvent('form-change', { bubbles: true }));
}

/**
 * Resolve a schema node to a concrete one: follow a ``$ref`` into ``$defs``,
 * collapse a single-branch ``allOf`` (Pydantic wraps a defaulted nested model
 * this way), and unwrap the non-null branch of an ``anyOf`` (how ``Optional``
 * is emitted). Returns the resolved node merged with any sibling keys (e.g. a
 * ``title``/``default`` that sits next to the ``$ref``).
 *
 * @param {JsonSchemaNode} root  The full schema document (holds ``$defs``).
 * @param {JsonSchemaNode} node
 * @returns {JsonSchemaNode}
 */
export function resolveNode(root, node) {
  if (!node || typeof node !== 'object') return node;

  if (typeof node.$ref === 'string') {
    const target = derefPointer(root, node.$ref);
    if (target) {
      // Merge sibling keys (a title/default next to the $ref) over the target,
      // dropping the now-resolved $ref itself.
      const rest = { ...node };
      delete rest.$ref;
      return resolveNode(root, { ...target, ...rest });
    }
  }

  if (Array.isArray(node.allOf) && node.allOf.length === 1) {
    const rest = { ...node };
    delete rest.allOf;
    return resolveNode(root, { ...resolveNode(root, node.allOf[0]), ...rest });
  }

  if (Array.isArray(node.anyOf)) {
    const branches = node.anyOf.filter((b) => !(b && b.type === 'null'));
    if (branches.length === 1) {
      const rest = { ...node };
      delete rest.anyOf;
      return resolveNode(root, { ...resolveNode(root, branches[0]), ...rest });
    }
  }

  return node;
}

/**
 * Follow a local JSON-Pointer ``$ref`` (``#/$defs/Name``) into ``root``.
 *
 * @param {JsonSchemaNode} root
 * @param {string} ref
 * @returns {JsonSchemaNode|null}
 */
function derefPointer(root, ref) {
  if (!ref.startsWith('#/')) return null;
  let cursor = /** @type {any} */ (root);
  for (const rawPart of ref.slice(2).split('/')) {
    const part = rawPart.replace(/~1/g, '/').replace(/~0/g, '~');
    if (cursor && typeof cursor === 'object' && part in cursor) {
      cursor = cursor[part];
    } else {
      return null;
    }
  }
  return cursor;
}

/**
 * The effective JSON-Schema type of a resolved node, inferring ``object`` /
 * ``array`` from the presence of ``properties`` / ``items`` when ``type`` is
 * absent.
 *
 * @param {JsonSchemaNode} node
 * @returns {string}
 */
function effectiveType(node) {
  if (typeof node.type === 'string') return node.type;
  if (node.properties) return 'object';
  if (node.items) return 'array';
  return 'string';
}

/**
 * True when a resolved node renders as a single compact control (fits a grid
 * cell / table cell): scalar types and enums.
 *
 * @param {JsonSchemaNode} node
 * @returns {boolean}
 */
function isScalar(node) {
  if (Array.isArray(node.enum)) return true;
  const t = effectiveType(node);
  return t === 'string' || t === 'integer' || t === 'number' || t === 'boolean';
}

/**
 * True when a resolved node is an object whose every property is scalar —
 * i.e. it can render as one table row.
 *
 * @param {JsonSchemaNode} root
 * @param {JsonSchemaNode} node
 * @returns {boolean}
 */
function isFlatObject(root, node) {
  if (effectiveType(node) !== 'object') return false;
  const properties = node.properties || {};
  const names = Object.keys(properties);
  if (names.length === 0) return false;
  return names.every((name) => isScalar(resolveNode(root, properties[name])));
}

/**
 * Parse one raw string into a scalar of the given schema type. Returns
 * ``null`` when the text isn't a valid value (fractional input on an integer
 * field, non-numeric text on a number field).
 *
 * @param {string} raw
 * @param {string} type
 * @returns {{value: unknown}|null}
 */
function parseScalar(raw, type) {
  if (type === 'integer') {
    const value = Number(raw);
    // Reject fractional/garbage rather than silently truncating
    // (parseInt('3.7', 10) === 3) on a path that ends in a write.
    return Number.isInteger(value) ? { value } : null;
  }
  if (type === 'number') {
    const value = Number(raw);
    return Number.isNaN(value) ? null : { value };
  }
  return { value: raw };
}

/**
 * Parse one list input's raw text into scalar values: split on whitespace and
 * commas (so a pasted list of names commits as many values), parse each token
 * per the item type, all-or-nothing. Returns ``null`` if any token is invalid,
 * so the caller can leave the whole text visibly unaccepted rather than
 * silently drop part of it. Shared by the chip and channel-list editors.
 *
 * @param {string} raw
 * @param {string} itemType
 * @returns {unknown[]|null}
 */
function parseValueList(raw, itemType) {
  const parts = raw.trim().split(/[\s,]+/).filter(Boolean);
  /** @type {unknown[]} */
  const accepted = [];
  for (const part of parts) {
    const parsed = parseScalar(part, itemType);
    if (!parsed) return null;
    accepted.push(parsed.value);
  }
  return accepted;
}

/**
 * Build one control (no label) for a schema node, seeded with ``value`` (or
 * the node's ``default`` when ``value`` is undefined).
 *
 * @param {JsonSchemaNode} root
 * @param {JsonSchemaNode} rawNode
 * @param {unknown} value
 * @returns {Field}
 */
function buildControl(root, rawNode, value) {
  const node = resolveNode(root, rawNode);
  const seed = value === undefined ? node.default : value;

  // A plan may opt one field into a purpose-built control via an `x-widget`
  // hint in its schema (Pydantic `json_schema_extra`) — a presentation
  // refinement layered over the type-driven default below, never a gate: an
  // unknown or type-mismatched hint just falls through to the generic control.
  const widget = node['x-widget'];

  if (Array.isArray(node.enum)) {
    if (widget === 'segmented') return buildSegmented(node, seed);
    return buildEnum(node, seed);
  }

  switch (effectiveType(node)) {
    case 'boolean':
      return buildBoolean(seed);
    case 'integer':
      return buildNumber(node, seed, true);
    case 'number':
      return buildNumber(node, seed, false);
    case 'array': {
      const item = node.items ? resolveNode(root, node.items) : { type: 'string' };
      if (isScalar(item)) {
        if (widget === 'channel-list') return buildChannelList(node, item, seed);
        return buildChips(node, item, seed);
      }
      if (isFlatObject(root, item)) return buildTable(root, node, item, seed);
      return buildArrayRows(root, node, item, seed);
    }
    case 'object':
      return buildObject(root, node, seed);
    default:
      return buildString(node, seed);
  }
}

/**
 * @param {JsonSchemaNode} node
 * @param {unknown} seed
 * @returns {Field}
 */
function buildString(node, seed) {
  const input = /** @type {HTMLInputElement} */ (
    h('input', {
      class: 'field-input',
      type: 'text',
      'aria-label': node.title || 'value',
      value: seed === undefined || seed === null ? '' : String(seed),
    })
  );
  return {
    el: input,
    collect: () => (input.value === '' ? OMIT : input.value),
    setValue: (value) => {
      input.value = value === undefined || value === null ? '' : String(value);
    },
  };
}

/**
 * @param {JsonSchemaNode} node
 * @param {unknown} seed
 * @param {boolean} integer
 * @returns {Field}
 */
function buildNumber(node, seed, integer) {
  /** @type {Record<string, unknown>} */
  const props = {
    class: 'field-input',
    type: 'number',
    step: integer ? '1' : 'any',
    'aria-label': node.title || 'value',
    value: seed === undefined || seed === null ? '' : String(seed),
  };
  // Surface the schema's bounds as native input constraints.
  if (typeof node.minimum === 'number') props.min = node.minimum;
  else if (typeof node.exclusiveMinimum === 'number') props.min = node.exclusiveMinimum;
  if (typeof node.maximum === 'number') props.max = node.maximum;
  else if (typeof node.exclusiveMaximum === 'number') props.max = node.exclusiveMaximum;

  const input = /** @type {HTMLInputElement} */ (h('input', props));
  return {
    el: input,
    collect: () => {
      if (input.value === '') return OMIT;
      const parsed = parseScalar(input.value, integer ? 'integer' : 'number');
      return parsed ? parsed.value : OMIT;
    },
    setValue: (value) => {
      input.value = value === undefined || value === null ? '' : String(value);
    },
  };
}

/**
 * Boolean → a toggle switch (a visually-hidden native checkbox driving a
 * styled track, so keyboard/AT semantics stay native).
 *
 * @param {unknown} seed
 * @returns {Field}
 */
function buildBoolean(seed) {
  // `seed` already folds in the schema default (buildControl resolves an
  // undefined `value` to `node.default` before calling in), so the
  // constructor's own resolved state *is* the schema default — capture it so
  // setValue(undefined) can reset to it instead of hard-resetting to false.
  const defaultChecked = Boolean(seed);
  const input = /** @type {HTMLInputElement} */ (
    h('input', { class: 'switch-input', type: 'checkbox', checked: defaultChecked })
  );
  const el = h(
    'label',
    { class: 'switch' },
    input,
    h('span', { class: 'switch-track' }, h('span', { class: 'switch-thumb' }))
  );
  // A checkbox always has a definite state, so it always contributes.
  return {
    el,
    collect: () => input.checked,
    // `undefined` means "this key was absent from the applied value" (e.g. a
    // partial nested-object apply) — reset to the constructor-seeded default
    // rather than coercing straight to false; any other value is a real
    // boolean the caller intends.
    setValue: (value) => {
      input.checked = value === undefined ? defaultChecked : Boolean(value);
    },
  };
}

/**
 * @param {JsonSchemaNode} node
 * @param {unknown} seed
 * @returns {Field}
 */
function buildEnum(node, seed) {
  const members = node.enum || [];
  // The option actually selected at construction time: the seed's matching
  // member, or (mirroring plain HTML `<select>` semantics when no `<option>`
  // carries `selected`) the first member. This is the "constructor-seed
  // state" that setValue falls back to below, so a cleared/non-matching
  // apply lands on a reachable option instead of the blank `selectedIndex
  // -1` state a bare `select.value = ''` would produce.
  const matchedSeed = members.find((opt) => seed !== undefined && opt === seed);
  const initialSelected = matchedSeed === undefined ? members[0] : matchedSeed;
  const options = members.map((opt) =>
    h('option', { value: String(opt), selected: opt === initialSelected }, String(opt))
  );
  const select = /** @type {HTMLSelectElement} */ (
    h('select', { class: 'field-input', 'aria-label': node.title || 'value' }, ...options)
  );
  return {
    el: select,
    collect: () => {
      const chosen = members.find((opt) => String(opt) === select.value);
      return chosen === undefined ? OMIT : chosen;
    },
    // `undefined` (key absent from an applied value) or a value matching no
    // member both fall back to `initialSelected` — the same constructor-seed
    // state buildSegmented's setValue uses — rather than the unreachable
    // blank `<select>` state.
    setValue: (value) => {
      let chosen = initialSelected;
      if (value !== undefined) {
        const matched = members.find((opt) => String(opt) === String(value));
        if (matched !== undefined) chosen = matched;
      }
      select.value = String(chosen);
    },
  };
}

/**
 * Shared engine for the two array-of-scalars list editors (the chip well and
 * the channel list). They differ only in DOM shape, so everything else lives
 * here: the parsed, all-or-nothing commit (Enter/comma/blur, with
 * whitespace/comma-separated pastes committing many at once and invalid text
 * left visibly in the input), Backspace-on-empty removal of the last value,
 * the ``OMIT``-or-array ``collect``, and the whole-value-replacement
 * ``setValue``. The caller owns the DOM: it passes the add-input, the root
 * element to mount, and a ``render(values)`` that paints the current values
 * into its own structure — called once at build time and after every mutation
 * (commit, Backspace, chip/row removal, ``setValue``).
 *
 * @param {JsonSchemaNode} itemSchema The resolved item schema (scalar).
 * @param {unknown} seed
 * @param {HTMLInputElement} input The add-input, owned by the caller.
 * @param {HTMLElement} el The root element to mount, owned by the caller.
 * @param {(values: unknown[]) => void} render Paint ``values`` into the DOM.
 * @returns {Field}
 */
function buildCommitList(itemSchema, seed, input, el, render) {
  const itemType = Array.isArray(itemSchema.enum) ? 'string' : effectiveType(itemSchema);
  /** @type {unknown[]} */
  const values = Array.isArray(seed) ? seed.slice() : [];

  /** Commit the input text as values; all-or-nothing (see parseValueList). */
  function commit() {
    const raw = input.value.trim();
    if (!raw) return;
    const accepted = parseValueList(raw, itemType);
    if (accepted === null) return; // leave the text in place — visibly not accepted
    values.push(...accepted);
    input.value = '';
    render(values);
    input.focus();
    emitChange(el);
  }

  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ',') {
      event.preventDefault();
      commit();
    } else if (event.key === 'Backspace' && input.value === '' && values.length > 0) {
      values.pop();
      render(values);
      input.focus();
      emitChange(el);
    }
  });
  input.addEventListener('blur', () => commit());

  render(values);

  return {
    el,
    collect: () => (values.length > 0 ? values.slice() : OMIT),
    // Whole-value replacement: an incoming array is a full new list, not a
    // per-item patch — a per-item registry would go stale across add/remove
    // edits, so this rebuilds `values` from scratch and re-renders. Also
    // discards any not-yet-committed text in the add-input: leaving it would
    // let a later blur commit stale text on top of the applied values.
    setValue: (value) => {
      values.length = 0;
      if (Array.isArray(value)) values.push(...value);
      input.value = '';
      render(values);
    },
  };
}

/**
 * Array of scalars → chip editor. Typing a value and pressing Enter (or
 * comma, or leaving the field) commits it as a removable chip; multiple
 * whitespace/comma-separated values paste in as multiple chips; Backspace on
 * an empty input removes the last chip. Invalid text (per the item type) is
 * left in the input rather than silently dropped.
 *
 * @param {JsonSchemaNode} node       The (resolved) array node.
 * @param {JsonSchemaNode} itemSchema The resolved item schema (scalar).
 * @param {unknown} seed
 * @returns {Field}
 */
function buildChips(node, itemSchema, seed) {
  const input = /** @type {HTMLInputElement} */ (
    h('input', {
      class: 'chips-input',
      type: 'text',
      placeholder: '+ type, press Enter',
      'aria-label': `add ${node.title || 'value'}`,
      autocomplete: 'off',
      spellcheck: false,
    })
  );
  const el = h('div', { class: 'chips' }, input);

  // Paint `values` into the chip well — a removable chip each, add-input last —
  // rebuilding the whole well every call.
  /** @param {unknown[]} values */
  function render(values) {
    el.replaceChildren();
    values.forEach((value, index) => {
      const remove = h('button', {
        type: 'button',
        class: 'chip-x',
        'aria-label': `remove ${String(value)}`,
        text: '×',
      });
      remove.addEventListener('click', () => {
        values.splice(index, 1);
        render(values);
        emitChange(el);
      });
      el.appendChild(h('span', { class: 'chip' }, h('span', { text: String(value) }), remove));
    });
    el.appendChild(input);
  }

  // Clicking anywhere in the chip well focuses the input, like a real tag box.
  el.addEventListener('click', (event) => {
    if (event.target === el) input.focus();
  });

  return buildCommitList(itemSchema, seed, input, el, render);
}

/**
 * Array of scalars → a vertical channel list (opt-in via ``x-widget:
 * "channel-list"``). Unlike the chip editor's wrapping well, entries stack one
 * per row inside a fixed-height scroll region with a live count header — built
 * for long instrument lists (tens of correctors/BPMs) that would otherwise
 * cram into an unreadable chip row. Editing is the chip editor's: type a value
 * and Enter/comma/paste commits it (whitespace/comma-separated pastes add many
 * at once, all-or-nothing per the item type), Backspace on an empty input
 * removes the last entry, and each row's × removes that entry.
 *
 * @param {JsonSchemaNode} node       The (resolved) array node.
 * @param {JsonSchemaNode} itemSchema The resolved item schema (scalar).
 * @param {unknown} seed
 * @returns {Field}
 */
function buildChannelList(node, itemSchema, seed) {
  const count = h('span', { class: 'channel-count' });
  const head = h('div', { class: 'channel-list-head' }, count);
  const list = h('ul', { class: 'channel-items', role: 'list' });
  const input = /** @type {HTMLInputElement} */ (
    h('input', {
      class: 'channel-add',
      type: 'text',
      placeholder: '+ add channel, press Enter',
      'aria-label': `add ${node.title || 'channel'}`,
      autocomplete: 'off',
      spellcheck: false,
    })
  );
  const el = h('div', { class: 'channel-list' }, head, list, input);

  // Paint `values` into the vertical list — one removable row each — and refresh
  // the live count header. Only the <ul> is rebuilt; the head/input persist.
  /** @param {unknown[]} values */
  function render(values) {
    list.replaceChildren();
    values.forEach((value, index) => {
      const remove = h('button', {
        type: 'button',
        class: 'chan-x',
        'aria-label': `remove ${String(value)}`,
        text: '×',
      });
      remove.addEventListener('click', () => {
        values.splice(index, 1);
        render(values);
        emitChange(el);
      });
      list.appendChild(
        h(
          'li',
          { class: 'channel-item' },
          h('span', { class: 'channel-name', text: String(value) }),
          remove
        )
      );
    });
    const n = values.length;
    count.textContent = `${n} channel${n === 1 ? '' : 's'}`;
  }

  return buildCommitList(itemSchema, seed, input, el, render);
}

/**
 * A small enum → a segmented control (opt-in via ``x-widget: "segmented"``):
 * every option is a visible, mutually-exclusive button, so a two-way choice
 * (e.g. ``bidirectional`` / ``monodirectional``) reads as a labeled toggle
 * rather than a collapsed ``<select>``. Native radiogroup semantics
 * (``role="radiogroup"`` + ``role="radio"``/``aria-checked``) keep it
 * keyboard/AT-legible. Always contributes its active value (a segmented
 * control has no blank state) — the seed's option when the seed matches one,
 * else the first.
 *
 * @param {JsonSchemaNode} node
 * @param {unknown} seed
 * @returns {Field}
 */
function buildSegmented(node, seed) {
  const options = (node.enum || []).map((opt) => String(opt));
  // The option actually active at construction time (the seed's matching
  // option, else the first) — the "constructor-seed state" setValue falls
  // back to below, both for an absent value and a non-matching one.
  const initialSelected = seed !== undefined && options.includes(String(seed)) ? String(seed) : options[0];
  let selected = initialSelected;

  const el = h('div', {
    class: 'segmented',
    role: 'radiogroup',
    'aria-label': node.title || 'value',
  });
  /** @type {HTMLElement[]} */
  const buttons = [];
  for (const opt of options) {
    const btn = h('button', {
      type: 'button',
      class: `segmented-option${opt === selected ? ' active' : ''}`,
      role: 'radio',
      'aria-checked': opt === selected ? 'true' : 'false',
      text: opt,
    });
    btn.addEventListener('click', () => {
      if (selected === opt) return;
      selected = opt;
      for (const other of buttons) {
        const on = other === btn;
        other.classList.toggle('active', on);
        other.setAttribute('aria-checked', on ? 'true' : 'false');
      }
      emitChange(el);
    });
    buttons.push(btn);
    el.appendChild(btn);
  }

  return {
    el,
    // Map the active label back to the enum's original (possibly non-string)
    // member, mirroring buildEnum's collect.
    collect: () => {
      const chosen = (node.enum || []).find((opt) => String(opt) === selected);
      return chosen === undefined ? OMIT : chosen;
    },
    // Always contributes (see module note above), so setValue always selects
    // a concrete option: the matching member, or `initialSelected` (the
    // constructor-seed state) when `value` is absent or doesn't match any —
    // an absent key must reset to the schema default, not hard-fall to the
    // first option regardless of what was originally seeded. Updates the
    // buttons' active state directly rather than via `.click()`, so no
    // click/emitChange fires.
    setValue: (value) => {
      selected = options.includes(String(value)) ? String(value) : initialSelected;
      buttons.forEach((btn, index) => {
        const on = options[index] === selected;
        btn.classList.toggle('active', on);
        btn.setAttribute('aria-checked', on ? 'true' : 'false');
      });
    },
  };
}

/**
 * Array of flat objects → an editable table: one column per item field, one
 * row per item, a remove control per row, and an add-row button. This is the
 * 2-D editor ``grid_scan``'s ``axes`` renders as.
 *
 * @param {JsonSchemaNode} root
 * @param {JsonSchemaNode} node       The (resolved) array node.
 * @param {JsonSchemaNode} itemSchema The resolved flat-object item schema.
 * @param {unknown} seed
 * @returns {Field}
 */
function buildTable(root, node, itemSchema, seed) {
  const properties = itemSchema.properties || {};
  const columnNames = Object.keys(properties);
  const required = new Set(itemSchema.required || []);

  const headRow = h(
    'tr',
    undefined,
    ...columnNames.map((name) => {
      const prop = resolveNode(root, properties[name]);
      return h(
        'th',
        { scope: 'col' },
        h('span', { text: prop.title || name }),
        required.has(name) ? h('span', { class: 'field-required', text: '*' }) : null
      );
    }),
    h('th', { class: 'th-x', 'aria-label': 'remove row' })
  );
  const tbody = h('tbody');
  const table = h('table', { class: 'obj-table' }, h('thead', undefined, headRow), tbody);

  /** @type {Array<{tr: HTMLElement, cells: Array<{name: string, collect: () => unknown}>}>} */
  const rows = [];

  /** @param {any} itemValue */
  function addRow(itemValue) {
    /** @type {Array<{name: string, collect: () => unknown}>} */
    const cells = [];
    const tds = columnNames.map((name) => {
      const control = buildControl(
        root,
        properties[name],
        itemValue && typeof itemValue === 'object' ? itemValue[name] : undefined
      );
      cells.push({ name, collect: control.collect });
      return h('td', undefined, control.el);
    });
    const removeBtn = h('button', {
      type: 'button',
      class: 'row-x',
      'aria-label': 'remove row',
      text: '×',
    });
    const tr = h('tr', undefined, ...tds, h('td', { class: 'td-x' }, removeBtn));
    const entry = { tr, cells };
    removeBtn.addEventListener('click', () => {
      const idx = rows.indexOf(entry);
      if (idx !== -1) rows.splice(idx, 1);
      tr.remove();
      emitChange(el);
    });
    rows.push(entry);
    tbody.appendChild(tr);
  }

  /** Remove every row (used by setValue for whole-table replacement). */
  function clearRows() {
    for (const row of rows) row.tr.remove();
    rows.length = 0;
  }

  const addBtn = h('button', {
    type: 'button',
    class: 'table-add',
    text: `+ ${itemSchema.title || 'row'}`,
  });
  addBtn.addEventListener('click', () => {
    addRow(undefined);
    emitChange(el);
  });

  const el = h('div', { class: 'table-field' }, table, addBtn);

  if (Array.isArray(seed) && seed.length > 0) {
    for (const item of seed) addRow(item);
  } else if ((node.minItems || 0) >= 1) {
    // A required list starts with one blank row so the columns are visible
    // immediately (the row collects to nothing until it's filled in).
    addRow(undefined);
  }

  return {
    el,
    collect: () => {
      /** @type {unknown[]} */
      const out = [];
      for (const row of rows) {
        /** @type {Record<string, unknown>} */
        const item = {};
        for (const cell of row.cells) {
          const v = cell.collect();
          if (v !== OMIT) item[cell.name] = v;
        }
        // A fully-blank row is scaffolding, not data.
        if (Object.keys(item).length > 0) out.push(item);
      }
      return out.length > 0 ? out : OMIT;
    },
    // Whole-table replacement: clear every row and rebuild from `value`,
    // mirroring the constructor's own seeding rule (one blank starter row for
    // a required table left empty) — a static per-row registry would go
    // stale the moment a row is added or removed.
    setValue: (value) => {
      clearRows();
      if (Array.isArray(value) && value.length > 0) {
        for (const item of value) addRow(item);
      } else if ((node.minItems || 0) >= 1) {
        addRow(undefined);
      }
    },
  };
}

/**
 * Generic fallback for arrays of non-flat items: a stack of removable rows,
 * each row built from the item schema.
 *
 * @param {JsonSchemaNode} root
 * @param {JsonSchemaNode} node
 * @param {JsonSchemaNode} itemSchema
 * @param {unknown} seed
 * @returns {Field}
 */
function buildArrayRows(root, node, itemSchema, seed) {
  const rowsEl = h('div', { class: 'array-rows' });
  /** @type {Array<{rowEl: HTMLElement, collect: () => unknown}>} */
  const rows = [];

  /** @param {unknown} itemValue */
  function addRow(itemValue) {
    const field = buildControl(root, itemSchema, itemValue);
    const removeBtn = h('button', {
      type: 'button',
      class: 'row-x',
      'aria-label': 'remove item',
      text: '×',
    });
    const rowEl = h(
      'div',
      { class: 'array-row' },
      h('div', { class: 'array-row-body' }, field.el),
      removeBtn
    );
    const entry = { rowEl, collect: field.collect };
    removeBtn.addEventListener('click', () => {
      const idx = rows.indexOf(entry);
      if (idx !== -1) rows.splice(idx, 1);
      rowEl.remove();
      emitChange(el);
    });
    rows.push(entry);
    rowsEl.appendChild(rowEl);
  }

  /** Remove every row (used by setValue for whole-array replacement). */
  function clearRows() {
    for (const row of rows) row.rowEl.remove();
    rows.length = 0;
  }

  const addBtn = h('button', {
    type: 'button',
    class: 'table-add',
    text: `+ ${itemSchema.title || 'item'}`,
  });
  addBtn.addEventListener('click', () => {
    addRow(undefined);
    emitChange(el);
  });

  const el = h('div', { class: 'array-field' }, rowsEl, addBtn);
  if (Array.isArray(seed)) for (const item of seed) addRow(item);

  return {
    el,
    collect: () => {
      const out = rows.map((row) => row.collect()).filter((v) => v !== OMIT);
      return out.length > 0 ? out : OMIT;
    },
    // Whole-array replacement, same rationale as buildTable.setValue.
    setValue: (value) => {
      clearRows();
      if (Array.isArray(value)) for (const item of value) addRow(item);
    },
  };
}

/**
 * A nested (non-array) object: labeled sub-rows inside a bordered group.
 *
 * @param {JsonSchemaNode} root
 * @param {JsonSchemaNode} node
 * @param {unknown} seed
 * @returns {Field}
 */
function buildObject(root, node, seed) {
  const properties = node.properties || {};
  const required = new Set(node.required || []);
  const seedObj = seed && typeof seed === 'object' ? /** @type {any} */ (seed) : {};

  const el = h('div', { class: 'object-field' });
  /** @type {Array<{name: string, collect: () => unknown, setValue: (value: unknown) => void}>} */
  const children = [];

  for (const [name, childNode] of Object.entries(properties)) {
    const labeled = buildLabeledField(root, name, childNode, required.has(name), seedObj[name]);
    children.push({ name, collect: labeled.collect, setValue: labeled.setValue });
    el.appendChild(labeled.el);
  }

  return {
    el,
    collect: () => {
      /** @type {Record<string, unknown>} */
      const out = {};
      for (const child of children) {
        const v = child.collect();
        if (v !== OMIT) out[child.name] = v;
      }
      return Object.keys(out).length === 0 ? OMIT : out;
    },
    // The object's shape (its child names) is static, so whole-value
    // replacement composes recursively — each child's own setValue is
    // authoritative for its slice of `value` (or clears to blank/default
    // when `value` doesn't carry that key), rather than this rebuilding the
    // sub-fields from scratch.
    setValue: (value) => {
      const obj = value && typeof value === 'object' ? /** @type {any} */ (value) : {};
      for (const child of children) child.setValue(obj[child.name]);
    },
  };
}

/**
 * Wrap a control in a labeled block: field name (with a required marker) and,
 * when present, the schema ``description`` as help text under the label.
 *
 * @param {JsonSchemaNode} root
 * @param {string} name
 * @param {JsonSchemaNode} rawNode
 * @param {boolean} required
 * @param {unknown} value
 * @returns {{ el: HTMLElement, collect: () => unknown, setValue: (value: unknown) => void }}
 */
function buildLabeledField(root, name, rawNode, required, value) {
  const node = resolveNode(root, rawNode);
  const control = buildControl(root, node, value);

  const row = h(
    'div',
    { class: 'param-row' },
    h(
      'div',
      { class: 'field-label' },
      h('span', { class: 'field-name', text: node.title || name }),
      required ? h('span', { class: 'field-required', title: 'required', text: '*' }) : null
    )
  );
  if (node.description) {
    row.appendChild(h('p', { class: 'field-help', text: node.description }));
  }
  row.appendChild(h('div', { class: 'field-control' }, control.el));

  return { el: row, collect: control.collect, setValue: control.setValue };
}

/**
 * Attach ``applyValues``/``fields`` onto a bare ``collect`` function, turning
 * it into the ``PlanArgsCollector`` this module returns — shared by the
 * empty-schema early return and the full render path so both apply the exact
 * same casting and event-dispatch logic. ``applyValues`` sets every field
 * present in ``values`` (a key absent from ``values`` leaves that field
 * untouched) via that field's ``setValue`` — which never fires native
 * ``input``/``change`` events — then dispatches exactly one ``form-change``
 * for the whole pass, documented here as programmatic so a host (and the
 * pending-key rule a caller layers on top) can tell it apart from a user
 * edit. A missing/non-object ``values`` (``undefined``, ``null``, a scalar)
 * is a safe no-op — no field is touched and no ``form-change`` fires — rather
 * than throwing on ``hasOwnProperty`` or silently emitting an empty pass.
 *
 * @param {() => Record<string, unknown>} collect
 * @param {Record<string, RegisteredField>} fieldRegistry
 * @param {HTMLElement} formEl
 * @returns {PlanArgsCollector}
 */
function withFieldRegistry(collect, fieldRegistry, formEl) {
  const collector = /** @type {PlanArgsCollector} */ (/** @type {any} */ (collect));
  collector.fields = fieldRegistry;
  collector.applyValues = (values) => {
    if (!values || typeof values !== 'object') return;
    const obj = /** @type {Record<string, unknown>} */ (values);
    for (const [name, field] of Object.entries(fieldRegistry)) {
      if (Object.prototype.hasOwnProperty.call(obj, name)) {
        field.setValue(obj[name]);
      }
    }
    emitChange(formEl);
  };
  return collector;
}

/**
 * Render a plan's top-level parameter schema into ``formEl`` as a 2-D grid of
 * labeled fields and return a ``collect()`` that reads the whole form into a
 * nested ``plan_args`` object. Fields left blank are omitted so plan-side
 * defaults apply.
 *
 * ``opts.layout`` — rows of field names — pins fields into side-by-side
 * columns (``[['correctors','detectors'], ['span_a','num']]``). Names not in
 * the schema are ignored; schema fields not in the layout are auto-flowed
 * after it (scalars packed up to three per row, wide editors full-width).
 *
 * The returned collector also carries ``applyValues(values)`` (apply a
 * server-held draft's values programmatically, one field per top-level
 * schema property) and ``fields`` (that same top-level field registry, keyed
 * by name) — see ``withFieldRegistry`` and the module docstring.
 *
 * @param {HTMLElement} formEl  The (emptied-and-refilled) form container.
 * @param {JsonSchemaNode|undefined} schema  A plan's ``model_json_schema()``.
 * @param {{layout?: string[][]}} [opts]
 * @returns {PlanArgsCollector}
 */
export function renderSchemaForm(formEl, schema, opts) {
  formEl.replaceChildren();

  const root = schema || {};
  const properties = root.properties || {};
  const names = Object.keys(properties);

  if (names.length === 0) {
    formEl.appendChild(h('p', { class: 'param-empty', text: 'This plan takes no parameters.' }));
    return withFieldRegistry(() => ({}), {}, formEl);
  }

  const required = new Set(root.required || []);

  // Resolve the row plan: explicit layout rows first, then auto-flow.
  /** @type {string[][]} */
  const rowPlan = [];
  const placed = new Set();
  const layout = opts && Array.isArray(opts.layout) ? opts.layout : null;
  if (layout) {
    for (const layoutRow of layout) {
      const rowNames = layoutRow.filter((n) => n in properties && !placed.has(n));
      if (rowNames.length > 0) {
        rowPlan.push(rowNames);
        for (const n of rowNames) placed.add(n);
      }
    }
  }
  /** @type {string[]} */
  let scalarBatch = [];
  const flush = () => {
    if (scalarBatch.length > 0) {
      rowPlan.push(scalarBatch);
      scalarBatch = [];
    }
  };
  for (const name of names) {
    if (placed.has(name)) continue;
    if (isScalar(resolveNode(root, properties[name]))) {
      scalarBatch.push(name);
      if (scalarBatch.length === 3) flush();
    } else {
      flush();
      rowPlan.push([name]);
    }
  }
  flush();

  /** @type {Array<{ name: string, collect: () => unknown }>} */
  const fields = [];
  /** @type {Record<string, RegisteredField>} */
  const fieldRegistry = {};
  for (const rowNames of rowPlan) {
    const rowEl = h('div', { class: 'form-row' });
    rowEl.style.setProperty('--cols', String(rowNames.length));
    for (const name of rowNames) {
      const labeled = buildLabeledField(root, name, properties[name], required.has(name), undefined);
      fields.push({ name, collect: labeled.collect });
      fieldRegistry[name] = { el: labeled.el, setValue: labeled.setValue };
      rowEl.appendChild(labeled.el);
    }
    formEl.appendChild(rowEl);
  }

  return withFieldRegistry(
    () => {
      /** @type {Record<string, unknown>} */
      const args = {};
      for (const field of fields) {
        const v = field.collect();
        if (v !== OMIT) args[field.name] = v;
      }
      return args;
    },
    fieldRegistry,
    formEl
  );
}
