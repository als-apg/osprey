/* OSPREY Web Terminal — Rich renderers for settings.json and .mcp.json
 *
 * Instead of dumping raw JSON, these renderers parse the content and present
 * structured, scannable views:
 *
 *   settings.json → Environment, Model, Permissions (allow/deny/ask), Hooks
 *   .mcp.json     → Grid of MCP server cards with module, env vars
 */

// ---------------------------------------------------------------------------
// settings.json renderer
// ---------------------------------------------------------------------------

export function renderSettingsJson(jsonString) {
  let data;
  try {
    data = JSON.parse(jsonString);
  } catch {
    return null;
  }

  const container = document.createElement('div');
  container.className = 'config-structured-view';

  // ---- Environment & Model ----
  if (data.env || data.model) {
    const section = _section('Environment');
    const grid = _el('div', 'config-env-grid');

    if (data.model) {
      grid.appendChild(_envRow('Model', data.model));
    }
    if (data.env) {
      for (const [key, value] of Object.entries(data.env)) {
        grid.appendChild(_envRow(_humanizeEnvKey(key), value));
      }
    }
    section.appendChild(grid);
    container.appendChild(section);
  }

  // ---- Permissions ----
  if (data.permissions) {
    const section = _section('Permissions');
    const columns = _el('div', 'config-permissions-columns');

    if (data.permissions.allow) {
      columns.appendChild(_permissionColumn('allow', data.permissions.allow));
    }
    if (data.permissions.ask) {
      columns.appendChild(_permissionColumn('ask', data.permissions.ask));
    }
    if (data.permissions.deny) {
      columns.appendChild(_permissionColumn('deny', data.permissions.deny));
    }

    section.appendChild(columns);
    container.appendChild(section);
  }

  // ---- Hooks ----
  if (data.hooks) {
    const section = _section('Hooks');

    for (const [eventName, hookGroups] of Object.entries(data.hooks)) {
      const eventSection = _el('div', 'config-hook-event');

      const eventHeader = _el('div', 'config-hook-event-header');
      eventHeader.innerHTML = `<span class="config-hook-chevron">\u25B6</span><span>${eventName}</span><span class="config-hook-count">${_countHooks(hookGroups)}</span>`;
      eventHeader.addEventListener('click', () => {
        eventSection.classList.toggle('expanded');
      });
      eventSection.appendChild(eventHeader);

      const eventBody = _el('div', 'config-hook-event-body');
      for (const group of hookGroups) {
        const matcher = group.matcher || '*';
        const matcherEl = _el('div', 'config-hook-matcher');

        const matcherLabel = _el('span', 'config-hook-matcher-label');
        matcherLabel.textContent = matcher;
        matcherEl.appendChild(matcherLabel);

        for (const hook of (group.hooks || [])) {
          const hookEl = _el('div', 'config-hook-entry');
          const cmd = hook.command || '';
          const scriptName = cmd.split('/').pop().replace(/"/g, '').replace(/\.py$/, '');
          hookEl.innerHTML = `<span class="config-hook-script">${_esc(scriptName)}</span>` +
            (hook.timeout ? `<span class="config-hook-timeout">${hook.timeout}s</span>` : '');
          matcherEl.appendChild(hookEl);
        }

        eventBody.appendChild(matcherEl);
      }
      eventSection.appendChild(eventBody);
      section.appendChild(eventSection);
    }

    container.appendChild(section);
  }

  return container;
}


// ---------------------------------------------------------------------------
// .mcp.json renderer
// ---------------------------------------------------------------------------

export function renderMcpJson(jsonString) {
  let data;
  try {
    data = JSON.parse(jsonString);
  } catch {
    return null;
  }

  const servers = data.mcpServers || {};
  if (Object.keys(servers).length === 0) return null;

  const container = document.createElement('div');
  container.className = 'config-structured-view';

  const section = _section(`MCP Servers (${Object.keys(servers).length})`);
  const grid = _el('div', 'config-mcp-grid');

  for (const [name, spec] of Object.entries(servers)) {
    const card = _el('div', 'config-mcp-card');

    // Header with server name
    const header = _el('div', 'config-mcp-card-header');
    const nameEl = _el('span', 'config-mcp-card-name');
    nameEl.textContent = name;
    header.appendChild(nameEl);
    card.appendChild(header);

    // Module path
    if (spec.args && spec.args.length > 0) {
      const moduleArg = spec.args.find(a => typeof a === 'string' && a !== '-m');
      if (moduleArg) {
        const moduleEl = _el('div', 'config-mcp-module');
        moduleEl.textContent = moduleArg;
        moduleEl.title = `${spec.command} ${spec.args.join(' ')}`;
        card.appendChild(moduleEl);
      }
    } else if (spec.command) {
      const cmdEl = _el('div', 'config-mcp-module');
      cmdEl.textContent = spec.command;
      card.appendChild(cmdEl);
    }

    // Environment variables
    if (spec.env && Object.keys(spec.env).length > 0) {
      const envList = _el('div', 'config-mcp-env-list');
      for (const [envKey, envVal] of Object.entries(spec.env)) {
        const row = _el('div', 'config-mcp-env-row');

        const keyEl = _el('span', 'config-mcp-env-key');
        keyEl.textContent = envKey;
        row.appendChild(keyEl);

        const valEl = _el('span', 'config-mcp-env-val');
        const displayVal = String(envVal);
        if (displayVal.startsWith('${') || displayVal.includes('/config.yml')) {
          valEl.classList.add('config-mcp-env-ref');
        }
        valEl.textContent = _truncate(displayVal, 40);
        valEl.title = displayVal;
        row.appendChild(valEl);

        envList.appendChild(row);
      }
      card.appendChild(envList);
    }

    grid.appendChild(card);
  }

  section.appendChild(grid);
  container.appendChild(section);
  return container;
}


// ---------------------------------------------------------------------------
// Permission column builder
// ---------------------------------------------------------------------------

function _permissionColumn(level, entries) {
  const col = _el('div', `config-perm-col config-perm-${level}`);

  const header = _el('div', 'config-perm-header');
  header.textContent = level.toUpperCase();
  col.appendChild(header);

  // Group entries by prefix (mcp server, file path, task, etc.)
  const groups = _groupPermissions(entries);

  for (const [groupName, items] of Object.entries(groups)) {
    if (groupName !== '_ungrouped') {
      const groupLabel = _el('div', 'config-perm-group-label');
      groupLabel.textContent = groupName;
      col.appendChild(groupLabel);
    }

    for (const item of items) {
      const row = _el('div', 'config-perm-entry');
      row.textContent = item.display;
      row.title = item.raw;
      col.appendChild(row);
    }
  }

  return col;
}

function _groupPermissions(entries) {
  const groups = {};
  const addTo = (group, raw, display) => {
    if (!groups[group]) groups[group] = [];
    groups[group].push({ raw, display });
  };

  for (const entry of entries) {
    if (entry.startsWith('mcp__')) {
      const parts = entry.split('__');
      const server = parts[1] || 'unknown';
      const tool = parts.slice(2).join('__') || '*';
      addTo(server, entry, tool);
    } else if (entry.startsWith('Task(')) {
      const agentName = entry.replace(/^Task\(/, '').replace(/\)$/, '');
      addTo('agents', entry, agentName);
    } else if (entry.startsWith('Read(') || entry.startsWith('NotebookEdit(')) {
      const match = entry.match(/^(\w+)\((.+)\)$/);
      if (match) {
        addTo('file access', entry, `${match[1]}: ${match[2]}`);
      } else {
        addTo('_ungrouped', entry, entry);
      }
    } else {
      addTo('_ungrouped', entry, entry);
    }
  }

  return groups;
}


// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function _section(title) {
  const section = _el('div', 'config-section');
  const header = _el('div', 'config-section-header');
  header.textContent = title;
  section.appendChild(header);
  return section;
}

function _envRow(label, value) {
  const row = _el('div', 'config-env-row');
  const labelEl = _el('span', 'config-env-label');
  labelEl.textContent = label;
  const valueEl = _el('span', 'config-env-value');
  valueEl.textContent = value;
  valueEl.title = value;
  row.appendChild(labelEl);
  row.appendChild(valueEl);
  return row;
}

function _humanizeEnvKey(key) {
  return key
    .replace(/^ANTHROPIC_/, '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase());
}

function _countHooks(hookGroups) {
  let count = 0;
  for (const g of hookGroups) {
    count += (g.hooks || []).length;
  }
  return count;
}

function _truncate(str, maxLen) {
  if (str.length <= maxLen) return str;
  return str.substring(0, maxLen - 1) + '\u2026';
}

function _el(tag, className) {
  const el = document.createElement(tag);
  el.className = className;
  return el;
}

function _esc(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
