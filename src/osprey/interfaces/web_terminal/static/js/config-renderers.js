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
// .mcp.json renderer — progressive enhancement with tool introspection
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

  // Phase 1: Render basic cards from raw JSON (instant)
  const cardMap = {};
  for (const [name, spec] of Object.entries(servers)) {
    const card = _mcpCard(name, spec);
    cardMap[name] = card;
    grid.appendChild(card);
  }

  section.appendChild(grid);
  container.appendChild(section);

  // Phase 2: Fetch enriched data and update cards in-place
  _fetchAndEnrichCards(cardMap);

  return container;
}


/**
 * Build a single MCP server card with header, module path, and env vars.
 * Returns the card element. The tools area is a placeholder for enrichment.
 */
function _mcpCard(name, spec) {
  const card = _el('div', 'config-mcp-card');

  // Header: category dot + server name + tool count placeholder
  const header = _el('div', 'config-mcp-card-header');

  const categoryDot = _el('span', 'config-mcp-category-dot');
  const isOsprey = (spec.args || []).some(a => typeof a === 'string' && a.startsWith('osprey.'));
  categoryDot.classList.add(isOsprey ? 'config-mcp-category-osprey' : 'config-mcp-category-external');
  categoryDot.title = isOsprey ? 'OSPREY server' : 'External server';
  header.appendChild(categoryDot);

  const nameEl = _el('span', 'config-mcp-card-name');
  nameEl.textContent = name;
  header.appendChild(nameEl);

  const countBadge = _el('span', 'config-mcp-tool-count');
  countBadge.dataset.serverName = name;
  countBadge.style.display = 'none';
  header.appendChild(countBadge);

  card.appendChild(header);

  // Server description (populated by enrichment)
  const descEl = _el('div', 'config-mcp-card-desc');
  descEl.dataset.serverName = name;
  card.appendChild(descEl);

  // Tools area (loading placeholder, replaced by enrichment)
  const toolsArea = _el('div', 'config-mcp-tools');
  toolsArea.dataset.serverName = name;

  const loadingEl = _el('span', 'config-mcp-tools-loading');
  loadingEl.textContent = 'discovering tools\u2026';
  toolsArea.appendChild(loadingEl);

  card.appendChild(toolsArea);

  // Config section (collapsed by default)
  const configToggle = _el('button', 'config-mcp-config-toggle');
  configToggle.textContent = '\u25B6 config';
  const configBody = _el('div', 'config-mcp-config-body');
  configBody.style.display = 'none';

  configToggle.addEventListener('click', () => {
    const expanded = configBody.style.display !== 'none';
    configBody.style.display = expanded ? 'none' : 'block';
    configToggle.textContent = (expanded ? '\u25B6' : '\u25BC') + ' config';
  });

  // Module path
  if (spec.args && spec.args.length > 0) {
    const moduleArg = spec.args.find(a => typeof a === 'string' && a !== '-m');
    if (moduleArg) {
      const moduleEl = _el('div', 'config-mcp-module');
      moduleEl.textContent = moduleArg;
      moduleEl.title = `${spec.command} ${spec.args.join(' ')}`;
      configBody.appendChild(moduleEl);
    }
  } else if (spec.command) {
    const cmdEl = _el('div', 'config-mcp-module');
    cmdEl.textContent = spec.command;
    configBody.appendChild(cmdEl);
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
    configBody.appendChild(envList);
  }

  card.appendChild(configToggle);
  card.appendChild(configBody);

  return card;
}


/**
 * Remove all children from an element (safe alternative to innerHTML = '').
 */
function _clearChildren(el) {
  while (el.firstChild) {
    el.removeChild(el.firstChild);
  }
}


/**
 * Fetch /api/mcp-servers and update cards with tool lists and descriptions.
 */
function _fetchAndEnrichCards(cardMap) {
  fetch('/api/mcp-servers')
    .then(r => r.ok ? r.json() : Promise.reject(r.status))
    .then(servers => {
      for (const server of servers) {
        const card = cardMap[server.name];
        if (!card) continue;

        // Set server description
        const descEl = card.querySelector('.config-mcp-card-desc');
        if (descEl && server.description) {
          descEl.textContent = server.description;
        }

        // Update tool count badge
        const badge = card.querySelector('.config-mcp-tool-count');
        if (badge && server.tool_count != null) {
          badge.textContent = server.tool_count;
          badge.title = `${server.tool_count} tool${server.tool_count !== 1 ? 's' : ''}`;
          badge.style.display = '';
        }

        // Replace tools area with vertical tool list
        const toolsArea = card.querySelector('.config-mcp-tools');
        if (!toolsArea) continue;
        _clearChildren(toolsArea);

        if (server.tools && server.tools.length > 0) {
          const list = _el('div', 'config-mcp-tool-list');
          for (const tool of server.tools) {
            const item = _el('div', 'config-mcp-tool-item');
            const parsed = _parseToolDescription(tool.description || '');

            // Header row: chevron + name + summary snippet
            const header = _el('div', 'config-mcp-tool-header');
            header.addEventListener('click', () => {
              item.classList.toggle('expanded');
            });

            const chevron = _el('span', 'config-mcp-tool-chevron');
            chevron.textContent = '\u25B6';
            header.appendChild(chevron);

            const nameEl = _el('span', 'config-mcp-tool-name');
            nameEl.textContent = tool.name;
            header.appendChild(nameEl);

            if (parsed.summary) {
              const summaryEl = _el('span', 'config-mcp-tool-summary');
              summaryEl.textContent = _truncate(parsed.summary, 60);
              summaryEl.title = parsed.summary;
              header.appendChild(summaryEl);
            }

            item.appendChild(header);

            // Expandable detail body
            if (parsed.summary || parsed.args.length > 0 || parsed.returns) {
              const body = _el('div', 'config-mcp-tool-body');

              if (parsed.summary) {
                const descBlock = _el('div', 'config-mcp-tool-desc-full');
                descBlock.textContent = parsed.summary;
                body.appendChild(descBlock);
              }

              if (parsed.args.length > 0) {
                const argsSection = _el('div', 'config-mcp-tool-section');
                const argsLabel = _el('div', 'config-mcp-tool-section-label');
                argsLabel.textContent = 'ARGS';
                argsSection.appendChild(argsLabel);

                for (const arg of parsed.args) {
                  const argRow = _el('div', 'config-mcp-tool-arg');
                  const argName = _el('span', 'config-mcp-tool-arg-name');
                  argName.textContent = arg.name;
                  argRow.appendChild(argName);
                  const argDesc = _el('span', 'config-mcp-tool-arg-desc');
                  argDesc.textContent = arg.desc;
                  argRow.appendChild(argDesc);
                  argsSection.appendChild(argRow);
                }
                body.appendChild(argsSection);
              }

              if (parsed.returns) {
                const retSection = _el('div', 'config-mcp-tool-section');
                const retLabel = _el('div', 'config-mcp-tool-section-label');
                retLabel.textContent = 'RETURNS';
                retSection.appendChild(retLabel);
                const retText = _el('div', 'config-mcp-tool-returns');
                retText.textContent = parsed.returns;
                retSection.appendChild(retText);
                body.appendChild(retSection);
              }

              item.appendChild(body);
            }

            list.appendChild(item);
          }
          toolsArea.appendChild(list);
        } else if (server.tools === null) {
          const fallback = _el('span', 'config-mcp-tools-fallback');
          fallback.textContent = 'tools not available';
          toolsArea.appendChild(fallback);
        } else {
          const empty = _el('span', 'config-mcp-tools-fallback');
          empty.textContent = 'no tools';
          toolsArea.appendChild(empty);
        }
      }
    })
    .catch(() => {
      // On failure, remove loading indicators and show fallback
      for (const card of Object.values(cardMap)) {
        const toolsArea = card.querySelector('.config-mcp-tools');
        if (toolsArea) {
          _clearChildren(toolsArea);
          const fallback = _el('span', 'config-mcp-tools-fallback');
          fallback.textContent = 'tools not available';
          toolsArea.appendChild(fallback);
        }
      }
    });
}


/**
 * Parse a Google-style docstring into summary, args, and returns.
 */
function _parseToolDescription(desc) {
  const result = { summary: '', args: [], returns: '' };
  if (!desc) return result;

  const lines = desc.split('\n');
  let section = 'summary';
  const summaryLines = [];
  const argsLines = [];
  const returnsLines = [];

  for (const raw of lines) {
    const trimmed = raw.trim();

    if (/^Args?\s*:?\s*$/i.test(trimmed) || /^Args:/i.test(trimmed)) {
      section = 'args';
      continue;
    }
    if (/^Returns?\s*:?\s*$/i.test(trimmed) || /^Returns:/i.test(trimmed)) {
      section = 'returns';
      continue;
    }
    if (/^Raises?\s*:?\s*$/i.test(trimmed)) {
      section = 'skip';
      continue;
    }

    if (section === 'summary') {
      if (trimmed) summaryLines.push(trimmed);
    } else if (section === 'args') {
      if (trimmed) argsLines.push(raw);
    } else if (section === 'returns') {
      if (trimmed) returnsLines.push(trimmed);
    }
  }

  result.summary = summaryLines.join(' ');
  result.returns = returnsLines.join(' ');

  // Parse args: lines starting with less indentation are names, continuation is description
  let currentArg = null;
  for (const raw of argsLines) {
    const match = raw.trim().match(/^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$/);
    if (match) {
      if (currentArg) result.args.push(currentArg);
      currentArg = { name: match[1], desc: match[2] };
    } else if (currentArg) {
      currentArg.desc += ' ' + raw.trim();
    }
  }
  if (currentArg) result.args.push(currentArg);

  return result;
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
