/* OSPREY Web Terminal — Operator Mode Module (Control-Room Log Console)
 *
 * Manages its own WebSocket connection (does NOT use createWebSocket from api.js
 * to avoid corrupting shared wsState used by the terminal status indicators).
 */

/** @type {WebSocket|null} */
let ws = null;
/** @type {'connected'|'connecting'|'disconnected'} */
let state = 'disconnected';
/** @type {HTMLElement|null} */
let messagesEl = null;
/** @type {HTMLTextAreaElement|null} */
let inputEl = null;
/** @type {HTMLButtonElement|null} */
let sendBtn = null;
/** @type {HTMLButtonElement|null} */
let stopBtn = null;
/** @type {HTMLElement|null} */
let processingEl = null;
/** @type {HTMLElement|null} */
let sessionLed = null;
/** @type {HTMLElement|null} */
let containerEl = null;
/** @type {HTMLElement|null} */
let statsCostEl = null;
/** @type {HTMLElement|null} */
let statsTurnsEl = null;
/** @type {HTMLElement|null} */
let statsDurationEl = null;
/** @type {boolean} */
let streaming = false;
/** @type {HTMLElement|null} */
let currentAssistantGroup = null;

// Running session totals
let totalCost = 0;
let totalTurns = 0;
let totalDuration = 0;

const stateListeners = [];

// Type registry for tool/category colors
let typeRegistry = {};
let artifactServerUrl = null;

// Plotly lazy-loading
let _plotlyLoaded = false;
let _plotlyLoading = null;

const _tsColorway = [
  '#4fd1c5', '#d4a574', '#9f7aea', '#3b82f6',
  '#22c55e', '#f59e0b', '#ef4444', '#e879f9',
];

// Map from tool_use_id → raw tool name, so tool_result can look up the tool
const toolUseMap = new Map();

function setState(newState) {
  state = newState;
  for (const fn of stateListeners) fn(state);
}

// ---------------------------------------------------------------------------
// WebSocket connection (self-managed)
// ---------------------------------------------------------------------------

export function connectOperator() {
  if (ws && ws.readyState <= WebSocket.OPEN) return;

  loadTypeRegistry();
  loadArtifactServerUrl();

  setState('connecting');
  const url = `ws://${location.host}/ws/operator`;
  ws = new WebSocket(url);

  ws.onopen = () => {
    setState('connected');
  };

  ws.onmessage = (e) => {
    try {
      const event = JSON.parse(e.data);
      handleEvent(event);
    } catch {
      // ignore non-JSON
    }
  };

  ws.onclose = () => {
    setState('disconnected');
    setStreaming(false);
  };

  ws.onerror = () => {
    // onclose will fire after this
  };
}

export function disconnectOperator() {
  if (ws) {
    ws.close();
    ws = null;
  }
  setState('disconnected');
  setStreaming(false);
  if (messagesEl) messagesEl.innerHTML = '';
  currentAssistantGroup = null;
  // Reset session stats
  totalCost = 0;
  totalTurns = 0;
  totalDuration = 0;
  updateSessionStats();
}

// ---------------------------------------------------------------------------
// Sending
// ---------------------------------------------------------------------------

export function sendPrompt(text) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  if (!text.trim()) return;

  ws.send(JSON.stringify({ type: 'prompt', text: text.trim() }));

  // Render operator log entry
  appendUserMessage(text.trim());

  // Start a new assistant group (SYSTEM entry)
  currentAssistantGroup = document.createElement('div');
  currentAssistantGroup.className = 'op-entry assistant';
  const prefix = document.createElement('div');
  prefix.className = 'op-entry-prefix';
  prefix.textContent = 'SYSTEM';
  const body = document.createElement('div');
  body.className = 'op-entry-body';
  currentAssistantGroup.appendChild(prefix);
  currentAssistantGroup.appendChild(body);
  messagesEl.appendChild(currentAssistantGroup);

  setStreaming(true);
  if (inputEl) inputEl.value = '';
  scrollToBottom();
}

function cancelResponse() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: 'cancel' }));
  setStreaming(false);
}

// ---------------------------------------------------------------------------
// UI setup
// ---------------------------------------------------------------------------

export function initOperatorInput(containerId) {
  containerEl = document.getElementById(containerId);
  if (!containerEl) return;

  // Session info header bar
  const sessionBar = document.createElement('div');
  sessionBar.className = 'op-session-bar';

  sessionLed = document.createElement('div');
  sessionLed.className = 'op-session-led';

  const sessionLabel = document.createElement('span');
  sessionLabel.className = 'op-session-label';
  sessionLabel.textContent = 'Operator Console';

  const statsContainer = document.createElement('div');
  statsContainer.className = 'op-session-stats';

  statsCostEl = document.createElement('span');
  statsCostEl.textContent = '$0.0000';
  statsTurnsEl = document.createElement('span');
  statsTurnsEl.textContent = '0 turns';
  statsDurationEl = document.createElement('span');
  statsDurationEl.textContent = '0.0s';

  statsContainer.appendChild(statsCostEl);
  statsContainer.appendChild(statsTurnsEl);
  statsContainer.appendChild(statsDurationEl);

  sessionBar.appendChild(sessionLed);
  sessionBar.appendChild(sessionLabel);
  sessionBar.appendChild(statsContainer);
  containerEl.appendChild(sessionBar);

  // Messages area (log console)
  messagesEl = document.createElement('div');
  messagesEl.className = 'op-messages';
  containerEl.appendChild(messagesEl);

  // Processing indicator (LED + text)
  processingEl = document.createElement('div');
  processingEl.className = 'op-processing';
  const procLed = document.createElement('div');
  procLed.className = 'op-processing-led';
  const procText = document.createElement('span');
  procText.textContent = 'Processing';
  processingEl.appendChild(procLed);
  processingEl.appendChild(procText);
  containerEl.appendChild(processingEl);

  // Command-line input area
  const inputArea = document.createElement('div');
  inputArea.className = 'op-input-area';

  const promptChar = document.createElement('span');
  promptChar.className = 'op-prompt-char';
  promptChar.textContent = '>';

  inputEl = document.createElement('textarea');
  inputEl.placeholder = 'Ask about the accelerator...';
  inputEl.rows = 1;

  // Auto-resize textarea
  inputEl.addEventListener('input', () => {
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
  });

  // Enter to send, Shift+Enter for newline
  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendPrompt(inputEl.value);
    }
  });

  const controls = document.createElement('div');
  controls.className = 'op-input-controls';

  sendBtn = document.createElement('button');
  sendBtn.textContent = 'Send';
  sendBtn.addEventListener('click', () => sendPrompt(inputEl.value));

  stopBtn = document.createElement('button');
  stopBtn.textContent = 'Stop';
  stopBtn.className = 'op-stop-btn';
  stopBtn.style.display = 'none';
  stopBtn.addEventListener('click', () => cancelResponse());

  controls.appendChild(sendBtn);
  controls.appendChild(stopBtn);

  inputArea.appendChild(promptChar);
  inputArea.appendChild(inputEl);
  inputArea.appendChild(controls);
  containerEl.appendChild(inputArea);
}

export function focusOperatorInput() {
  if (inputEl) inputEl.focus();
}

export function onOperatorStateChange(fn) {
  stateListeners.push(fn);
}

// ---------------------------------------------------------------------------
// Event rendering
// ---------------------------------------------------------------------------

function handleEvent(event) {
  switch (event.type) {
    case 'text':
      appendToAssistant(() => {
        const p = document.createElement('p');
        p.style.whiteSpace = 'pre-wrap';
        p.textContent = event.content;
        return p;
      });
      break;

    case 'thinking':
      appendToAssistant(() => {
        const details = document.createElement('details');
        details.className = 'op-thinking';
        const summary = document.createElement('summary');
        summary.textContent = 'Thinking\u2026';
        const pre = document.createElement('pre');
        pre.textContent = event.content;
        details.appendChild(summary);
        details.appendChild(pre);
        return details;
      });
      break;

    case 'tool_use': {
      const toolKey = stripMcpPrefix(event.tool_name_raw || '');
      toolUseMap.set(event.tool_use_id, toolKey);
      const color = toolColor(toolKey);
      const label = toolLabel(toolKey) || event.tool_name;
      const catKey = toolToCategory(toolKey);
      const catColor = catKey ? categoryColor(catKey) : null;
      const catLabel = catKey ? categoryLabel(catKey) : null;

      appendToAssistant(() => {
        const details = document.createElement('details');
        details.className = 'op-tool';
        details.setAttribute('data-tool-use-id', event.tool_use_id);
        if (color) details.style.setProperty('--tool-color', color);

        const summary = document.createElement('summary');
        const led = document.createElement('span');
        led.className = 'op-tool-led';
        if (color) led.style.background = color;

        const nameSpan = document.createElement('span');
        nameSpan.className = 'op-tool-name';
        nameSpan.textContent = label;
        if (color) nameSpan.style.color = color;

        summary.appendChild(led);
        summary.appendChild(nameSpan);

        if (catLabel) {
          const badge = document.createElement('span');
          badge.className = 'op-category-badge';
          badge.textContent = catLabel;
          if (catColor) {
            badge.style.color = catColor;
            badge.style.borderColor = catColor;
          }
          summary.appendChild(badge);
        }

        const pre = document.createElement('pre');
        pre.textContent = JSON.stringify(event.input, null, 2);
        details.appendChild(summary);
        details.appendChild(pre);
        return details;
      });
      break;
    }

    case 'tool_result': {
      const toolEl = messagesEl.querySelector(
        `[data-tool-use-id="${event.tool_use_id}"]`
      );
      if (toolEl) {
        toolEl.classList.add('completed');
        if (event.is_error) toolEl.classList.add('errored');

        const rawToolKey = toolUseMap.get(event.tool_use_id) || '';
        renderToolResult(toolEl, event, rawToolKey);
      }
      break;
    }

    case 'result':
      setStreaming(false);
      // Update running session totals
      if (event.total_cost_usd != null) totalCost += event.total_cost_usd;
      if (event.num_turns != null) totalTurns += event.num_turns;
      if (event.duration_ms != null) totalDuration += event.duration_ms;
      updateSessionStats();

      appendToAssistant(() => {
        const bar = document.createElement('div');
        bar.className = 'op-result-bar';
        const parts = [];
        if (event.total_cost_usd != null) parts.push(`$${event.total_cost_usd.toFixed(4)}`);
        if (event.num_turns != null) parts.push(`${event.num_turns} turn${event.num_turns !== 1 ? 's' : ''}`);
        if (event.duration_ms != null) parts.push(`${(event.duration_ms / 1000).toFixed(1)}s`);
        if (event.is_error) parts.push('(error)');
        bar.textContent = parts.join(' \u00b7 ');
        return bar;
      });
      currentAssistantGroup = null;
      break;

    case 'error':
      appendToAssistant(() => {
        const div = document.createElement('div');
        div.className = 'op-error-block';
        div.textContent = event.message || 'Unknown error';
        return div;
      });
      setStreaming(false);
      break;

    case 'system':
      appendSystemMessage(event.subtype || 'system');
      break;

    case 'keepalive':
      // ignored
      break;

    default:
      break;
  }

  scrollToBottom();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function appendUserMessage(text) {
  const entry = document.createElement('div');
  entry.className = 'op-entry operator';
  const prefix = document.createElement('div');
  prefix.className = 'op-entry-prefix';
  prefix.textContent = 'OPERATOR';
  const body = document.createElement('div');
  body.className = 'op-entry-body';
  body.textContent = text;
  entry.appendChild(prefix);
  entry.appendChild(body);
  messagesEl.appendChild(entry);
}

function appendToAssistant(createEl) {
  if (!currentAssistantGroup) {
    currentAssistantGroup = document.createElement('div');
    currentAssistantGroup.className = 'op-entry assistant';
    const prefix = document.createElement('div');
    prefix.className = 'op-entry-prefix';
    prefix.textContent = 'SYSTEM';
    const body = document.createElement('div');
    body.className = 'op-entry-body';
    currentAssistantGroup.appendChild(prefix);
    currentAssistantGroup.appendChild(body);
    messagesEl.appendChild(currentAssistantGroup);
  }
  // Append to the body element within the entry
  const body = currentAssistantGroup.querySelector('.op-entry-body');
  if (body) {
    body.appendChild(createEl());
  } else {
    currentAssistantGroup.appendChild(createEl());
  }
}

function appendSystemMessage(text) {
  const div = document.createElement('div');
  div.className = 'op-system';
  div.textContent = text;
  messagesEl.appendChild(div);
}

function setStreaming(value) {
  streaming = value;
  if (processingEl) {
    processingEl.classList.toggle('active', streaming);
  }
  if (sendBtn) sendBtn.disabled = streaming;
  if (stopBtn) stopBtn.style.display = streaming ? '' : 'none';
  if (inputEl) inputEl.disabled = streaming;
  // Toggle CRT inner glow and session LED
  if (containerEl) containerEl.classList.toggle('streaming', streaming);
  if (sessionLed) sessionLed.classList.toggle('active', streaming);
}

function updateSessionStats() {
  if (statsCostEl) statsCostEl.textContent = `$${totalCost.toFixed(4)}`;
  if (statsTurnsEl) statsTurnsEl.textContent = `${totalTurns} turn${totalTurns !== 1 ? 's' : ''}`;
  if (statsDurationEl) statsDurationEl.textContent = `${(totalDuration / 1000).toFixed(1)}s`;
}

function scrollToBottom() {
  if (messagesEl) {
    requestAnimationFrame(() => {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    });
  }
}

// ---------------------------------------------------------------------------
// Type registry helpers
// ---------------------------------------------------------------------------

async function loadTypeRegistry() {
  if (typeRegistry.tool_types) return;
  try {
    const resp = await fetch('/api/type-registry');
    typeRegistry = await resp.json();
  } catch {
    // Fallback — registry stays empty, colors default to accent
  }
}

async function loadArtifactServerUrl() {
  if (artifactServerUrl) return;
  try {
    const resp = await fetch('/api/artifact-server');
    const data = await resp.json();
    artifactServerUrl = data.url;
  } catch {
    artifactServerUrl = 'http://127.0.0.1:8086';
  }
}

function stripMcpPrefix(raw) {
  return raw.replace(/^mcp__[^_]+__/, '');
}

function toolColor(key) {
  const info = typeRegistry.tool_types && typeRegistry.tool_types[key];
  return info ? info.color : null;
}

function toolLabel(key) {
  const info = typeRegistry.tool_types && typeRegistry.tool_types[key];
  return info ? info.label : null;
}

function categoryColor(key) {
  const info = typeRegistry.categories && typeRegistry.categories[key];
  return info ? info.color : null;
}

function categoryLabel(key) {
  const info = typeRegistry.categories && typeRegistry.categories[key];
  return info ? info.label : null;
}

const TOOL_TO_CATEGORY = {
  archiver_read: 'archiver_data',
  channel_read: 'channel_values',
  channel_write: 'write_results',
  execute: 'code_output',
  channel_find: 'channel_finder',
  create_static_plot: 'visualization',
  create_interactive_plot: 'visualization',
  create_dashboard: 'dashboard',
  create_document: 'document',
  screen_capture: 'screenshot',
  screenshot_capture: 'screenshot',
  graph_extract: 'graph_extraction',
  graph_compare: 'graph_comparison',
  graph_save_reference: 'graph_reference',
  ariel_search: 'search_results',
  artifact_save: 'user_artifact',
};

function toolToCategory(toolKey) {
  return TOOL_TO_CATEGORY[toolKey] || null;
}

// ---------------------------------------------------------------------------
// Smart tool result rendering
// ---------------------------------------------------------------------------

function renderToolResult(toolEl, event, toolKey) {
  const raw = typeof event.content === 'string' ? event.content : JSON.stringify(event.content, null, 2);

  let parsed = null;
  try {
    parsed = JSON.parse(raw);
  } catch {
    // Not JSON — render as plain text
  }

  if (parsed && parsed.category === 'archiver_data' && parsed.artifact_id) {
    renderArchiverResult(toolEl, parsed, event.is_error);
    return;
  }

  if (parsed && parsed.status === 'success' && parsed.artifact_id) {
    renderArtifactResult(toolEl, parsed, event.is_error);
    return;
  }

  // Default: plain text
  const resultDiv = document.createElement('div');
  resultDiv.className = 'op-tool-result' + (event.is_error ? ' op-error' : '');
  resultDiv.textContent = raw;
  toolEl.appendChild(resultDiv);
}

function renderArtifactResult(toolEl, data, isError) {
  const resultDiv = document.createElement('div');
  resultDiv.className = 'op-tool-result op-artifact-result' + (isError ? ' op-error' : '');

  const catKey = data.category;
  const catInfo = catKey && typeRegistry.categories && typeRegistry.categories[catKey];
  const color = catInfo ? catInfo.color : '#64748b';
  const label = catInfo ? catInfo.label : (catKey || data.artifact_type || 'Artifact');

  let html = '<div class="op-artifact-header">';
  html += `<span class="op-artifact-badge" style="color:${color};border-color:${color}">${esc(label)}</span>`;
  html += `<span class="op-artifact-title">${esc(data.title || '')}</span>`;
  html += '</div>';

  if (data.summary) {
    html += `<div class="op-artifact-summary">${esc(data.summary)}</div>`;
  }

  resultDiv.innerHTML = html;
  toolEl.appendChild(resultDiv);
}

async function renderArchiverResult(toolEl, data, isError) {
  const resultDiv = document.createElement('div');
  resultDiv.className = 'op-tool-result op-archiver-result' + (isError ? ' op-error' : '');

  const color = categoryColor('archiver_data') || '#4fd1c5';

  // Header with category badge + title
  let headerHtml = '<div class="op-artifact-header">';
  headerHtml += `<span class="op-artifact-badge" style="color:${color};border-color:${color}">Archiver Data</span>`;
  headerHtml += `<span class="op-artifact-title">${esc(data.title || '')}</span>`;
  headerHtml += '</div>';

  // Summary info
  if (data.summary) {
    headerHtml += `<div class="op-artifact-summary">${esc(data.summary)}</div>`;
  }

  // Chart container (will be populated async)
  headerHtml += '<div class="op-ts-chart-area" data-op-ts-chart></div>';

  resultDiv.innerHTML = headerHtml;
  toolEl.appendChild(resultDiv);
  scrollToBottom();

  // Fetch chart data from artifact server
  const serverUrl = artifactServerUrl || 'http://127.0.0.1:8086';
  const chartEl = resultDiv.querySelector('[data-op-ts-chart]');

  try {
    const resp = await fetch(
      `${serverUrl}/api/artifacts/${data.artifact_id}/data?format=chart&max_points=2000`
    );
    if (!resp.ok) throw new Error(`Chart fetch: ${resp.status}`);
    const chartData = await resp.json();
    await renderOperatorChart(chartEl, chartData);
  } catch (err) {
    console.warn('Archiver chart render failed:', err);
    chartEl.innerHTML = '<span class="op-ts-fallback">Interactive chart unavailable — view in Workspace panel</span>';
  }
}

// ---------------------------------------------------------------------------
// Plotly chart rendering (inline in operator console)
// ---------------------------------------------------------------------------

function ensurePlotlyLoaded() {
  if (_plotlyLoaded) return Promise.resolve();
  if (_plotlyLoading) return _plotlyLoading;
  _plotlyLoading = new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = 'https://cdn.plot.ly/plotly-2.35.2.min.js';
    script.onload = () => { _plotlyLoaded = true; resolve(); };
    script.onerror = () => reject(new Error('Failed to load Plotly'));
    document.head.appendChild(script);
  });
  return _plotlyLoading;
}

async function renderOperatorChart(el, chartData) {
  if (!el || !chartData || !chartData.columns) return;
  await ensurePlotlyLoaded();

  const columns = chartData.columns || [];

  // Info badges
  let infoHtml = '<div class="op-ts-info">';
  columns.forEach(c => {
    const short = c.split(':').pop() || c;
    infoHtml += `<span class="op-ts-badge op-ts-badge-ch">${esc(short)}</span>`;
  });
  infoHtml += `<span class="op-ts-badge op-ts-badge-rows">${chartData.total_rows.toLocaleString()} rows</span>`;
  if (chartData.downsampled) {
    infoHtml += `<span class="op-ts-badge op-ts-badge-ds">${chartData.returned_points.toLocaleString()} pts (downsampled)</span>`;
  }
  infoHtml += '</div>';

  // Chart div
  infoHtml += '<div class="op-ts-plotly" data-op-plotly></div>';
  el.innerHTML = infoHtml;

  const plotEl = el.querySelector('[data-op-plotly]');

  const traces = columns.map((col, ci) => ({
    x: chartData.index,
    y: chartData.data.map(row => row[ci]),
    name: col.split(':').pop() || col,
    type: 'scattergl',
    mode: 'lines',
    hovertemplate: '%{y:.4g}<extra>%{fullData.name}</extra>',
  }));

  const layout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { family: 'JetBrains Mono, Fira Code, monospace', size: 10, color: '#94a3b8' },
    margin: { l: 50, r: 20, t: 10, b: 40 },
    xaxis: {
      gridcolor: 'rgba(148,163,184,0.08)',
      linecolor: 'rgba(148,163,184,0.15)',
    },
    yaxis: {
      gridcolor: 'rgba(148,163,184,0.08)',
      linecolor: 'rgba(148,163,184,0.15)',
    },
    colorway: _tsColorway,
    legend: { orientation: 'h', y: -0.2, font: { size: 9 } },
    showlegend: columns.length > 1,
  };

  Plotly.newPlot(plotEl, traces, layout, {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
  });

  scrollToBottom();
}

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}
