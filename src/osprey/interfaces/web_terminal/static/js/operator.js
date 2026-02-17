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

function setState(newState) {
  state = newState;
  for (const fn of stateListeners) fn(state);
}

// ---------------------------------------------------------------------------
// WebSocket connection (self-managed)
// ---------------------------------------------------------------------------

export function connectOperator() {
  if (ws && ws.readyState <= WebSocket.OPEN) return;

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

    case 'tool_use':
      appendToAssistant(() => {
        const details = document.createElement('details');
        details.className = 'op-tool';
        details.setAttribute('data-tool-use-id', event.tool_use_id);
        const summary = document.createElement('summary');
        // LED indicator
        const led = document.createElement('span');
        led.className = 'op-tool-led';
        const nameSpan = document.createElement('span');
        nameSpan.textContent = event.tool_name;
        summary.appendChild(led);
        summary.appendChild(nameSpan);
        const pre = document.createElement('pre');
        pre.textContent = JSON.stringify(event.input, null, 2);
        details.appendChild(summary);
        details.appendChild(pre);
        return details;
      });
      break;

    case 'tool_result': {
      // Find matching tool_use element and append result inside it
      const toolEl = messagesEl.querySelector(
        `[data-tool-use-id="${event.tool_use_id}"]`
      );
      if (toolEl) {
        // Mark tool as completed
        toolEl.classList.add('completed');
        if (event.is_error) {
          toolEl.classList.add('errored');
        }

        const resultDiv = document.createElement('div');
        resultDiv.className = 'op-tool-result' + (event.is_error ? ' op-error' : '');
        if (typeof event.content === 'string') {
          resultDiv.textContent = event.content;
        } else {
          resultDiv.textContent = JSON.stringify(event.content, null, 2);
        }
        toolEl.appendChild(resultDiv);
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
