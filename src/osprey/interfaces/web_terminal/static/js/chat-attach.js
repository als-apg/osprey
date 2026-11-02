// @ts-check
/* OSPREY Web Terminal — operator chat: files attached to a message.
 *
 * The composer takes files three ways: the paperclip button, a paste, and a
 * drop onto the composer or the log. Each file is uploaded as it is picked and
 * shown as a chip until the message is sent. Sending appends a block to the
 * prompt that names each stored path, so the agent can open the file with its
 * Read tool. The operator's bubble keeps the typed text and shows the names; a
 * replayed bubble that carries the block is drawn the same way.
 *
 * A send waits while an upload is in flight, and is refused while a chip holds
 * an error. Enter and the Send button are taken in the capture phase, before the
 * controller sees them. The files of a sent message are held aside until the
 * agent starts on it; a turn that never starts gives them back to the tray with
 * the text, so nothing the operator attached is lost to a refused send.
 *
 * @module chat-attach
 */

import { facilityWallClock } from '/design-system/js/facility-time.js';
import { uuid4 } from '/design-system/js/uuid.js';

import { removeChatAttachment, uploadChatAttachment } from './chat-client.js';
import { button, elem } from './chat-dom.js';
import { onChat } from './chat-events.js';
import { chatIcon } from './chat-icons.js';

/** The most files one message may carry. The server refuses an eleventh as well. */
export const MAX_FILES_PER_MESSAGE = 10;

/** The largest file, in bytes. Checked before the upload, so a large file is not sent for nothing. */
export const MAX_FILE_BYTES = 20 * 1024 * 1024;

/** What the file picker offers. The server decides what it accepts. */
const ACCEPT = '.png,.jpg,.jpeg,.gif,.webp,.pdf,.txt,.log,.csv,.json,.yaml,.yml,.md,.py';

/** The line that opens the block a sent message carries after its typed text. */
const BLOCK_TITLE = 'Attached files (read them with the Read tool):';
const BLOCK_HEAD = `\n\n${BLOCK_TITLE}\n`;
/** One file of the block: its path, then its type and size in brackets. */
const BLOCK_LINE = /^- (.+) \(([^()]*), ([^()]*)\)$/;
/** The random prefix the server puts in front of every stored name. */
const STORED_PREFIX = /^[0-9a-f]{16}-/;

/** The sentence each refusal slug shows on its chip. */
const REFUSALS = new Map([
  ['unsupported_type', 'This file type cannot be attached. Use an image, a PDF or plain text.'],
  ['file_too_large', `This file is larger than ${MAX_FILE_BYTES / (1024 * 1024)} MB.`],
  ['chat_full', 'This chat has reached its limit for attachments.'],
  ['uploads_full', 'The server has reached its limit for attachments.'],
  ['too_many_files', `A message can carry ${MAX_FILES_PER_MESSAGE} files.`],
  ['empty_file', 'This file is empty.'],
  ['invalid_name', 'This file name cannot be used.'],
  ['invalid_chat_id', 'This chat cannot take attachments.'],
  ['store_write_failed', 'The file could not be saved.'],
]);
const UPLOAD_FAILED = 'The file could not be attached.';
const REMOVE_TO_SEND = 'Remove the file that could not be attached to send it.';
const HELD = 'Sending when the upload finishes.';
const NO_TEXT = 'Type what you want to know about the file, then send.';
const WAIT = 'Attach files once the agent has finished replying.';
const FULL = `A message can carry ${MAX_FILES_PER_MESSAGE} files.`;

const PASTED_EXTENSIONS = new Map([
  ['image/png', 'png'],
  ['image/jpeg', 'jpg'],
  ['image/gif', 'gif'],
  ['image/webp', 'webp'],
]);

const SVG_NS = 'http://www.w3.org/2000/svg';
/** A paperclip on the 24-unit grid the other composer icons use. */
const PAPERCLIP =
  'M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48';

/**
 * A file the operator attached, as the block names it.
 * @typedef {{ path: string, type: string, size: number }} AttachedFile
 */

/**
 * One file in the tray.
 * @typedef {object} Chip
 * @property {number} id
 * @property {string} key - the chat the file is stored under
 * @property {string} batch - the message the file belongs to
 * @property {string} name - the name shown; the stored name once the upload lands
 * @property {number} size - bytes
 * @property {'uploading' | 'ready' | 'error'} state
 * @property {string} path - the stored path, once uploaded
 * @property {string} type - the MIME type the server found
 * @property {string} error - the sentence a refusal shows, when the state is error
 * @property {AbortController} [controller] - aborts the upload in flight
 */

/**
 * @typedef {object} ChipNode
 * @property {HTMLElement} root
 * @property {HTMLElement} name
 * @property {HTMLElement} meta
 * @property {HTMLButtonElement} dismiss
 */

/**
 * The block a sent message carries after its typed text: one line per stored
 * file, the path first, so the agent can open it.
 * @param {AttachedFile[]} files
 * @returns {string}
 */
export function attachmentBlock(files) {
  const lines = files.map((file) => `- ${file.path} (${file.type}, ${formatSize(file.size)})`);
  return `${BLOCK_HEAD}${lines.join('\n')}`;
}

/**
 * Split a message into the text the operator typed and the names of the files
 * its block names, or `null` when the message carries no block.
 * @param {string} text
 * @returns {{ typed: string, names: string[] } | null}
 */
export function splitAttachmentBlock(text) {
  const at = text.lastIndexOf(BLOCK_HEAD);
  if (at === -1) return null;
  const names = [];
  for (const line of text.slice(at + BLOCK_HEAD.length).split('\n')) {
    const match = BLOCK_LINE.exec(line);
    if (!match) return null;
    names.push(storedName(match[1]).replace(STORED_PREFIX, ''));
  }
  return { typed: text.slice(0, at), names };
}

/**
 * A byte count as the chat states it: bytes, kilobytes, or megabytes to one decimal.
 * @param {number} bytes
 * @returns {string}
 */
export function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * The name a pasted image is filed under: `pasted-image-<time>.<ext>`, with the
 * time on the facility clock to the second, so the clipboard's own name never
 * shows.
 * @param {Date} when
 * @param {string} [mime]
 * @returns {string}
 */
export function pastedImageName(when, mime = 'image/png') {
  // `YYYY-MM-DD HH:MM:SS.mmm` in the facility zone, cut at the second and kept as digits.
  const digits = facilityWallClock(when).slice(0, 19).replace(/\D/g, '');
  const ext = PASTED_EXTENSIONS.get(mime) ?? 'png';
  return `pasted-image-${digits.slice(0, 8)}-${digits.slice(8)}.${ext}`;
}

/**
 * The sentence a failed upload shows, keyed on the refusal's slug.
 * @param {unknown} err
 * @returns {string}
 */
export function refusalText(err) {
  const slug = typeof err === 'object' && err !== null && 'slug' in err ? err.slug : undefined;
  return (typeof slug === 'string' && REFUSALS.get(slug)) || UPLOAD_FAILED;
}

/**
 * Give the composer its file handling: the paperclip, paste, drop, the tray of
 * chips, and the hold on a send that has to wait. Called once by initChat,
 * through chat-features.js.
 * @param {import('./chat-features.js').ChatFeatureContext} ctx
 */
export function initChatAttachments(ctx) {
  const { inputArea, textarea, messages } = ctx;
  const toolbar = inputArea.querySelector('.op-composer-bar');
  const found = inputArea.querySelector('.op-send-btn');
  if (!toolbar || !(found instanceof HTMLButtonElement)) return;
  const sendBtn = found;

  /** @type {Chip[]} */
  let chips = [];
  /** The files of the message on its way: back in the tray if the turn never starts. */
  let sentChips = /** @type {Chip[]} */ ([]);
  /** @type {Map<number, ChipNode>} */
  const nodes = new Map();
  let nextId = 1;
  /** The message the next file joins; null until a file starts one. */
  let batch = /** @type {string | null} */ (null);
  /** A send is waiting for its uploads to finish. */
  let held = false;
  /** Our own click is on its way to the controller, so let it through. */
  let releasing = false;
  /** A message is on its way and the agent has not started on it yet. */
  let awaiting = false;
  /** The line the composer says now; the next change clears it. */
  let notice = '';

  // ---- Controls: the paperclip, the picker and the tray ----

  const attachBtn = button('op-attach-btn');
  attachBtn.title = 'Attach files';
  attachBtn.setAttribute('aria-label', 'Attach files');
  attachBtn.append(paperclipIcon());

  const picker = document.createElement('input');
  picker.type = 'file';
  picker.multiple = true;
  picker.accept = ACCEPT;
  picker.hidden = true;
  picker.tabIndex = -1;

  const tray = elem('div', 'op-attach-tray');
  const chipList = elem('div', 'op-attach-chips');
  const statusLine = elem('p', 'op-attach-status');
  statusLine.setAttribute('role', 'status');
  tray.append(chipList, statusLine);

  textarea.before(tray);
  const cmdBtn = toolbar.querySelector('.op-cmd-btn');
  if (cmdBtn) cmdBtn.after(attachBtn);
  else toolbar.prepend(attachBtn);
  inputArea.append(picker);

  // The paperclip is unavailable exactly when the composer is: the controller
  // disables the textarea while a turn runs or a hand-off is pending.
  new MutationObserver(() => {
    if (!textarea.disabled && notice === WAIT) notice = '';
    sync();
  }).observe(textarea, { attributes: true, attributeFilter: ['disabled'] });

  /** Paint the tray, the paperclip and the status line from the current state. */
  function sync() {
    for (const [id, node] of nodes) {
      if (chips.some((chip) => chip.id === id)) continue;
      node.root.remove();
      nodes.delete(id);
    }
    for (const chip of chips) {
      let node = nodes.get(chip.id);
      if (!node) {
        node = chipNode(chip);
        nodes.set(chip.id, node);
        chipList.append(node.root);
      }
      paintChip(node, chip);
    }
    attachBtn.disabled = textarea.disabled || chips.length >= MAX_FILES_PER_MESSAGE;
    statusLine.textContent = statusCopy();
    tray.hidden = chips.length === 0 && statusLine.textContent === '';
  }

  function statusCopy() {
    if (notice) return notice;
    if (chips.some((chip) => chip.state === 'error')) return REMOVE_TO_SEND;
    if (held) return HELD;
    return '';
  }

  /** @param {Chip} chip */
  function chipNode(chip) {
    const root = elem('span', 'op-attach-chip');
    const name = elem('span', 'op-attach-chip-name');
    const meta = elem('span', 'op-attach-chip-meta');
    const dismiss = button('op-attach-chip-dismiss');
    dismiss.append(chatIcon('x'));
    dismiss.addEventListener('click', () => discard(chip));
    root.append(name, meta, dismiss);
    return { root, name, meta, dismiss };
  }

  /**
   * @param {ChipNode} node
   * @param {Chip} chip
   */
  function paintChip(node, chip) {
    node.root.dataset.state = chip.state;
    node.name.textContent = chip.name;
    node.meta.textContent =
      chip.state === 'uploading'
        ? 'Uploading…'
        : chip.state === 'error'
          ? chip.error
          : formatSize(chip.size);
    node.dismiss.setAttribute('aria-label', `Remove ${chip.name}`);
  }

  // ---- Adding files ----

  /** @param {File[]} files */
  function attachFiles(files) {
    if (files.length === 0) return;
    if (textarea.disabled) {
      notice = WAIT;
      sync();
      return;
    }
    for (const file of files) {
      if (chips.length >= MAX_FILES_PER_MESSAGE) {
        notice = FULL;
        break;
      }
      notice = '';
      if (!batch) batch = uuid4();
      /** @type {Chip} */
      const chip = {
        id: nextId++,
        key: ctx.getKey(),
        batch,
        name: file.name,
        size: file.size,
        state: 'uploading',
        path: '',
        type: '',
        error: '',
      };
      chips.push(chip);
      if (file.size === 0) refuse(chip, 'empty_file');
      else if (file.size > MAX_FILE_BYTES) refuse(chip, 'file_too_large');
      else void upload(chip, file);
    }
    sync();
  }

  /**
   * @param {Chip} chip
   * @param {string} slug
   */
  function refuse(chip, slug) {
    chip.state = 'error';
    chip.error = refusalText({ slug });
  }

  /**
   * Upload one file and settle its chip. A chip taken back while the bytes were
   * in flight is not shown again, and a file that lands after that is removed.
   * @param {Chip} chip
   * @param {File} file
   */
  async function upload(chip, file) {
    const controller = new AbortController();
    chip.controller = controller;
    try {
      const stored = await uploadChatAttachment(chip.key, file, {
        name: chip.name,
        batch: chip.batch,
        signal: controller.signal,
      });
      if (chips.includes(chip)) {
        Object.assign(chip, {
          state: 'ready',
          name: stored.name,
          path: stored.path,
          type: stored.type,
          size: stored.size,
        });
      } else {
        void removeChatAttachment(chip.key, storedName(stored.path), { batch: chip.batch }).catch(
          () => {}
        );
      }
    } catch (err) {
      if (chips.includes(chip) && !controller.signal.aborted) {
        Object.assign(chip, { state: 'error', error: refusalText(err) });
      }
    }
    chip.controller = undefined;
    settle();
  }

  attachBtn.addEventListener('click', () => picker.click());
  picker.addEventListener('change', () => {
    const files = Array.from(picker.files ?? []);
    picker.value = '';
    attachFiles(files);
  });

  // Typing answers the line that asked for words.
  textarea.addEventListener('input', () => {
    if (notice !== NO_TEXT) return;
    notice = '';
    sync();
  });

  // A pasted image is filed under a name of its own, not the clipboard's.
  textarea.addEventListener('paste', (e) => {
    const data = e.clipboardData;
    const files = Array.from(data?.files ?? []);
    if (files.length === 0 || data?.getData('text/plain')) return;
    e.preventDefault();
    const when = new Date();
    attachFiles(
      files.map((file) =>
        file.type.startsWith('image/')
          ? new File([file], pastedImageName(when, file.type), { type: file.type })
          : file
      )
    );
  });

  /** @param {DragEvent} e */
  const carriesFiles = (e) => Array.from(e.dataTransfer?.types ?? []).includes('Files');
  let dragDepth = 0;
  for (const zone of [inputArea, messages]) {
    zone.addEventListener('dragenter', (e) => {
      if (!carriesFiles(e)) return;
      e.preventDefault();
      dragDepth += 1;
      inputArea.classList.add('op-attach-over');
    });
    zone.addEventListener('dragover', (e) => {
      if (!carriesFiles(e)) return;
      e.preventDefault();
      if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
    });
    zone.addEventListener('dragleave', (e) => {
      if (!carriesFiles(e)) return;
      dragDepth = Math.max(0, dragDepth - 1);
      if (dragDepth === 0) inputArea.classList.remove('op-attach-over');
    });
    zone.addEventListener('drop', (e) => {
      if (!carriesFiles(e)) return;
      e.preventDefault();
      dragDepth = 0;
      inputArea.classList.remove('op-attach-over');
      attachFiles(Array.from(e.dataTransfer?.files ?? []));
    });
  }

  // ---- Taking files back ----

  /**
   * The operator took a chip back. A file still uploading is abandoned; a stored
   * one is removed from the server. A held send goes if that was the last upload.
   * @param {Chip} chip
   */
  function discard(chip) {
    if (!chips.includes(chip)) return;
    chips = chips.filter((other) => other !== chip);
    forget(chip);
    notice = '';
    settle();
    textarea.focus();
  }

  /**
   * Drop a chip's server side: abandon an upload in flight, or remove a stored
   * file. A file already sent is not removed; the transcript names it now.
   * @param {Chip} chip
   * @param {boolean} [keepalive] - let the removal outlive the page
   */
  function forget(chip, keepalive = false) {
    if (chip.state === 'uploading') chip.controller?.abort();
    else if (chip.state === 'ready') {
      void removeChatAttachment(chip.key, storedName(chip.path), {
        batch: chip.batch,
        keepalive,
      }).catch(() => {});
    }
  }

  // A tab closed or navigated away with files still in its tray takes them
  // back, so they do not sit in the store counting against the chat.
  window.addEventListener('pagehide', () => {
    for (const chip of chips) forget(chip, true);
  });

  /** The chat moved to another conversation: nothing in the tray belongs to what follows. */
  function clearTray() {
    for (const chip of chips) forget(chip);
    chips = [];
    sentChips = [];
    held = false;
    awaiting = false;
    batch = null;
    notice = '';
    sync();
  }

  // ---- Sending ----

  /**
   * Whether the composer keeps a send from going out now, and says why. Called
   * for Enter and for the Send button, before the controller sees either.
   * @returns {boolean}
   */
  function holdSend() {
    if (chips.length === 0 || slashOpen()) return false;
    if (chips.some((chip) => chip.state === 'error')) {
      notice = '';
      sync();
      return true;
    }
    if (!textarea.value.trim()) {
      notice = NO_TEXT;
      sync();
      return true;
    }
    if (chips.some((chip) => chip.state === 'uploading')) {
      held = true;
      notice = '';
      sync();
      return true;
    }
    return false;
  }

  /** An open slash-command list owns Enter, so a send does not take it. */
  function slashOpen() {
    const popup = inputArea.querySelector('.op-slash-popup');
    return popup instanceof HTMLElement && !popup.hidden;
  }

  /** A held send goes once nothing is uploading, unless a file failed or the text is gone. */
  function settle() {
    if (held && !chips.some((chip) => chip.state === 'uploading')) {
      held = false;
      // A file that failed keeps the send where it is: its line says what to do.
      if (!chips.some((chip) => chip.state === 'error')) {
        if (textarea.value.trim()) release();
        else notice = NO_TEXT;
      }
    }
    sync();
  }

  /** Send the held message through the controller's own Send button. */
  function release() {
    releasing = true;
    try {
      sendBtn.click();
    } finally {
      releasing = false;
    }
  }

  inputArea.addEventListener(
    'keydown',
    (e) => {
      if (releasing || e.target !== textarea || e.key !== 'Enter') return;
      if (e.shiftKey || e.isComposing || !holdSend()) return;
      e.preventDefault();
      e.stopPropagation();
    },
    true
  );
  inputArea.addEventListener(
    'click',
    (e) => {
      if (releasing || !sendBtn.contains(/** @type {Node} */ (e.target))) return;
      if (!holdSend()) return;
      e.preventDefault();
      e.stopPropagation();
    },
    true
  );

  // The prompt the controller sends carries the block. The bubble it has just
  // drawn gets the names, so the log and the prompt the agent read agree.
  ctx.onBeforeSend((prompt) => {
    const ready = chips.filter((chip) => chip.state === 'ready');
    if (ready.length === 0) return prompt;
    sentChips = ready;
    chips = chips.filter((chip) => !ready.includes(chip));
    lastOperatorBubble(messages)?.append(chipRow(ready.map((chip) => chip.name)));
    batch = null;
    awaiting = true;
    notice = '';
    sync();
    return prompt + attachmentBlock(ready.map(({ path, type, size }) => ({ path, type, size })));
  });

  // ---- The chat's lifecycle ----

  // The first live frame is the agent taking the message: its files now belong
  // to the transcript. A turn that ends with no frame was refused before it
  // started, so its files come back to the tray with the text.
  onChat('frame', ({ replay }) => {
    if (replay || !awaiting) return;
    awaiting = false;
    sentChips = [];
  });
  onChat('turn-end', () => {
    if (!awaiting) return;
    awaiting = false;
    chips = [...sentChips, ...chips];
    sentChips = [];
    sync();
  });
  // A bubble that carries a block, replayed or mirrored from another tab, is
  // drawn with its names. A live send's bubble has no block yet; see above.
  onChat('user-message', ({ text }) => {
    const parts = splitAttachmentBlock(text);
    if (!parts) return;
    const bubble = lastOperatorBubble(messages);
    if (bubble) bubble.replaceChildren(document.createTextNode(parts.typed), chipRow(parts.names));
  });
  // The log is cleared both when the console moves to another conversation
  // (New conversation, a session switch) and when it reads the same one again
  // (a reconnect gap, the end of a mirrored turn, a re-bind after a hand-off).
  // Only a move empties the tray: files attached and not yet sent belong to the
  // conversation, and a re-read must not delete them. chat.js names the new key
  // before it clears the log, so the key read here is the one that follows.
  let trayKey = ctx.getKey();
  onChat('reset', () => {
    const key = ctx.getKey();
    if (key === trayKey) return;
    trayKey = key;
    clearTray();
  });

  // The tray starts empty, but the paperclip's state is painted from the start.
  sync();
}

/**
 * The names of a sent message's files, drawn on its bubble.
 * @param {string[]} names
 * @returns {HTMLElement}
 */
function chipRow(names) {
  const row = elem('div', 'op-attach-row');
  for (const name of names) row.append(elem('span', 'op-attach-file', name));
  return row;
}

/**
 * The operator's most recent bubble, which a send or a replay has just drawn.
 * @param {HTMLElement} messages
 * @returns {HTMLElement | null}
 */
function lastOperatorBubble(messages) {
  const bubbles = messages.querySelectorAll('.op-entry.operator > .op-bubble');
  const last = bubbles[bubbles.length - 1];
  return last instanceof HTMLElement ? last : null;
}

/**
 * The file's last path segment: the name it is stored under.
 * @param {string} path
 * @returns {string}
 */
function storedName(path) {
  return path.slice(path.lastIndexOf('/') + 1);
}

/** The paperclip, on the grid the other composer icons use. @returns {SVGSVGElement} */
function paperclipIcon() {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('class', 'op-icon');
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', '2');
  svg.setAttribute('stroke-linecap', 'round');
  svg.setAttribute('stroke-linejoin', 'round');
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('focusable', 'false');
  const path = document.createElementNS(SVG_NS, 'path');
  path.setAttribute('d', PAPERCLIP);
  svg.append(path);
  return svg;
}
