// @ts-check
/**
 * ARIEL Entries Helpers
 *
 * Pure, DOM-free helpers shared between entries.js and components.js.
 */

/**
 * Check if an attachment is an image, inferring from filename when type is missing.
 * @param {{type?: string, filename?: string}} att - Attachment object with optional type and filename
 * @returns {boolean} True if the attachment is an image
 */
export function isImageAttachment(att) {
  if (att.type && att.type.startsWith('image/')) return true;
  if (att.filename) {
    const ext = (att.filename.split('.').pop() || '').toLowerCase();
    return ['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg', 'bmp'].includes(ext);
  }
  return false;
}

/**
 * Format file size in human-readable format.
 * @param {number} bytes - Size in bytes
 * @returns {string} Formatted size
 */
export function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

/**
 * Split an entry's raw_text into a subject (first line) and details (the
 * remainder, falling back to the full text when there is no remainder).
 * @param {string} [rawText] - Raw entry text
 * @returns {{subject: string, details: string}} Parsed subject/details
 */
export function parseEntryText(rawText) {
  const text = rawText || '';
  const lines = text.split('\n');
  const subject = lines[0] || 'Untitled';
  const details = lines.slice(1).join('\n').trim() || text;
  return { subject, details };
}
