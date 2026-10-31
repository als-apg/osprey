/**
 * What a `system` frame does to the operator log. Most system frames draw
 * nothing. The ones that do change the loader's activity line, close the text
 * being streamed, settle a background agent's chip, or add a notice.
 *
 * Pure: the renderer applies the effect, so the frame vocabulary lives here
 * and the renderer keeps only the drawing.
 *
 * @module chat-system
 */

/** The loader's label when the agent is between steps. */
export const WORKING_LABEL = 'Working…';

/** The loader's label while the turn waits on a background agent. */
export const BACKGROUND_WAIT_LABEL = 'Waiting for the background agent…';

/**
 * @typedef {object} SystemEffect
 * @property {boolean} [stopStreaming] - the reply text being streamed ends here
 * @property {string} [activity] - the loader's new activity line
 * @property {boolean} [closeText] - text after this frame starts a new paragraph
 * @property {string} [notice] - a notice for the turn, or for the log
 * @property {{ id: string, outcome: { status?: string, summary?: string, is_error: boolean } }} [agentDone]
 *   - a background agent's report, which settles its chip
 */

/**
 * @param {Record<string, any>} event - a `system` frame
 * @returns {SystemEffect | null} null when the frame draws nothing
 */
export function systemFrame(event) {
  switch (event.subtype) {
    case 'background_wait':
      // The agent handed work to a background agent: the turn stays open
      // until it reports, and the loader says what it is waiting on.
      return {
        stopStreaming: true,
        activity: BACKGROUND_WAIT_LABEL,
        closeText: true,
      };
    case 'agent_done':
      return {
        closeText: true,
        agentDone: {
          id: String(event.tool_use_id ?? ''),
          outcome: {
            status: event.status,
            summary: event.summary,
            is_error: event.status === 'failed',
          },
        },
      };
    case 'compact_boundary': {
      const tokens =
        typeof event.pre_tokens === 'number'
          ? ` · ${Math.round(event.pre_tokens / 1000)}k tokens summarized`
          : '';
      const how = event.trigger === 'auto' ? ' automatically' : '';
      return { closeText: true, notice: `Conversation compacted${how}${tokens}` };
    }
    case 'status':
      return event.status === 'compacting'
        ? { stopStreaming: true, activity: 'Compacting the conversation…' }
        : null;
    case 'api_retry': {
      const code = event.error_status ? ` (${event.error_status})` : '';
      const of = event.max_retries ? `/${event.max_retries}` : '';
      return {
        stopStreaming: true,
        activity: `Request failed${code} · retrying, attempt ${event.attempt ?? '?'}${of}…`,
      };
    }
    default:
      return null;
  }
}
