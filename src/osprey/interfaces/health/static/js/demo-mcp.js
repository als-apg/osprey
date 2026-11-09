// @ts-check
/*
 * System Health dashboard — `?demo` MCP introspection payload (the GET /mcp
 * shape: two stdio servers with a few tools, one http server) and its
 * envelope summary.
 */

export const DEMO_MCP = {
  probed_at: 0,
  servers: [
    {
      name: 'controls',
      transport: 'stdio',
      command: 'python',
      args: ['-m', 'osprey.mcp_server.control_system'],
      server_name: 'controls',
      server_version: '3.4.11',
      protocol: '2025-11-25',
      instructions: 'Read and write control-system channels and query archiver history',
      init_ms: 4590,
      list_ms: 166,
      tools: [
        {
          name: 'channel_read',
          title: null,
          description:
            'Read the current value of one or more channels.\n\n- `channels`: addresses to read\n- returns value, timestamp and alarm',
          inputSchema: {
            type: 'object',
            required: ['channels'],
            properties: {
              channels: {
                type: 'array',
                items: { type: 'string' },
                description: 'Channel addresses.',
              },
              timeout: {
                anyOf: [{ type: 'number' }, { type: 'null' }],
                default: null,
                description: 'Seconds.',
              },
            },
          },
          annotations: null,
        },
        {
          name: 'channel_write',
          title: null,
          description:
            'Write a value to a channel. Gated by the write switch, limits and approval hooks.',
          inputSchema: {
            type: 'object',
            required: ['writes'],
            properties: {
              writes: {
                type: 'array',
                description: 'Channel/value pairs.',
                items: {
                  type: 'object',
                  required: ['channel', 'value'],
                  properties: {
                    channel: { type: 'string', description: 'Channel address.' },
                    value: { type: 'number', description: 'New value.' },
                  },
                },
              },
            },
          },
          annotations: null,
        },
        {
          name: 'archiver_read',
          title: null,
          description: 'Retrieve historical archived data for one or more channels.',
          inputSchema: {
            type: 'object',
            properties: {
              channels: { type: 'array', items: { type: 'string' } },
              start_time: { type: 'string' },
            },
          },
          annotations: { readOnlyHint: true },
        },
      ],
    },
    {
      name: 'workspace',
      transport: 'stdio',
      command: 'python',
      args: ['-m', 'osprey.mcp_server.workspace'],
      server_name: 'workspace',
      server_version: '3.4.11',
      protocol: '2025-11-25',
      instructions: 'Workspace files and artifacts',
      init_ms: 3120,
      list_ms: 41,
      tools: [
        {
          name: 'artifact_create',
          title: null,
          description: 'Create an artifact.',
          inputSchema: { type: 'object', properties: { title: { type: 'string' } } },
          annotations: null,
        },
        {
          name: 'artifact_list',
          title: null,
          description: 'List artifacts.',
          inputSchema: { type: 'object', properties: {} },
          annotations: null,
        },
      ],
    },
    {
      name: 'canopy',
      transport: 'http',
      url: 'http://localhost:10400/mcp',
      command: '',
      args: [],
      tools: null,
    },
  ],
};

// What the server adds from the render's hook_config.json: each tool's write
// and gated coverage.
const DEMO_WRITES = new Set(['controls/channel_write', 'workspace/artifact_create']);
const DEMO_GATED = new Set(['controls/channel_write']);
for (const s of DEMO_MCP.servers) {
  for (const t of /** @type {any[]} */ (s.tools || [])) {
    const key = s.name + '/' + t.name;
    t.hooks = { write: DEMO_WRITES.has(key), gated: DEMO_GATED.has(key) };
  }
}

/** @param {any} data */
export function mcpOf(data) {
  const stdio = data.servers.filter((/** @type {any} */ s) => s.transport !== 'http');
  const ans = stdio.filter((/** @type {any} */ s) => Array.isArray(s.tools));
  return {
    configured: data.servers.map((/** @type {any} */ s) => s.name),
    stdio: stdio.length,
    answered: ans.length,
    tools: ans.reduce((/** @type {number} */ a, /** @type {any} */ s) => a + s.tools.length, 0),
    probed_at: data.probed_at,
  };
}
