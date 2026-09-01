The terminal no longer shows a red `Auto-update failed` line under the prompt.
The agent's Claude Code CLI is pinned by the deployment, so its background
auto-updater is now switched off: it could only ever fail inside the container,
and succeeding would have quietly replaced the pinned version. Upgrading the
CLI stays an image rebuild against a new pin.
