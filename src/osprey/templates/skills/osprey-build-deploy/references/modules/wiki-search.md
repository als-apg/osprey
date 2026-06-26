# Module: wiki_search

Facility wiki search — typically Confluence, sometimes MediaWiki, occasionally a custom in-house wiki. Provides Claude with read access to facility documentation pages so it can ground answers in operator-authored content (procedures, schematics, meeting notes). The search MCP server runs as a stdio subprocess inside the web terminal (or, for some setups, as a containerized HTTP service) and reaches the wiki's REST API directly using a token stored in `.env`.

**Enabled when**: `modules.wiki_search.enabled: true`

## Configuration

```yaml
modules:
  wiki_search:
    enabled: true
    type: "confluence"                       # confluence | mediawiki | custom
    base_url: "https://wiki.example.org"     # no trailing slash
    api_path: "/rest/api/"                   # path appended to base_url for API calls
    auth_method: "bearer"                    # bearer | basic | api_key
    token_env_var: "WIKI_ACCESS_TOKEN"       # name of env var in .env holding the credential
    spaces:                                  # restrict search scope; omit or empty list = search everything
      - "OPERATIONS"
      - "PHYSICS"
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `type` | enum | yes | `confluence`, `mediawiki`, or `custom`. Drives which client the MCP server uses internally. |
| `base_url` | URL | yes | Wiki root URL, no trailing slash. |
| `api_path` | string | yes | API path relative to `base_url`. Defaults differ per wiki type — see "Wiki types" below. |
| `auth_method` | enum | yes | `bearer` (Personal Access Token), `basic` (user:password), or `api_key` (custom header). |
| `token_env_var` | string | yes | Name of the `.env` var holding the credential. Value never stored in `facility-config.yml`. |
| `spaces` | list of strings | no | If set, search is restricted to these space keys. Empty list or omitted = global search. |

## What scaffolding adds when this module is enabled

- compose: nothing if the search server runs as a stdio subprocess inside the web terminal (default). If the facility opts to run it as an HTTP service, a `${config.facility.prefix}-mcp-wiki-search` block is added on `${config.ports.wiki_search}` (allocate this port if needed).
- .gitlab-ci.yml: adds a `build-wiki-search` job only if running as an HTTP container; otherwise nothing.
- scripts/deploy.sh: nothing.
- scripts/verify.sh: a check that calls the wiki API's "current user" endpoint with the configured auth method and expects a 200 response. Advisory only.
- .env.template: adds an entry for `${config.modules.wiki_search.token_env_var}` with a placeholder value and a comment pointing at the relevant section of the wiki's user-settings page for token generation.

## Wiki types

### Confluence (Atlassian)

Default for most facilities running Atlassian Confluence Server / Data Center.

- `api_path`: `/rest/api/` (Confluence Server) or `/wiki/rest/api/` (Confluence Cloud — note the `/wiki` prefix).
- Auth: prefer `bearer` with a Personal Access Token over `basic` with username + password. PATs can be revoked individually and have configurable scope.
- Spaces are identified by their **space key** (typically uppercase, e.g., `OPERATIONS`), visible at the top of any page's URL or in space settings.
- Search endpoint: `${config.modules.wiki_search.base_url}${config.modules.wiki_search.api_path}content/search?cql=...` using CQL (Confluence Query Language).
- Token generation: `${config.modules.wiki_search.base_url}/plugins/personalaccesstokens/usertokens.action` (Server) or `https://id.atlassian.com/manage-profile/security/api-tokens` (Cloud).

### MediaWiki

For facilities running MediaWiki (less common in accelerator facilities, occasionally found at university-affiliated labs).

- `api_path`: `/api.php` (note the trailing `.php`, not `.api.php` and not `/api/`).
- Auth: typically `api_key` via custom header, or `basic`. MediaWiki's bearer-token support depends on installed extensions.
- Spaces are not a MediaWiki concept; the equivalent is **categories** or **namespaces**. Use the `spaces` field to list category names if your client supports category filtering, otherwise leave empty.
- Search endpoint: `${config.modules.wiki_search.base_url}${config.modules.wiki_search.api_path}?action=opensearch&search=...&format=json`.

### Custom

For facilities with an in-house wiki that doesn't fit either pattern. The pattern is:

1. Set `type: "custom"`.
2. Author your own MCP server using the `modules.custom_mcp_servers` module (see `references/modules/custom-mcp-servers.md`).
3. The custom server reads `${config.modules.wiki_search.base_url}` + `${config.modules.wiki_search.token_env_var}` from its env block (passed via compose) and implements whatever auth pattern the in-house wiki requires.
4. The `wiki_search` module here is then just a configuration carrier; the actual server lives under `custom_mcp_servers`.

## Operating the module

### Verify connectivity

```bash
# Confluence example — get the authenticated user's profile
ssh ${config.deploy.host} "curl -s -H 'Authorization: Bearer $${config.modules.wiki_search.token_env_var}' \
  ${config.modules.wiki_search.base_url}${config.modules.wiki_search.api_path}user/current | python3 -m json.tool"
```

Expected: a JSON object with the user's `username`, `displayName`, etc. A 401 means the token is wrong or expired. A 404 means `api_path` is wrong.

### Test a search

```bash
# Confluence CQL search
ssh ${config.deploy.host} "curl -s -H 'Authorization: Bearer $${config.modules.wiki_search.token_env_var}' \
  '${config.modules.wiki_search.base_url}${config.modules.wiki_search.api_path}content/search?cql=text~\"beam+orbit\"&limit=3' \
  | python3 -m json.tool"
```

If results are empty but the API responds 200, the most likely causes are: (a) the token's user lacks read permission on the searched spaces, or (b) `spaces` in `facility-config.yml` is too narrow and excludes the content.

### Rotate the token

1. Generate a new token via the wiki's user-settings page.
2. Update the value in `.env` (NOT `.env.template`, which is generated and gitignored values don't live there).
3. `${config.runtime.compose_command} restart ${config.facility.prefix}-mcp-wiki-search` (if HTTP container) — or restart any web terminal that hosts it as a subprocess.

### Adjust scope

If users complain about missing pages: edit `modules.wiki_search.spaces` in `facility-config.yml` to add the missing space key(s), then re-scaffold so the change reaches the running service. If users complain about irrelevant results: tighten the spaces list.

## Common failures

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `401 Unauthorized` | Token expired, revoked, or wrong env var name | Regenerate token; verify `${config.modules.wiki_search.token_env_var}` matches what's in `.env` |
| `404 Not Found` on every endpoint | Wrong `api_path` for the wiki type | Check the wiki type's API docs; Confluence Cloud needs `/wiki/rest/api/`, Server needs `/rest/api/` |
| `403 Forbidden` on specific pages | Token's user lacks read permission on the space | Grant the bot user space-level read access; do NOT widen the token's scope just to cover one space |
| Empty results for queries that should match | `spaces` list too restrictive, OR content not indexed by the wiki's search engine | Widen `spaces` first; if still empty, search the wiki UI directly to confirm the content is indexed |
| Intermittent timeouts | Wiki rate-limiting | Reduce concurrent queries; for Confluence, check the per-user request limit in the wiki admin panel |
| `SSLError` on HTTPS endpoints | Wiki uses an internal CA not trusted by the container | Mount the CA bundle into the container or set `REQUESTS_CA_BUNDLE` env var |

## Cross-references

- If `modules.custom_mcp_servers.enabled` is also true and `wiki_search.type == "custom"`, the actual MCP server lives there. This module is just the config carrier.
- If `modules.web_terminals.enabled` is also true (the typical case), the wiki search server runs as a stdio subprocess inside each web terminal container — no port allocation needed in `${config.ports}`.
- If `modules.integration_tests` is the always-on integration tests module, its `wiki_search` check category will report on whether the API is reachable.

## Disabling

Set `modules.wiki_search.enabled: false` (or remove the block) and re-scaffold. Then audit:

- The `verify.sh` wiki check is removed.
- The `${config.modules.wiki_search.token_env_var}` entry in `.env.template` is removed (the value in `.env` can stay; it's just unused).
- If the wiki search was an HTTP container, the compose service and CI build job are removed; pull `${config.runtime.compose_command} down` to stop the running service.
- If a profile YAML referenced wiki search as an MCP server, remove that entry too — otherwise Claude will try to connect at runtime and fail.
