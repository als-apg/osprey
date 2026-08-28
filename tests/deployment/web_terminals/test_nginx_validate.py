"""Durable `nginx -t` validation of the rendered nginx fragment.

`test_render.py`'s and `test_auth_tls_seams.py`'s assertions are string matches
on the Jinja output — cheap and fast, but they can't catch an nginx syntax error
the strings happen to satisfy. This module closes that gap for real by actually
invoking `nginx -t` inside `nginx:1.27-alpine` (the base image the deployed stack
uses) against each render shape the auth/TLS seams can produce:

- the default render (`auth.method` unset -> "token", `tls.enabled` unset ->
  False): no wall and no injection, each terminal reached through its own
  `?token=` exchange — the pre-auth plain-http posture.
- the fully enabled render (`tls.enabled: true` with a real self-signed cert/key
  pair generated fresh per test run, plus `auth.method: password`): two server
  blocks, the plain port's `return 301 https://…`, `listen 443 ssl` +
  `ssl_certificate*`, the two `map`s, a per-user `auth_request` and `internal`
  verify target, the shared named 401 handler, and the public `/auth/` prefix.
- the cleartext-auth render (`auth.method: password` with
  `allow_insecure_http`, the shape a facility behind its own TLS terminator
  deploys): the same auth surface inside a *single* server block.
- the open render (`auth.method: none`, with a `login: false` entry beside a
  normal one): no sidecar surface at all, and yet the per-user `include` +
  envsubst chain still runs — the only shape where a secret snippet is read at
  start with no `auth_request` in front of it, and the only one whose two
  ungated locations are shaped differently from each other.

Every one of these tests asserts the constructs it means to validate are
actually present in the rendered fragment before handing it to nginx — a
template that stopped emitting the auth surface would otherwise still "pass"
`nginx -t` and report a vacuous green.

Skipped entirely when docker (or openssl) is unavailable, mirroring
`tests/e2e/test_dockerfile_e2e.py`'s `_docker_available()` pattern.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from osprey.deployment.web_terminals.render import (
    NGINX_TEMPLATES_OUTPUT_DIR,
    TERMINAL_SECRET_HEADER,
    render_web_terminals,
    terminal_secret_env_var,
)

_BASE_PORTS = {"web": 9091, "artifact": 9291, "ariel": 9391, "lattice": 9491}
_NGINX_IMAGE = "nginx:1.27-alpine"

#: Where the base image's entrypoint writes the envsubst'd per-user snippets, and
#: which variables it may substitute — kept byte-identical to the values the
#: compose overlay (docker-compose.web.yml.j2) sets on the nginx service, so this
#: exercises the exact include + envsubst chain a real deployment runs. The
#: output directory is pointedly NOT the entrypoint default of /etc/nginx/conf.d
#: (which the base image globs into `http{}`, where a per-user header would reach
#: every upstream); it is read only by the per-user `include` inside each
#: authenticated location.
_NGINX_ENVSUBST_OUTPUT_DIR = "/etc/nginx/osprey"
_NGINX_ENVSUBST_FILTER = "^OSPREY_TERMINAL_SECRET_"
#: The mode the compose overlay mounts that tmpfs with (its service-level
#: `tmpfs: [- /etc/nginx/osprey:mode=0700]`), replayed here as docker-run's
#: `--tmpfs <dir>:mode=` — the same option string, which is why the compose
#: entry carries it as a string rather than as a YAML-octal `mode:` key. The substituted snippets
#: hold every roster user's operator secret in plaintext at 0644, so the
#: directory around them is what keeps them off every other process in the
#: container. Root-owned 0700 is enough for both readers: the entrypoint writes
#: as root, and nginx's MASTER process (also root) is what opens the includes.
_ENVSUBST_OUTPUT_MODE = "0700"
#: The default.conf path as `nginx -T` labels it in its per-file dump — the base
#: image maps a `conf.d/*.conf` mount to this fixed name.
_DEFAULT_CONF_PATH = "/etc/nginx/conf.d/default.conf"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = [
    pytest.mark.dockerbuild,
    pytest.mark.skipif(
        not (_docker_available() and shutil.which("openssl")),
        reason="docker (or openssl, for the self-signed cert fixture) not available",
    ),
]


def _config(tls: dict | None = None, auth: dict | None = None, users: list | None = None) -> dict:
    """A rendering config; *users* takes plain names or full roster entries (a
    `login: false` one is what the open render's exempt case needs)."""
    web_terminals: dict = {
        "enabled": True,
        "nginx_port": 9080,
        "web_base_port": _BASE_PORTS["web"],
        "artifact_base_port": _BASE_PORTS["artifact"],
        "ariel_base_port": _BASE_PORTS["ariel"],
        "lattice_base_port": _BASE_PORTS["lattice"],
        "users": ["alice", "bob"] if users is None else users,
    }
    if tls is not None:
        web_terminals["tls"] = tls
    if auth is not None:
        web_terminals["auth"] = auth
    return {
        "facility": {"name": "Demo Light Source", "prefix": "dls", "timezone": "UTC"},
        "registry": {"url": "git.dls.example.org:5050/physics/production/dls-profiles"},
        "deploy": {"host": "dls-deploy", "fqdn": "dls-deploy.dls.example.org"},
        "modules": {"web_terminals": web_terminals},
    }


def _generate_self_signed_cert(certs_dir: Path) -> tuple[Path, Path]:
    """Generate a throwaway self-signed cert/key pair for the enabled TLS render."""
    cert_path = certs_dir / "dls.crt"
    key_path = certs_dir / "dls.key"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key_path),
            "-out",
            str(cert_path),
            "-days",
            "1",
            "-subj",
            "/CN=dls-deploy.dls.example.org",
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )
    return cert_path, key_path


def _directives(conf: str) -> str:
    """`conf` with its comment lines dropped.

    The template's prose names `auth_request`, `/verify` and `$request_uri` at
    length, so counting or absence assertions have to ignore it.
    """
    return "\n".join(line for line in conf.splitlines() if not line.lstrip().startswith("#"))


def _assert_auth_surface_present(nginx_conf: str, users: list[str]) -> None:
    """Every construct the enabled renders exist to hand nginx is really there.

    Without this, a template that stopped emitting the auth surface would still
    produce a fragment `nginx -t` accepts, and these tests would report a green
    that validated nothing about authentication.
    """
    directives = _directives(nginx_conf)
    assert directives.count("auth_request ") == len(users)
    for user in users:
        assert f"auth_request /_osprey_auth/{user};" in directives
        assert f"location = /_osprey_auth/{user} {{" in directives
        assert f"/verify?user={user};" in directives
    assert directives.count("internal;") == len(users)
    assert "map $uri $osprey_auth_next {" in directives
    assert "map $http_upgrade$http_accept $osprey_auth_wants_login_page {" in directives
    assert "error_page 401 = @osprey_auth_401;" in directives
    assert "location @osprey_auth_401 {" in directives
    assert "location /auth/ {" in directives
    assert "return 200;" not in nginx_conf


def _terminal_secrets(users: list[str]) -> dict[str, str]:
    """A distinct, envsubst-legal operator secret per user.

    Distinct values are what let the no-leak assertions tell one user's rendered
    header apart from another's in an ``nginx -T`` dump: a snippet that leaked a
    neighbour's variable would show that neighbour's value where it does not
    belong.
    """
    return {user: f"terminal-secret-for-{user}" for user in users}


def _write_secret_templates(artifacts: dict[str, str], dest: Path) -> None:
    """Lay a render's per-user secret templates out flat under *dest*.

    ``render_web_terminals(..., terminal_secrets=...)`` returns each snippet
    keyed ``nginx/templates/secret-<user>.conf.template``; the base image's
    entrypoint expects them directly inside the mounted ``/etc/nginx/templates``,
    so the leading directory is stripped here — exactly what the write seam does
    when it lays the templates directory out for the compose mount.
    """
    prefix = f"{NGINX_TEMPLATES_OUTPUT_DIR}/"
    for path, content in artifacts.items():
        if path.startswith(prefix):
            (dest / path[len(prefix) :]).write_text(content)


def _run_nginx_t(
    conf_dir: Path,
    certs_dir: Path | None,
    *,
    templates_dir: Path | None = None,
    secret_env: dict[str, str] | None = None,
    command: tuple[str, ...] = ("nginx", "-t"),
    entrypoint: str | None = None,
) -> subprocess.CompletedProcess:
    """Run *command* (default ``nginx -t``) in ``nginx:1.27-alpine`` against the
    rendered fragment.

    When *templates_dir* is given, the base image's entrypoint is driven exactly
    as the rendered compose overlay drives it: the per-user secret templates are
    mounted read-only at ``/etc/nginx/templates`` and envsubst'd — from the
    ``OSPREY_TERMINAL_SECRET_*`` values in *secret_env* — into
    ``/etc/nginx/osprey`` before the config is read, so the per-user ``include``
    each authenticated location carries resolves to a real file. That output
    directory is provided as a ``--tmpfs`` — the docker-run equivalent of the
    ``tmpfs: /etc/nginx/osprey`` the compose overlay now declares on the nginx
    service — because the entrypoint's own ``-w`` guard skips the whole
    substitution when the directory is not already writable, and the base image
    ships none. No writable bind mount and no host scaffold: this exercises the
    exact self-provisioning mechanism a real ``osprey up`` relies on.

    *entrypoint* overrides the image's own. Only one caller needs it: the base
    entrypoint ``exec``s whatever it is handed, so a test that has to LOOK at
    the substituted directory afterwards has to drive the substitution step
    itself rather than let nginx replace the process.
    """
    mounts = [
        "-v",
        f"{conf_dir}/default.conf:{_DEFAULT_CONF_PATH}:ro",
    ]
    if certs_dir is not None:
        mounts += ["-v", f"{certs_dir}:/etc/nginx/certs:ro"]
    extra: list[str] = []
    if templates_dir is not None:
        mounts += ["-v", f"{templates_dir}:/etc/nginx/templates:ro"]
        extra += [
            "--tmpfs",
            f"{_NGINX_ENVSUBST_OUTPUT_DIR}:mode={_ENVSUBST_OUTPUT_MODE}",
            "-e",
            f"NGINX_ENVSUBST_OUTPUT_DIR={_NGINX_ENVSUBST_OUTPUT_DIR}",
            "-e",
            f"NGINX_ENVSUBST_FILTER={_NGINX_ENVSUBST_FILTER}",
        ]
        for var, value in (secret_env or {}).items():
            extra += ["-e", f"{var}={value}"]
    if entrypoint is not None:
        extra += ["--entrypoint", entrypoint]
    return subprocess.run(
        ["docker", "run", "--rm", *mounts, *extra, _NGINX_IMAGE, *command],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _config_file_sections(dump: str) -> dict[str, str]:
    """Split an ``nginx -T`` dump into ``{path: body}`` by its file markers.

    ``nginx -T`` prints every config file it read under a
    ``# configuration file <path>:`` banner, including each ``include``d one —
    so a per-user secret snippet lands in its OWN section keyed by
    ``/etc/nginx/osprey/secret-<user>.conf``, not inline in the location that
    included it. Splitting on the banner is what lets the no-leak assertions read
    one user's snippet in isolation.
    """
    sections: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for line in dump.splitlines():
        match = re.match(r"# configuration file (.+):$", line)
        if match:
            if current is not None:
                sections[current] = "\n".join(lines)
            current = match.group(1)
            lines = []
        elif current is not None:
            lines.append(line)
    if current is not None:
        sections[current] = "\n".join(lines)
    return sections


def _location_body(conf: str, user: str) -> str:
    """The brace-matched body of *user*'s ``location /u/<user>/ { ... }`` block.

    Used to prove which secret ``include`` a given location carries: string
    counting over the whole file could not tell a directive inside alice's
    location from one inside bob's.
    """
    marker = f"location /u/{user}/ {{"
    start = conf.index(marker)
    depth = 0
    for i in range(start, len(conf)):
        if conf[i] == "{":
            depth += 1
        elif conf[i] == "}":
            depth -= 1
            if depth == 0:
                return conf[start : i + 1]
    raise AssertionError(f"unterminated location block for {user!r}")


def test_default_gated_off_render_passes_nginx_t() -> None:
    """C1: the default (auth token / tls off) render is not just inert by
    string-match — it's a config `nginx -t` actually accepts."""
    # Arrange
    artifacts = render_web_terminals(_config())

    with tempfile.TemporaryDirectory() as tmp:
        conf_dir = Path(tmp)
        (conf_dir / "default.conf").write_text(artifacts["nginx/nginx.conf"])

        # Act
        result = _run_nginx_t(conf_dir, certs_dir=None)

    # Assert
    assert result.returncode == 0, result.stderr


def test_enabled_tls_and_auth_render_passes_nginx_t() -> None:
    """C2: the fully enabled render (tls.enabled + a real self-signed cert/key,
    auth.method: password) — two server blocks, the plain port's 301, the ssl
    listener and cert pair, the two maps, a per-user `auth_request` and
    `internal` verify target, the shared named 401 handler and the public
    `/auth/` prefix — is validated by running nginx, not string-matched.

    Several of those constructs are ones a fragment can get syntactically wrong
    in ways no string match notices: a named location referenced by `error_page`
    but never defined, a `map` at the wrong context, or an `if` block nginx
    rejects where it stands.
    """
    users = ["alice", "bob"]
    secrets = _terminal_secrets(users)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        conf_dir = tmp_path / "conf"
        certs_dir = tmp_path / "certs"
        templates_dir = tmp_path / "templates"
        conf_dir.mkdir()
        certs_dir.mkdir()
        templates_dir.mkdir()

        cert_path, key_path = _generate_self_signed_cert(certs_dir)

        # Arrange — cert/key paths must match where they'll be mounted inside
        # the container (/etc/nginx/certs/...), not the host tmp path. Rendering
        # WITH terminal_secrets additionally emits the per-user snippet templates
        # the authenticated locations `include`; without them (or their envsubst
        # output) nginx dies at start on the missing include, so `nginx -t`
        # would never reach the surface this test means to validate.
        artifacts = render_web_terminals(
            _config(
                users=users,
                tls={
                    "enabled": True,
                    "cert": f"/etc/nginx/certs/{cert_path.name}",
                    "key": f"/etc/nginx/certs/{key_path.name}",
                },
                auth={"method": "password"},
            ),
            terminal_secrets=secrets,
        )
        nginx_conf = artifacts["nginx/nginx.conf"]
        # Guard against a vacuous green: nginx would happily accept a fragment
        # that had quietly stopped emitting the whole gated surface.
        _assert_auth_surface_present(nginx_conf, users)
        assert "listen 443 ssl;" in nginx_conf
        assert f"ssl_certificate /etc/nginx/certs/{cert_path.name};" in nginx_conf
        assert f"ssl_certificate_key /etc/nginx/certs/{key_path.name};" in nginx_conf
        assert "return 301 https://$host$request_uri;" in nginx_conf

        (conf_dir / "default.conf").write_text(nginx_conf)
        _write_secret_templates(artifacts, templates_dir)
        secret_env = {terminal_secret_env_var(user): value for user, value in secrets.items()}

        # Act
        result = _run_nginx_t(
            conf_dir,
            certs_dir=certs_dir,
            templates_dir=templates_dir,
            secret_env=secret_env,
        )

    # Assert
    assert result.returncode == 0, result.stderr


def test_cleartext_auth_render_passes_nginx_t() -> None:
    """C3: the auth-without-TLS render (`allow_insecure_http`, the shape a
    facility behind its own TLS terminator deploys) puts the same gated surface
    inside a SINGLE server block — a different nesting of the same directives,
    and one nothing else runs nginx against."""
    users = ["alice", "bob"]
    secrets = _terminal_secrets(users)

    # Arrange — render WITH terminal_secrets so the per-user snippet templates
    # the authenticated locations `include` are materialized and envsubst'd (see
    # the C2 test for why the include is a hard startup dependency).
    artifacts = render_web_terminals(
        _config(users=users, auth={"method": "password", "allow_insecure_http": True}),
        terminal_secrets=secrets,
    )
    nginx_conf = artifacts["nginx/nginx.conf"]
    _assert_auth_surface_present(nginx_conf, users)
    assert "absolute_redirect off;" in nginx_conf
    assert "listen 443 ssl;" not in nginx_conf

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        conf_dir = tmp_path / "conf"
        templates_dir = tmp_path / "templates"
        conf_dir.mkdir()
        templates_dir.mkdir()
        (conf_dir / "default.conf").write_text(nginx_conf)
        _write_secret_templates(artifacts, templates_dir)
        secret_env = {terminal_secret_env_var(user): value for user, value in secrets.items()}

        # Act
        result = _run_nginx_t(
            conf_dir,
            certs_dir=None,
            templates_dir=templates_dir,
            secret_env=secret_env,
        )

    # Assert
    assert result.returncode == 0, result.stderr


def test_open_render_with_an_exempt_entry_passes_nginx_t() -> None:
    """C4: the open render (`auth.method: none`) — the only posture that reads a
    per-user secret snippet at start with no `auth_request` in front of it.

    Worth running real nginx against rather than string-matching, because what
    is new here is a startup-time file dependency without the surface that used
    to imply it. The `include` is a hard startup failure when the envsubst
    output is missing, and under `none` there is no sidecar, no `/auth/` prefix
    and no named 401 handler around it to fail first — so a template that got
    the injection predicate and the artifact predicate even slightly out of
    step would produce a config that renders cleanly and refuses to start.

    A `login: false` entry sits beside the injected one deliberately: this is
    the only render where two ungated locations are shaped differently from
    each other, one carrying an include and the other the clear that replaces
    it.

    Asserted clean rather than merely accepted: a duplicate header name — which
    is exactly what a clear written beside the include would be, and the
    snippet is a separate file this test is the first to resolve — leaves nginx
    exiting 0 with `could not build optimal proxy_headers_hash` on stderr, on
    every start and every reload.
    """
    roster = [{"name": "alice", "index": 0}, {"name": "kiosk", "index": 1, "login": False}]
    users = ["alice", "kiosk"]
    secrets = _terminal_secrets(users)

    # Arrange
    artifacts = render_web_terminals(
        _config(users=roster, auth={"method": "none"}), terminal_secrets=secrets
    )
    nginx_conf = artifacts["nginx/nginx.conf"]

    # Arrange — guard against a vacuous green twice over: a render that had
    # stopped injecting would pass `nginx -t` with nothing to validate, and one
    # that had grown a login wall would not be this posture at all.
    directives = _directives(nginx_conf)
    assert "include /etc/nginx/osprey/secret-alice.conf;" in directives
    assert "secret-kiosk.conf" not in nginx_conf
    assert f'{TERMINAL_SECRET_HEADER} "";' in directives
    assert "auth_request" not in directives
    assert "location /auth/" not in directives

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        conf_dir = tmp_path / "conf"
        templates_dir = tmp_path / "templates"
        conf_dir.mkdir()
        templates_dir.mkdir()
        (conf_dir / "default.conf").write_text(nginx_conf)
        # Under `none` the snippets ARE written — the artifact predicate is the
        # injection predicate — so the same entrypoint chain the gated renders
        # drive applies here, for the one non-exempt user.
        _write_secret_templates(artifacts, templates_dir)
        assert [path.name for path in sorted(templates_dir.iterdir())] == [
            "secret-alice.conf.template"
        ]
        secret_env = {terminal_secret_env_var(user): value for user, value in secrets.items()}

        # Act — `-T` so the resolved snippet is observable, not just accepted.
        result = _run_nginx_t(
            conf_dir,
            certs_dir=None,
            templates_dir=templates_dir,
            secret_env=secret_env,
            command=("nginx", "-T"),
        )

    # Assert — accepted, and accepted without complaint.
    assert result.returncode == 0, result.stderr
    assert "[warn]" not in result.stderr, result.stderr

    # Assert — the injected user's secret is really in the resolved config, and
    # the exempt user's is nowhere in it. Two occurrences of the header name:
    # alice's snippet sets it, kiosk's location clears it.
    dump = result.stdout
    assert dump.count(TERMINAL_SECRET_HEADER) == 2
    snippet = _config_file_sections(dump)[f"{_NGINX_ENVSUBST_OUTPUT_DIR}/secret-alice.conf"]
    assert f'{TERMINAL_SECRET_HEADER} "{secrets["alice"]}";' in snippet
    assert secrets["kiosk"] not in dump

    # Assert — and the ungated entry claims the same name by clearing it, which
    # is what keeps a client from supplying one.
    kiosk_location = _location_body(_config_file_sections(dump)[_DEFAULT_CONF_PATH], "kiosk")
    assert f'{TERMINAL_SECRET_HEADER} "";' in kiosk_location


def test_secret_include_is_scoped_per_user_with_no_cross_leak() -> None:
    """IA-5: the operator secret nginx stamps on a proxied request is per-user
    and un-leakable across locations.

    Rendered through the same per-user ``include`` + envsubst chain the compose
    overlay wires, an ``nginx -T`` dump of the fully resolved config must show
    the ``X-Osprey-Terminal-Secret`` header exactly once per roster user — no
    stray copy at ``http``/``server`` scope, where the base image's default
    envsubst output directory would have put it and where it would reach every
    upstream — with each user's header carrying ONLY that user's secret, from a
    snippet ``include``d ONLY by that user's ``/u/<user>/`` location. A snippet
    that named a neighbour's variable, or a location that pulled in a
    neighbour's snippet, would hand one operator's credential to another's
    terminal — the exact isolation this carrier exists to establish.
    """
    users = ["alice", "bob"]
    secrets = _terminal_secrets(users)
    artifacts = render_web_terminals(
        _config(users=users, auth={"method": "password", "allow_insecure_http": True}),
        terminal_secrets=secrets,
    )
    nginx_conf = artifacts["nginx/nginx.conf"]
    _assert_auth_surface_present(nginx_conf, users)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        conf_dir = tmp_path / "conf"
        templates_dir = tmp_path / "templates"
        conf_dir.mkdir()
        templates_dir.mkdir()
        (conf_dir / "default.conf").write_text(nginx_conf)
        _write_secret_templates(artifacts, templates_dir)
        secret_env = {terminal_secret_env_var(user): value for user, value in secrets.items()}

        # Act — dump the fully resolved config (`-T`), including every included
        # per-user snippet after envsubst, so the header's real placement and
        # value are observable rather than string-matched off the Jinja output.
        result = _run_nginx_t(
            conf_dir,
            certs_dir=None,
            templates_dir=templates_dir,
            secret_env=secret_env,
            command=("nginx", "-T"),
        )

    # Assert
    assert result.returncode == 0, result.stderr
    dump = result.stdout

    # One header per user and no more: a copy at http/server scope (the default
    # envsubst output directory's failure mode) would push this above the roster.
    assert dump.count(TERMINAL_SECRET_HEADER) == len(users)

    sections = _config_file_sections(dump)
    default_conf = sections[_DEFAULT_CONF_PATH]

    for user in users:
        others = [other for other in users if other != user]

        # This user's snippet lives in its own included file, carrying its own
        # header and secret and nothing of any other user's.
        snippet_path = f"{_NGINX_ENVSUBST_OUTPUT_DIR}/secret-{user}.conf"
        assert snippet_path in sections, sorted(sections)
        snippet = sections[snippet_path]
        assert f'{TERMINAL_SECRET_HEADER} "{secrets[user]}";' in snippet
        for other in others:
            assert secrets[other] not in snippet
            assert terminal_secret_env_var(other) not in snippet

        # That snippet is reached ONLY from this user's own location — its
        # include appears inside `/u/<user>/` and no other user's snippet does.
        location_body = _location_body(default_conf, user)
        assert f"include {snippet_path};" in location_body
        for other in others:
            assert f"secret-{other}.conf" not in location_body

    # Each user's secret appears exactly once in the whole resolved config —
    # a value showing up twice would mean a snippet or location leaked it.
    for user in users:
        assert dump.count(secrets[user]) == 1


def _log_formats(dump: str) -> list[str]:
    """Every ``log_format`` definition in an ``nginx -T`` dump, whitespace-joined.

    A ``log_format`` may be written across several continuation lines (this
    deployment's is), so the assertions below need the whole directive as one
    string rather than the line the name happens to sit on.
    """
    formats: list[str] = []
    for raw in re.split(r"(?m)^(?=\s*log_format\s)", dump)[1:]:
        directive = raw[: raw.index(";") + 1] if ";" in raw else raw
        formats.append(" ".join(directive.split()))
    return formats


def test_access_log_never_carries_the_query_string() -> None:
    """C6: no format nginx ends up using may log a variable holding the query.

    The `?token=` URL is the ONLY way into an auth-off multi-user terminal, so a
    `combined`-format access line — which logs `$request`, query string and all
    — writes a live operator secret to container stdout, where `docker logs` and
    every log shipper can read it and nothing can revoke it.

    Read off `nginx -T` rather than off the rendered fragment, because the base
    image's own main config carries a `log_format main` and an http-level
    `access_log` of its own. The fragment's directives have to actually
    SUPPRESS that one, which is a property of the resolved config and of nothing
    the template says about itself.
    """
    users = ["alice", "bob"]
    secrets = _terminal_secrets(users)
    artifacts = render_web_terminals(
        _config(users=users, auth={"method": "password", "allow_insecure_http": True}),
        terminal_secrets=secrets,
    )
    nginx_conf = artifacts["nginx/nginx.conf"]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        conf_dir = tmp_path / "conf"
        templates_dir = tmp_path / "templates"
        conf_dir.mkdir()
        templates_dir.mkdir()
        (conf_dir / "default.conf").write_text(nginx_conf)
        _write_secret_templates(artifacts, templates_dir)
        secret_env = {terminal_secret_env_var(user): value for user, value in secrets.items()}

        result = _run_nginx_t(
            conf_dir,
            certs_dir=None,
            templates_dir=templates_dir,
            secret_env=secret_env,
            command=("nginx", "-T"),
        )

    assert result.returncode == 0, result.stderr
    dump = result.stdout

    # Every `access_log` that names a format names the sanitized one. The base
    # image's http-level default is still IN the dump (it is a real directive in
    # a config file nginx read); what matters is that this deployment's server
    # blocks define their own, which is what stops the http-level one being
    # inherited.
    selected = re.findall(r"(?m)^\s*access_log\s+(\S+)\s+(\S+?);", dump)
    server_level = [fmt for path, fmt in selected if path == "/dev/stdout"]
    assert server_level, f"no sanitized access_log in the dump: {selected}"
    assert set(server_level) == {"osprey_sanitized"}
    # One per server block: the content server, and the redirect-only one is not
    # rendered in this (no-TLS) shape.
    assert len(server_level) == 1

    # SERVER level, not http. nginx inherits access_log from the level above
    # only when the current level defines none, and every access_log on ONE
    # level is written — so an http-level directive here would ADD the sanitized
    # line beside the base image's `combined`-format one rather than replace it.
    fragment = _directives(nginx_conf)
    server_start = fragment.index("server {")
    assert "access_log" not in fragment[:server_start], fragment[:server_start]
    assert "access_log /dev/stdout osprey_sanitized;" in fragment[server_start:]

    # And the format itself carries nothing that can hold a query string.
    sanitized = [fmt for fmt in _log_formats(dump) if "osprey_sanitized" in fmt]
    assert len(sanitized) == 1, _log_formats(dump)
    for forbidden in ("$request ", "$args", "$query_string", "$request_uri", "$http_referer"):
        assert forbidden not in sanitized[0], sanitized[0]


def test_no_location_forwards_the_browsers_whole_cookie_jar() -> None:
    """I1: one origin serves every terminal, so the raw `Cookie` header names
    every user that browser has unlocked — and, with auth on, the sidecar's
    origin-wide session too.

    Forwarding it into a container that runs agent-generated code puts those
    credentials outside the perimeter. So no location may pass `$http_cookie`:
    a gated one cuts the header entirely (it authenticates by the injected
    operator secret), and an ungated one narrows it to that user's own session
    cookie. Checked on the resolved config, mixed roster included, because the
    two branches are exactly where a later edit would reintroduce the bare
    forward.
    """
    users = ["alice", "kiosk"]
    secrets = _terminal_secrets(users)
    config = _config(users=users, auth={"method": "password", "allow_insecure_http": True})
    config["modules"]["web_terminals"]["users"] = [
        {"name": "alice", "index": 0},
        {"name": "kiosk", "index": 1, "login": False},
    ]
    artifacts = render_web_terminals(config, terminal_secrets=secrets)
    nginx_conf = artifacts["nginx/nginx.conf"]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        conf_dir = tmp_path / "conf"
        templates_dir = tmp_path / "templates"
        conf_dir.mkdir()
        templates_dir.mkdir()
        (conf_dir / "default.conf").write_text(nginx_conf)
        _write_secret_templates(artifacts, templates_dir)
        secret_env = {terminal_secret_env_var(user): value for user, value in secrets.items()}

        result = _run_nginx_t(
            conf_dir,
            certs_dir=None,
            templates_dir=templates_dir,
            secret_env=secret_env,
            command=("nginx", "-T"),
        )

    assert result.returncode == 0, result.stderr
    dump = result.stdout
    default_conf = _config_file_sections(dump)[_DEFAULT_CONF_PATH]

    assert "$http_cookie" not in dump

    # The gated user authenticates by the header nginx injected, so its cookie
    # is cut outright.
    gated = _location_body(default_conf, "alice")
    assert 'proxy_set_header Cookie "";' in gated

    # The exempt user has no injected header, so its own session cookie is the
    # only gate left and must survive — and nothing else may travel with it.
    exempt = _location_body(default_conf, "kiosk")
    cookie = f"osprey_terminal_session_{_BASE_PORTS['web'] + 1}"
    assert f'proxy_set_header Cookie "{cookie}=$cookie_{cookie}";' in exempt
    assert 'proxy_set_header Cookie "";' not in exempt


def test_the_secret_directory_holds_exactly_the_users_with_a_gated_location() -> None:
    """Every file in /etc/nginx/osprey is included by exactly one location.

    Both directions matter and neither is free. A file with no include is a
    plaintext operator secret materialized inside the nginx container for no
    reader — it widens what any later change to that container (a support shell,
    a sidecar exporter, the tmpfs mode regressing) can reach, and it falsifies
    the property the per-user no-leak tests reason about. An include with no
    file is worse: nginx refuses to start, and the error names a generated path
    rather than the roster.

    Read from the real container, not from the render, because the question is
    what the entrypoint actually put in that directory. The substitution step is
    driven directly here: the image's own entrypoint ``exec``s nginx, which
    would replace the process before anything could look.
    """
    users = ["alice", "kiosk"]
    secrets = _terminal_secrets(users)
    config = _config(users=users, auth={"method": "password", "allow_insecure_http": True})
    config["modules"]["web_terminals"]["users"] = [
        {"name": "alice", "index": 0},
        {"name": "kiosk", "index": 1, "login": False},
    ]
    artifacts = render_web_terminals(config, terminal_secrets=secrets)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        conf_dir = tmp_path / "conf"
        templates_dir = tmp_path / "templates"
        conf_dir.mkdir()
        templates_dir.mkdir()
        (conf_dir / "default.conf").write_text(artifacts["nginx/nginx.conf"])
        _write_secret_templates(artifacts, templates_dir)
        secret_env = {terminal_secret_env_var(user): value for user, value in secrets.items()}

        listing = _run_nginx_t(
            conf_dir,
            certs_dir=None,
            templates_dir=templates_dir,
            secret_env=secret_env,
            entrypoint="sh",
            command=(
                "-c",
                "/docker-entrypoint.d/20-envsubst-on-templates.sh >/dev/null 2>&1; "
                f"ls -1 {_NGINX_ENVSUBST_OUTPUT_DIR}",
            ),
        )
        resolved = _run_nginx_t(
            conf_dir,
            certs_dir=None,
            templates_dir=templates_dir,
            secret_env=secret_env,
            command=("nginx", "-T"),
        )

    assert listing.returncode == 0, listing.stderr
    assert resolved.returncode == 0, resolved.stderr

    materialized = {name for name in listing.stdout.split() if name}
    included = {
        Path(path).name
        for path in re.findall(
            rf"^\s*include ({_NGINX_ENVSUBST_OUTPUT_DIR}/\S+);",
            _config_file_sections(resolved.stdout)[_DEFAULT_CONF_PATH],
            flags=re.MULTILINE,
        )
    }
    # alice is gated and reads her snippet; kiosk is served ungated with the
    # header cleared, so nothing would ever open a file for it.
    assert materialized == included == {"secret-alice.conf"}


def test_every_terminal_location_clears_credentials_nginx_did_not_set() -> None:
    """M4: the only credential reaching an upstream is one this perimeter put there.

    `Authorization` is cleared on every `/u/<user>/` location whatever the auth
    mode — the app's own bearer tier is loopback traffic inside the container and
    never crosses this proxy, so a value arriving here is the client's and means
    nothing. `X-Osprey-Terminal-Secret` is cleared wherever nginx does not
    inject it; leaving it alone there would let anyone who can reach the port
    present the credential the app trusts absolutely.
    """
    users = ["alice", "kiosk"]
    secrets = _terminal_secrets(users)
    config = _config(users=users, auth={"method": "password", "allow_insecure_http": True})
    config["modules"]["web_terminals"]["users"] = [
        {"name": "alice", "index": 0},
        {"name": "kiosk", "index": 1, "login": False},
    ]
    artifacts = render_web_terminals(config, terminal_secrets=secrets)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        conf_dir = tmp_path / "conf"
        templates_dir = tmp_path / "templates"
        conf_dir.mkdir()
        templates_dir.mkdir()
        (conf_dir / "default.conf").write_text(artifacts["nginx/nginx.conf"])
        _write_secret_templates(artifacts, templates_dir)
        secret_env = {terminal_secret_env_var(user): value for user, value in secrets.items()}

        result = _run_nginx_t(
            conf_dir,
            certs_dir=None,
            templates_dir=templates_dir,
            secret_env=secret_env,
            command=("nginx", "-T"),
        )

    assert result.returncode == 0, result.stderr
    default_conf = _config_file_sections(result.stdout)[_DEFAULT_CONF_PATH]

    for user in users:
        body = _location_body(default_conf, user)
        assert 'proxy_set_header Authorization "";' in body

    # Injected here, so not cleared here.
    gated = _location_body(default_conf, "alice")
    assert f'proxy_set_header {TERMINAL_SECRET_HEADER} "";' not in gated
    assert "include /etc/nginx/osprey/secret-alice.conf;" in gated

    # Not injected here, so cleared here.
    exempt = _location_body(default_conf, "kiosk")
    assert f'proxy_set_header {TERMINAL_SECRET_HEADER} "";' in exempt


def test_a_blank_terminal_secret_takes_down_only_that_users_terminal() -> None:
    """M3: an empty operator secret must not stop nginx from starting.

    The compose carrier renders `${VAR:-}`, so a blank entry in the deploy .env
    reaches nginx as a variable that is PRESENT and empty — which envsubst
    resolves to nothing. Unquoted, the snippet then reads
    `proxy_set_header X-Osprey-Terminal-Secret ;`, which nginx refuses to parse:
    every user's terminal is down over one user's blank variable, with an error
    naming a generated file. Quoted, only that user's app sees an empty
    credential and refuses it.
    """
    users = ["alice", "bob"]
    secrets = _terminal_secrets(users)
    artifacts = render_web_terminals(
        _config(users=users, auth={"method": "password", "allow_insecure_http": True}),
        terminal_secrets=secrets,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        conf_dir = tmp_path / "conf"
        templates_dir = tmp_path / "templates"
        conf_dir.mkdir()
        templates_dir.mkdir()
        (conf_dir / "default.conf").write_text(artifacts["nginx/nginx.conf"])
        _write_secret_templates(artifacts, templates_dir)
        # alice's variable is present and EMPTY — exactly what `${VAR:-}` hands
        # the container for a blank .env entry.
        secret_env = {terminal_secret_env_var("alice"): ""}
        secret_env[terminal_secret_env_var("bob")] = secrets["bob"]

        result = _run_nginx_t(
            conf_dir,
            certs_dir=None,
            templates_dir=templates_dir,
            secret_env=secret_env,
        )

    assert result.returncode == 0, result.stderr
