"""The multi-user pages' roster snippets pass the web-terminal lint, or fail it as stated.

A reader pastes these snippets into ``profile.yml``. The pages say how: each
goes under ``config:`` and its keys merge into the ``modules.web_terminals:``
(and ``agent_record:``) entries ``osprey init --preset demo``
already wrote. This module does exactly that for every profile snippet on
``login.rst`` and ``multi-user/index.rst``, resolves the result through the
profile model, and runs the web-stack checks ``osprey profile validate`` and
``osprey build`` run on it: :func:`build_lint_view`, then
:func:`lint_profile_config`, ``login_tiers_key_errors`` and the named-reader
errors and warnings. The checks those commands run on other parts of the
profile (limits, time zone, facility writes, the CANOPY home screen) are not
repeated here. Each snippet must produce the codes listed for it: none, or the
refusal the page names.

Some snippets refine an earlier one (the ``claims_in_id_token`` fragment
extends the OIDC stanza; a role map needs single sign-on). Those cases name the
snippets layered under them, by a line each one carries.

``web_terminals.open_mode_egress`` is not checked here: it reads each render's
``.claude/settings.json``, so the lint raises it only on a built project
(``osprey scaffold web-terminals lint``, ``osprey up``), not on a profile.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.agent_record.config import reader_errors, reader_warnings
from osprey.cli.build_profile import _presets_dir, resolve_build_profile
from osprey.cli.build_profile_deploy import build_lint_view, profile_dispatch_tier
from osprey.cli.build_profile_presets import _load_preset_raw
from osprey.cli.build_profile_schema import login_tiers_key_errors
from osprey.deployment.web_terminals.lint import _nest_dotted, lint_profile_config
from tests.docs.test_web_terminal_quoted_strings import _LOGIN, _MULTI_USER, _block_with, _blocks

_PRESET = "demo"
_PROJECT = "demo"  # `osprey init demo --preset demo`


@dataclass(frozen=True)
class _Case:
    """One snippet, the snippets under it, and the lint codes it must produce."""

    id: str
    page: Path
    needle: str
    expected: tuple[str, ...] = ()
    layers: tuple[tuple[Path, str], ...] = ()
    # `modules.web_terminals` keys the test adds, for a context the page states in
    # prose; `users` entries are appended to the roster rather than replacing it.
    extra: dict[str, Any] = field(default_factory=dict)
    # An app file the snippet names that the preset does not ship.
    extra_apps: tuple[str, ...] = ()


_OIDC = (_LOGIN, "client_id_env: OSPREY_AUTH_OIDC_CLIENT_ID      # names in .env.auth")
_OIDC_ROSTER = (_LOGIN, "access: any                # any roster login opens this card")

_CASES = (
    _Case(
        "login-open-mode",
        _LOGIN,
        "method: none",
        # The preset's carol is on the admin tier; with no login wall,
        # `osprey profile validate` prints this advisory and passes.
        expected=("web_terminals.unauthenticated_privileged_deployment",),
    ),
    _Case("login-password-tls", _LOGIN, "host_cert_dir: /etc/ssl/facility     # host side"),
    _Case("login-oidc", _LOGIN, _OIDC[1]),
    _Case("login-claims-in-id-token", _LOGIN, "claims_in_id_token: true", layers=(_OIDC,)),
    _Case("login-role-from-sso", _LOGIN, "ca-operators: operator", layers=(_OIDC,)),
    _Case(
        "login-shared-app-access",
        _LOGIN,
        '"user:grace@example.org"',
        layers=(_OIDC_ROSTER,),
        # `tier:admin` needs a login on that tier with an `oidc_subject`, and
        # `controls` an app file; the page states both in prose.
        extra={
            "users": [
                {"name": "carol", "index": 3, "tier": "admin", "oidc_subject": "c0ffee00-..."}
            ]
        },
        extra_apps=("controls",),
    ),
    _Case("login-shared-card-any", _LOGIN, _OIDC_ROSTER[1]),
    _Case("login-shared-card-domain", _LOGIN, "access: [domain:example.org]"),
    _Case("login-names-hidden", _LOGIN, "names: hidden"),
    _Case(
        "login-names-hidden-under-token",
        _LOGIN,
        "names: hidden",
        extra={"auth": {"method": "token"}},
        expected=(
            "web_terminals.landing_names_hidden_without_sign_in",
            "web_terminals.unauthenticated_privileged_deployment",
        ),
    ),
    _Case(
        "login-control-identity",
        _LOGIN,
        "control_identity: cjones",
    ),
    _Case(
        "login-tls-in-front",
        _LOGIN,
        "external_origin: https://terminals.example.org",
    ),
    _Case("login-throttle", _LOGIN, "forget_after_s: 300"),
    _Case("index-landing-notices", _MULTI_USER, "data/landing/working-safely.md"),
    _Case(
        "index-history-scope",
        _MULTI_USER,
        "operator: {tier: readwrite, history: all}",
    ),
)


def _merge(base: Any, over: Any) -> Any:
    """*over*'s keys merged into *base*: mappings recurse, anything else replaces.

    A list in a snippet stands as the whole list, as both pages tell the reader:
    a snippet's ``users:`` is the roster, not entries added to the preset's.
    (``extends:`` merges lists across profile files; a reader editing one
    ``profile.yml`` has one ``users:`` key.)
    """
    if not (isinstance(base, dict) and isinstance(over, dict)):
        return over
    merged = dict(base)
    for key, value in over.items():
        merged[key] = _merge(base[key], value) if key in base else value
    return merged


def _snippet(page: Path, needle: str) -> dict[str, Any]:
    """The snippet holding *needle*, as profile keys (``config:`` added when it is implied)."""
    doc = yaml.safe_load(_block_with(page, "yaml", needle))
    assert isinstance(doc, dict)
    return doc if "config" in doc else {"config": doc}


def _profile(case: _Case) -> dict[str, Any]:
    """The preset's profile with the case's snippets merged in, as the pages instruct."""
    raw, _ = _load_preset_raw(_PRESET)
    profile: dict[str, Any] = {**raw, "project_name": _PROJECT, "data": "data"}
    for page, needle in (*case.layers, (case.page, case.needle)):
        profile = _merge(profile, _snippet(page, needle))
    if case.extra:
        wt = profile["config"]["modules.web_terminals"]
        extra = dict(case.extra)
        users = extra.pop("users", [])
        wt = _merge(wt, extra)
        wt["users"] = [*wt["users"], *users]
        profile["config"]["modules.web_terminals"] = wt
    return profile


def _lint(profile: dict[str, Any], repo: Path, extra_apps: tuple[str, ...] = ()) -> list[str]:
    """Every web-stack code and message ``osprey profile validate`` would draw for *profile*."""
    for dirname in ("tiers", "apps"):
        shutil.copytree(_presets_dir() / dirname, repo / dirname)
    for app in extra_apps:
        shutil.copy(repo / "apps" / "logbook.yml", repo / "apps" / f"{app}.yml")
    (repo / "data").mkdir()
    profile_file = repo / "profile.yml"
    profile_file.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")

    build_profile, _ = resolve_build_profile(profile_file, None, ())
    view = build_lint_view(
        build_profile.deploy,
        build_profile.config,
        build_profile.project_name,
        profile_root=repo,
        dispatch_tier=profile_dispatch_tier(build_profile),
    )
    nested = _nest_dotted(view)
    codes = [finding.code for finding in lint_profile_config(view, profile_root=repo)]
    return [
        *codes,
        *login_tiers_key_errors(view),
        *reader_errors(build_profile.agent_record, nested),
        *reader_warnings(build_profile.agent_record, nested),
    ]


def test_the_preset_alone_lints_clean(tmp_path: Path) -> None:
    """The context every case starts from draws nothing of its own."""
    raw, _ = _load_preset_raw(_PRESET)
    assert _lint({**raw, "project_name": _PROJECT, "data": "data"}, tmp_path) == []


@pytest.mark.parametrize("case", [pytest.param(c, id=c.id) for c in _CASES])
def test_a_roster_snippet_lints_as_its_page_says(case: _Case, tmp_path: Path) -> None:
    codes = _lint(_profile(case), tmp_path, case.extra_apps)
    assert sorted(codes) == sorted(case.expected)


def test_every_profile_snippet_on_the_pages_is_a_case() -> None:
    """A new ``modules.web_terminals`` snippet on either page needs a case here."""
    covered = {
        _block_with(page, "yaml", needle)
        for case in _CASES
        for page, needle in (*case.layers, (case.page, case.needle))
    }
    # The preset excerpt and the registry-mode example are not profile snippets;
    # test_web_terminal_quoted_strings and test_web_terminal_image_names hold them.
    snippets = [
        body
        for page in (_LOGIN, _MULTI_USER)
        for body in _blocks(page, "yaml")
        if "modules.web_terminals:" in body
    ]
    assert [body for body in snippets if body not in covered] == []


def test_the_check_catches_a_snippet_the_lint_refuses(tmp_path: Path) -> None:
    """Negative control: a roster entry with a malformed name is reported."""
    case = _Case("control", _LOGIN, "forget_after_s: 300")
    profile = _profile(case)
    profile["config"]["modules.web_terminals"]["users"].append({"name": "Not A Name", "index": 9})
    assert "web_terminals.invalid_username_charset" in _lint(profile, tmp_path)
