"""Unit tests for the OpenObserve root-password entry in the token-mint map.

control_assistant ships telemetry LIVE against a co-deployed OpenObserve store,
so ``osprey up`` must self-provision a ``ZO_ROOT_USER_PASSWORD`` into the
project ``.env`` (``_SERVICE_TOKEN_VARS["openobserve"]``). Two properties make
this entry different from every other minted token and are pinned here:

* OpenObserve refuses to start unless the root password has all four character
  classes (lower/upper/digit/special) — so the mint uses a dedicated recipe
  (``_generate_openobserve_password``), not the ``token_urlsafe`` default whose
  ``[A-Za-z0-9_-]`` alphabet would crash-loop the container non-deterministically
  (the same failure shape as ``BLUESKY_TILED_API_KEY``/Tiled).
* Its compose template carries an insecure ``:-Complexpass#123`` default, so the
  motivation is not fail-closed arming (as with dispatch/bluesky) but replacing a
  shared, transcript-guarding default with a per-deploy secret. That same
  published default is refused outright when an operator sets it — the one
  failure no format rule can express, since the value satisfies OpenObserve's
  policy in full (see ``TestPublishedDefaultIsRefused``).
"""

from __future__ import annotations

import base64
import os
import re
from pathlib import Path

import pytest

from osprey.deployment import container_lifecycle
from osprey.deployment.service_tokens import _VAR_FORBIDDEN_VALUES
from osprey.utils.dotenv import format_env_line

_MINT_SAMPLES = 200


def _parse_dotenv(path):
    from osprey.utils.dotenv import parse_dotenv_file

    return parse_dotenv_file(path) if path.is_file() else {}


def _satisfies_openobserve_policy(pw: str) -> bool:
    return (
        8 <= len(pw) <= 128
        and any(c.islower() for c in pw)
        and any(c.isupper() for c in pw)
        and any(c.isdigit() for c in pw)
        and any(not c.isalnum() for c in pw)
    )


# ---------------------------------------------------------------------------
# Generator — must satisfy OpenObserve's policy on EVERY mint. A single sample
# proves nothing: a broken recipe could still pass by luck. Assert over many.
# ---------------------------------------------------------------------------


def test_openobserve_password_generator_always_satisfies_policy():
    pws = [container_lifecycle._generate_openobserve_password() for _ in range(_MINT_SAMPLES)]

    offenders = [p for p in pws if not _satisfies_openobserve_policy(p)]
    assert not offenders, (
        f"{len(offenders)}/{_MINT_SAMPLES} fail OpenObserve policy: {offenders[:3]}"
    )
    # Its own validator must agree with the policy check on every mint.
    assert all(container_lifecycle._validate_openobserve_password(p) for p in pws)
    # Random, not patterned. This recipe mints deliberately short (see the
    # human-legibility exception in _VAR_GENERATORS), so the bar here is
    # per-deploy uniqueness rather than the module's 256-bit default: ~65 bits
    # over this sample makes a collision astronomically unlikely, and a repeat
    # would mean the recipe is drawing from something other than `secrets`.
    assert len(set(pws)) == _MINT_SAMPLES, "generator is not random"


def test_openobserve_password_uses_only_dotenv_safe_characters():
    """A minted password must survive ``.env`` write/parse verbatim.

    ``# $ " ' `` backslash ``= space`` and control chars would break dotenv
    parsing or shell reuse; the special alphabet deliberately excludes them.
    """
    unsafe = set("#$\"'`\\= \t\n\r")
    for _ in range(_MINT_SAMPLES):
        pw = container_lifecycle._generate_openobserve_password()
        assert not (set(pw) & unsafe), f"minted a dotenv-hostile password: {pw!r}"


def test_openobserve_password_routes_through_its_own_generator():
    """``_generate_token`` consults the registry rather than the default recipe."""
    a = container_lifecycle._generate_token("ZO_ROOT_USER_PASSWORD")
    b = container_lifecycle._generate_token("ZO_ROOT_USER_PASSWORD")
    assert _satisfies_openobserve_policy(a)
    assert a != b  # random, not a constant


# ---------------------------------------------------------------------------
# Mint path (_ensure_service_tokens) — the value that reaches .env is what the
# OpenObserve container parses, so assert the property end-to-end, and prove the
# minted value round-trips through parse_dotenv_file intact.
# ---------------------------------------------------------------------------


def test_deploying_openobserve_mints_a_policy_valid_password_every_time(tmp_path, monkeypatch):
    monkeypatch.delenv("ZO_ROOT_USER_PASSWORD", raising=False)
    config = {"deployed_services": ["openobserve"]}

    for i in range(50):
        env_path = tmp_path / f"{i}.env"
        container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)
        pw = _parse_dotenv(env_path)["ZO_ROOT_USER_PASSWORD"]
        assert _satisfies_openobserve_policy(pw), f"mint {i} would crash-loop OpenObserve: {pw!r}"


def test_openobserve_mint_writes_both_halves_of_the_login(tmp_path, monkeypatch):
    """The password is minted; the account email is written at its default.

    Not because the email is a secret — it is a username — but because a login
    nobody can find is a login nobody can use. Before this, the email existed
    only as a ``:-`` fallback inline in two templates, so an operator grepping
    ``.env`` for their credentials found half a login and no hint where the
    other half lived.
    """
    monkeypatch.delenv("ZO_ROOT_USER_PASSWORD", raising=False)
    monkeypatch.delenv("ZO_ROOT_USER_EMAIL", raising=False)
    config = {"deployed_services": ["openobserve"]}
    env_path = tmp_path / ".env"

    container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)

    env = _parse_dotenv(env_path)
    assert env.get("ZO_ROOT_USER_PASSWORD")
    assert env.get("ZO_ROOT_USER_EMAIL") == "root@example.com"


def test_openobserve_mint_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.delenv("ZO_ROOT_USER_PASSWORD", raising=False)
    config = {"deployed_services": ["openobserve"]}
    env_path = tmp_path / ".env"

    container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)
    first = _parse_dotenv(env_path)["ZO_ROOT_USER_PASSWORD"]
    container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)
    second = _parse_dotenv(env_path)["ZO_ROOT_USER_PASSWORD"]

    assert first == second
    assert env_path.read_text().count("ZO_ROOT_USER_PASSWORD=") == 1


def test_minted_password_yields_a_valid_openobserve_auth_header(tmp_path, monkeypatch):
    """Cross-check the two halves: a minted password flows through the agent's
    telemetry resolver into a clean base64 Basic-auth header (no ${VAR} leak)."""
    from osprey.build.claude_code_telemetry import _openobserve_auth_header

    monkeypatch.delenv("ZO_ROOT_USER_PASSWORD", raising=False)
    env_path = tmp_path / ".env"
    container_lifecycle._ensure_service_tokens(
        {"deployed_services": ["openobserve"]}, expose_network=False, env_path=env_path
    )
    pw = _parse_dotenv(env_path)["ZO_ROOT_USER_PASSWORD"]

    key, value = _openobserve_auth_header(
        {"openobserve": {"user": "root@example.com", "password": pw}}
    )
    assert key == "Authorization"
    decoded = base64.b64decode(value.removeprefix("Basic ")).decode()
    assert decoded == f"root@example.com:{pw}"


# ---------------------------------------------------------------------------
# Deploy-boundary validation of an OPERATOR-SUPPLIED password (never minted).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "weak",
    [
        "alllowercase1!",  # no uppercase
        "ALLUPPERCASE1!",  # no lowercase
        "NoSpecialChars123",  # no special
        "NoDigits!ABCdef",  # no digit
        "aB3!",  # too short (<8)
    ],
)
def test_ensure_service_tokens_rejects_weak_operator_password_from_dotenv(
    tmp_path, monkeypatch, weak
):
    monkeypatch.delenv("ZO_ROOT_USER_PASSWORD", raising=False)
    assert not _satisfies_openobserve_policy(weak), "sample must be genuinely weak"
    config = {"deployed_services": ["openobserve"]}
    env_path = tmp_path / ".env"
    env_path.write_text(f"ZO_ROOT_USER_PASSWORD={weak}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="ZO_ROOT_USER_PASSWORD") as exc_info:
        container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)

    # The offending value is never echoed in the error.
    assert weak not in str(exc_info.value)


def test_ensure_service_tokens_rejects_weak_password_from_process_env(tmp_path, monkeypatch):
    config = {"deployed_services": ["openobserve"]}
    env_path = tmp_path / ".env"
    monkeypatch.setenv("ZO_ROOT_USER_PASSWORD", "nospecialchars123AB")

    with pytest.raises(RuntimeError, match="ZO_ROOT_USER_PASSWORD") as exc_info:
        container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)

    assert "nospecialchars123AB" not in str(exc_info.value)


def test_ensure_service_tokens_accepts_a_freshly_minted_password(tmp_path, monkeypatch):
    """Happy path: an unset password is minted, then validated against the same
    policy — the mint always satisfies its own validator."""
    monkeypatch.delenv("ZO_ROOT_USER_PASSWORD", raising=False)
    config = {"deployed_services": ["openobserve"]}
    env_path = tmp_path / ".env"

    container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)

    assert _satisfies_openobserve_policy(_parse_dotenv(env_path)["ZO_ROOT_USER_PASSWORD"])


def test_ensure_service_tokens_accepts_a_strong_operator_password(tmp_path, monkeypatch):
    monkeypatch.delenv("ZO_ROOT_USER_PASSWORD", raising=False)
    config = {"deployed_services": ["openobserve"]}
    env_path = tmp_path / ".env"
    strong = "MyStr0ng#Facility@Pass"
    env_path.write_text(f"ZO_ROOT_USER_PASSWORD={strong}\n", encoding="utf-8")

    container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)

    assert _parse_dotenv(env_path)["ZO_ROOT_USER_PASSWORD"] == strong  # untouched


def test_service_token_vars_map_includes_openobserve():
    """Lock in the map shape: only the password, not the email."""
    assert container_lifecycle._SERVICE_TOKEN_VARS["openobserve"] == ("ZO_ROOT_USER_PASSWORD",)


# ---------------------------------------------------------------------------
# The account email — a NON-SECRET written for discoverability, which is why it
# lives in its own map rather than in _SERVICE_TOKEN_VARS. That map's contract
# is "machine-generated secret, minted through _VAR_GENERATORS"; a constant
# email registered there would have to declare a generator, and every generator
# is bound by the module's CSPRNG bar. Two concerns, two maps.
# ---------------------------------------------------------------------------


def test_the_written_email_matches_the_fallback_the_template_ships():
    """The default written to ``.env`` must be the value the container would
    otherwise have used, or writing it down CHANGES the login instead of
    recording it.

    Parsed out of the shipped template rather than restated, for the same reason
    ``TestPublishedDefaultIsRefused`` parses the password fallback: a template
    edit that moves the default while this map keeps the old one would silently
    hand operators an email their store does not know.
    """
    import osprey.templates

    template = (
        Path(osprey.templates.__file__).parent
        / "services"
        / "openobserve"
        / "docker-compose.yml.j2"
    )
    shipped = re.search(r"\$\{ZO_ROOT_USER_EMAIL:-([^}]*)\}", template.read_text(encoding="utf-8"))
    assert shipped, f"no ${{ZO_ROOT_USER_EMAIL:-...}} fallback found in {template}"

    written = dict(container_lifecycle._SERVICE_DEFAULT_VARS["openobserve"])
    assert written["ZO_ROOT_USER_EMAIL"] == shipped.group(1)


def test_an_operator_email_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.delenv("ZO_ROOT_USER_PASSWORD", raising=False)
    monkeypatch.delenv("ZO_ROOT_USER_EMAIL", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("ZO_ROOT_USER_EMAIL=ops@facility.example\n", encoding="utf-8")

    container_lifecycle._ensure_service_tokens(
        {"deployed_services": ["openobserve"]}, expose_network=False, env_path=env_path
    )

    assert _parse_dotenv(env_path)["ZO_ROOT_USER_EMAIL"] == "ops@facility.example"
    assert env_path.read_text().count("ZO_ROOT_USER_EMAIL=") == 1


def test_a_stale_empty_export_does_not_outrank_the_written_email(tmp_path, monkeypatch):
    """Writing the line is not enough if the process still says otherwise.

    The deploy loads the project ``.env`` over its own environment before
    reaching here, so a name the file spells empty leaves an empty copy in
    ``os.environ``. That copy outranks the ``--env-file`` line for compose's
    interpolation, so the container would receive the empty value while ``.env``
    shows the default — the account name written down and not delivered. The
    mint repairs exactly this for the secrets it generates; the defaults block
    has to repair it too or it is only half a fix.
    """
    monkeypatch.delenv("ZO_ROOT_USER_PASSWORD", raising=False)
    monkeypatch.setenv("ZO_ROOT_USER_EMAIL", "")
    env_path = tmp_path / ".env"
    env_path.write_text("ZO_ROOT_USER_EMAIL=\n", encoding="utf-8")

    container_lifecycle._ensure_service_tokens(
        {"deployed_services": ["openobserve"]}, expose_network=False, env_path=env_path
    )

    assert _parse_dotenv(env_path)["ZO_ROOT_USER_EMAIL"] == "root@example.com"
    assert os.environ["ZO_ROOT_USER_EMAIL"] == "root@example.com"


def test_the_email_is_written_even_when_the_password_already_exists(tmp_path, monkeypatch):
    """The upgrade path, and the reason this cannot ride along with the mint.

    Every project deployed before the email was written has a ``.env`` carrying
    a password and no email. The mint is idempotent, so it generates nothing on
    that project's next deploy — if writing the email were folded into the
    minted block it would never run, and exactly the deployments that need the
    line would be the ones that never get it.
    """
    monkeypatch.delenv("ZO_ROOT_USER_PASSWORD", raising=False)
    monkeypatch.delenv("ZO_ROOT_USER_EMAIL", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("ZO_ROOT_USER_PASSWORD=MyStr0ng#Facility@Pass\n", encoding="utf-8")

    container_lifecycle._ensure_service_tokens(
        {"deployed_services": ["openobserve"]}, expose_network=False, env_path=env_path
    )

    assert _parse_dotenv(env_path)["ZO_ROOT_USER_EMAIL"] == "root@example.com"


def test_no_email_is_written_when_openobserve_is_not_deployed(tmp_path, monkeypatch):
    """Keyed by deployed service, exactly like the token map."""
    monkeypatch.delenv("ZO_ROOT_USER_EMAIL", raising=False)
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "a" * 40)
    monkeypatch.setenv("DISPATCH_WORKER_TOKEN", "b" * 40)
    env_path = tmp_path / ".env"

    container_lifecycle._ensure_service_tokens(
        {"deployed_services": ["event_dispatcher"]}, expose_network=False, env_path=env_path
    )

    assert "ZO_ROOT_USER_EMAIL" not in _parse_dotenv(env_path)


def test_writing_the_email_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.delenv("ZO_ROOT_USER_PASSWORD", raising=False)
    monkeypatch.delenv("ZO_ROOT_USER_EMAIL", raising=False)
    env_path = tmp_path / ".env"
    config = {"deployed_services": ["openobserve"]}

    container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)
    container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)

    assert env_path.read_text().count("ZO_ROOT_USER_EMAIL=") == 1


# ---------------------------------------------------------------------------
# Password SHAPE. This store is a localhost demo backend whose password a human
# reads off a terminal and types into a browser login form — the only minted
# credential in the system with a human in the loop. The length is chosen for
# that, and bounded by the expose-time refusal below.
# ---------------------------------------------------------------------------


def test_minted_password_is_short_enough_to_type():
    for _ in range(_MINT_SAMPLES):
        assert len(container_lifecycle._generate_openobserve_password()) == 12


def test_minted_password_avoids_visually_ambiguous_characters():
    """``l I 1 O 0`` are the characters people transcribe wrongly off a screen.

    A password you cannot read back is as bad as one you cannot type, and a
    failed OpenObserve login says nothing about which character was misread.
    """
    ambiguous = set("lI1O0")
    for _ in range(_MINT_SAMPLES):
        pw = container_lifecycle._generate_openobserve_password()
        assert not (set(pw) & ambiguous), f"minted an easily-misread password: {pw!r}"


# ---------------------------------------------------------------------------
# Off-host exposure raises the bar. Minting is idempotent, so a short password
# minted on a loopback deploy SURVIVES a later `osprey up --expose` — scoping
# the recipe to expose_network at mint time would be false comfort. The refusal
# is what makes the short mint safe: it reads the effective value on every
# exposed deploy, whatever its origin.
# ---------------------------------------------------------------------------


def test_an_exposed_deploy_refuses_the_short_demo_password(tmp_path, monkeypatch):
    monkeypatch.delenv("ZO_ROOT_USER_PASSWORD", raising=False)
    monkeypatch.delenv("ZO_ROOT_USER_EMAIL", raising=False)
    env_path = tmp_path / ".env"
    config = {"deployed_services": ["openobserve"]}
    # Mint on loopback, the way a demo deploy does.
    container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)
    minted = _parse_dotenv(env_path)["ZO_ROOT_USER_PASSWORD"]

    with pytest.raises(RuntimeError, match="ZO_ROOT_USER_PASSWORD") as exc_info:
        container_lifecycle._ensure_service_tokens(config, expose_network=True, env_path=env_path)

    assert minted not in str(exc_info.value)
    assert "reachable off-host" in str(exc_info.value)


def test_an_exposed_deploy_accepts_a_long_operator_password(tmp_path, monkeypatch):
    monkeypatch.delenv("ZO_ROOT_USER_PASSWORD", raising=False)
    monkeypatch.delenv("ZO_ROOT_USER_EMAIL", raising=False)
    env_path = tmp_path / ".env"
    strong = "Str0ng@Facility%Password%ForRealDeployments"
    env_path.write_text(f"ZO_ROOT_USER_PASSWORD={strong}\n", encoding="utf-8")

    container_lifecycle._ensure_service_tokens(
        {"deployed_services": ["openobserve"]}, expose_network=True, env_path=env_path
    )

    assert _parse_dotenv(env_path)["ZO_ROOT_USER_PASSWORD"] == strong


def test_a_loopback_deploy_accepts_the_short_demo_password(tmp_path, monkeypatch):
    """The refusal must not fire on the ordinary demo deploy it exists to allow."""
    monkeypatch.delenv("ZO_ROOT_USER_PASSWORD", raising=False)
    monkeypatch.delenv("ZO_ROOT_USER_EMAIL", raising=False)
    env_path = tmp_path / ".env"
    config = {"deployed_services": ["openobserve"]}

    container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)
    container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)

    assert len(_parse_dotenv(env_path)["ZO_ROOT_USER_PASSWORD"]) == 12


def test_the_exposure_bar_is_opt_in_per_var(tmp_path, monkeypatch):
    """A short token on an unregistered var is not this rule's business.

    The dispatch tokens are machine-to-machine and carry their own recipes; the
    length bar exists for the one credential a human shortened on purpose.
    """
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "short1")
    monkeypatch.setenv("DISPATCH_WORKER_TOKEN", "short2")

    container_lifecycle._ensure_service_tokens(
        {"deployed_services": ["event_dispatcher"]},
        expose_network=True,
        env_path=tmp_path / ".env",
    )


# ---------------------------------------------------------------------------
# The published compose default is refused outright — the one failure no format
# rule can express, because that value satisfies OpenObserve's policy in full.
# ---------------------------------------------------------------------------


class TestPublishedDefaultIsRefused:
    """A password that is public is not a password, however well-formed.

    ``ZO_ROOT_USER_PASSWORD`` is interpolated in the compose template with a
    ``:-`` fallback, so a project that never sets it still starts — on a value
    printed in every rendered copy of the template, guarding a store that holds
    full agent conversation transcripts. The mint exists to replace that value,
    but nothing stopped an operator from writing it back into ``.env`` by hand,
    and it passes every character-class check there is.

    The refusal is deliberately narrow. It reads the *effective* value at the
    deploy boundary, after the mint, so it can only fire on a value an operator
    supplied — never on the fresh-repo path, where the template default resolves
    transiently and the mint replaces it before this check ever runs.
    """

    def test_the_registered_default_is_the_one_the_template_ships(self):
        """The declaration must track the template, or the refusal guards nothing.

        Parse the fallback out of the shipped compose template rather than
        restating it: a template edit that changes the published default while
        this map keeps the old one would leave the new default accepted, and
        nothing else in the suite would notice.
        """
        import osprey.templates

        template = (
            Path(osprey.templates.__file__).parent
            / "services"
            / "openobserve"
            / "docker-compose.yml.j2"
        )
        shipped = re.search(
            r"\$\{ZO_ROOT_USER_PASSWORD:-([^}]*)\}", template.read_text(encoding="utf-8")
        )
        assert shipped, f"no ${{ZO_ROOT_USER_PASSWORD:-...}} fallback found in {template}"

        assert shipped.group(1) in _VAR_FORBIDDEN_VALUES["ZO_ROOT_USER_PASSWORD"]

    def test_the_published_default_would_pass_every_format_check(self):
        """Why this needs its own mechanism rather than another validator."""
        for published in _VAR_FORBIDDEN_VALUES["ZO_ROOT_USER_PASSWORD"]:
            assert _satisfies_openobserve_policy(published)
            assert container_lifecycle._validate_var("ZO_ROOT_USER_PASSWORD", published)

    @pytest.mark.parametrize("published", sorted(_VAR_FORBIDDEN_VALUES["ZO_ROOT_USER_PASSWORD"]))
    def test_an_operator_supplied_default_refuses_the_deploy(
        self, tmp_path, monkeypatch, published
    ):
        monkeypatch.delenv("ZO_ROOT_USER_PASSWORD", raising=False)
        env_path = tmp_path / ".env"
        env_path.write_text(format_env_line("ZO_ROOT_USER_PASSWORD", published) + "\n")
        # The value must survive the .env round-trip, or the test would be
        # asserting the refusal fires on something the operator never set.
        assert _parse_dotenv(env_path)["ZO_ROOT_USER_PASSWORD"] == published

        with pytest.raises(RuntimeError) as exc_info:
            container_lifecycle._ensure_service_tokens(
                {"deployed_services": ["openobserve"]}, expose_network=False, env_path=env_path
            )

        assert str(exc_info.value) == (
            "ZO_ROOT_USER_PASSWORD is invalid: must not be the default published in the "
            "openobserve compose template — that value ships in every rendered project, "
            "so it is a shared password rather than a secret, and this store holds full "
            "agent conversation transcripts; remove the line from .env (and unset any "
            "shell export of the same name) so the next run mints a per-deploy value. "
            "Refusing to deploy. (Value not shown.)"
        )
        # Names only: the refusal must not echo the credential it rejected.
        assert published not in str(exc_info.value)

    def test_a_shell_export_of_the_default_refuses_too(self, tmp_path, monkeypatch):
        """Process env wins over ``.env``, so it is just as much "set to the default"."""
        published = next(iter(_VAR_FORBIDDEN_VALUES["ZO_ROOT_USER_PASSWORD"]))
        monkeypatch.setenv("ZO_ROOT_USER_PASSWORD", published)

        with pytest.raises(RuntimeError, match="ZO_ROOT_USER_PASSWORD") as exc_info:
            container_lifecycle._ensure_service_tokens(
                {"deployed_services": ["openobserve"]},
                expose_network=False,
                env_path=tmp_path / ".env",
            )

        assert published not in str(exc_info.value)

    def test_a_fresh_repo_mints_and_does_not_refuse(self, tmp_path, monkeypatch):
        """The narrowness this check is designed around.

        Nothing in the process env, nothing on disk — the state every first
        deploy starts from, and the state in which the template's ``:-``
        fallback is what the value *would* resolve to. The mint runs first, so
        the effective value is a per-deploy secret and the deploy proceeds. A
        build-time twin of this refusal would fire here, on every new project,
        which is why there is none.
        """
        monkeypatch.delenv("ZO_ROOT_USER_PASSWORD", raising=False)
        env_path = tmp_path / ".env"
        assert not env_path.exists()

        minted = container_lifecycle._ensure_service_tokens(
            {"deployed_services": ["openobserve"]}, expose_network=False, env_path=env_path
        )

        assert minted == {"ZO_ROOT_USER_PASSWORD"}
        assert (
            _parse_dotenv(env_path)["ZO_ROOT_USER_PASSWORD"]
            not in _VAR_FORBIDDEN_VALUES["ZO_ROOT_USER_PASSWORD"]
        )

    def test_a_second_run_over_a_minted_value_still_does_not_refuse(self, tmp_path, monkeypatch):
        """Idempotence survives the new check: a minted value is not forbidden."""
        monkeypatch.delenv("ZO_ROOT_USER_PASSWORD", raising=False)
        env_path = tmp_path / ".env"
        config = {"deployed_services": ["openobserve"]}

        container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)
        minted = _parse_dotenv(env_path)["ZO_ROOT_USER_PASSWORD"]
        container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)

        assert _parse_dotenv(env_path)["ZO_ROOT_USER_PASSWORD"] == minted

    def test_the_generator_can_never_mint_a_forbidden_value(self):
        """The mint and the refusal must not be able to disagree."""
        for _ in range(_MINT_SAMPLES):
            assert not container_lifecycle._is_forbidden_value(
                "ZO_ROOT_USER_PASSWORD", container_lifecycle._generate_openobserve_password()
            )

    def test_only_the_registered_var_has_a_forbidden_value(self, tmp_path, monkeypatch):
        """Opt-in per var, exactly like ``_VAR_VALIDATORS``.

        The same string set on an unregistered var is nobody's published
        default, so it passes — the check is a named-value refusal, not a
        password-strength opinion applied everywhere.
        """
        published = next(iter(_VAR_FORBIDDEN_VALUES["ZO_ROOT_USER_PASSWORD"]))
        assert not container_lifecycle._is_forbidden_value("EVENT_DISPATCHER_TOKEN", published)

        monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", published)
        monkeypatch.setenv("DISPATCH_WORKER_TOKEN", published)
        container_lifecycle._ensure_service_tokens(
            {"deployed_services": ["event_dispatcher"]},
            expose_network=False,
            env_path=tmp_path / ".env",
        )


# ---------------------------------------------------------------------------
# Volume-continuity warning — the one thing a re-mint cannot fix by itself
# ---------------------------------------------------------------------------


class TestVolumeContinuityReporting:
    """A re-minted password is silently rejected by a volume that already exists.

    OpenObserve and Postgres read their root credential ONLY when they
    initialize a fresh data volume. Every other minted token takes effect on the
    next restart; these two do not. So the dangerous moment is an operator who
    lost ``.env`` — or just deleted its minted section — while the volumes
    lived: the next deploy mints a NEW secret, the surviving volume keeps
    ignoring it, and the failure surfaces as "I cannot log in" with nothing
    anywhere connecting that to the mint.

    Neither variable can be guarded by compose — both are interpolated with a
    ``:-`` DEFAULT in their templates, so no ``:?`` guard aborts the deploy and
    says why. What closes the gap instead is
    ``_preflight_stale_store_volumes``, and the mint's only job is to tell it
    which vars are newly minted. That hand-off is what is pinned here; the
    refusal it feeds is pinned in test_stale_store_volume_preflight.py.
    """

    def test_a_minted_password_is_reported_as_a_volume_continuity_hazard(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("ZO_ROOT_USER_PASSWORD", raising=False)
        config = {"deployed_services": ["openobserve"]}

        minted = container_lifecycle._ensure_service_tokens(
            config, expose_network=False, env_path=tmp_path / ".env"
        )

        assert minted == {"ZO_ROOT_USER_PASSWORD"}

    def test_an_adopted_password_is_not_reported(self, tmp_path, monkeypatch):
        """Nothing was minted, so the volume and the value agree by construction.

        The distinction the whole preflight rests on: only a *new* value can
        disagree with a volume that already exists.
        """
        monkeypatch.delenv("ZO_ROOT_USER_PASSWORD", raising=False)
        # Must satisfy the same complexity constraint a real adopted value would.
        (tmp_path / ".env").write_text("ZO_ROOT_USER_PASSWORD=Preexisting1!\n", encoding="utf-8")

        minted = container_lifecycle._ensure_service_tokens(
            {"deployed_services": ["openobserve"]},
            expose_network=False,
            env_path=tmp_path / ".env",
        )

        assert minted == set()

    def test_the_warning_never_prints_the_value(self, tmp_path, monkeypatch, caplog):
        """Names only, never values — the same rule every other line here follows."""
        monkeypatch.delenv("ZO_ROOT_USER_PASSWORD", raising=False)
        env_path = tmp_path / ".env"

        with caplog.at_level("WARNING"):
            container_lifecycle._ensure_service_tokens(
                {"deployed_services": ["openobserve"]}, expose_network=False, env_path=env_path
            )

        secret = _parse_dotenv(env_path)["ZO_ROOT_USER_PASSWORD"]
        assert secret
        assert secret not in caplog.text

    def test_a_password_that_was_already_there_draws_no_warning(
        self, tmp_path, monkeypatch, caplog
    ):
        """The warning is about the MINT, not about the variable.

        A deploy that finds an existing value changes nothing, so there is
        nothing for a volume to disagree with. Warning here would train the
        operator to ignore it on every single deploy, which is the same as not
        having it.
        """
        monkeypatch.setenv("ZO_ROOT_USER_PASSWORD", "Alr3ady#Set!")

        with caplog.at_level("WARNING"):
            container_lifecycle._ensure_service_tokens(
                {"deployed_services": ["openobserve"]},
                expose_network=False,
                env_path=tmp_path / ".env",
            )

        assert "fresh data volume" not in caplog.text

    def test_an_ordinary_token_draws_no_volume_warning(self, tmp_path, monkeypatch, caplog):
        """Only the two volume-initialized vars qualify.

        A dispatch token is read from the environment on every start, so a
        re-mint takes effect immediately and there is nothing to warn about.
        """
        for var in ("EVENT_DISPATCHER_TOKEN", "DISPATCH_WORKER_TOKEN"):
            monkeypatch.delenv(var, raising=False)

        with caplog.at_level("WARNING"):
            container_lifecycle._ensure_service_tokens(
                {"deployed_services": ["event_dispatcher"]},
                expose_network=False,
                env_path=tmp_path / ".env",
            )

        assert "fresh data volume" not in caplog.text
